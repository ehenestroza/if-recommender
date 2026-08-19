"""Dynamic int8 quantization for the cross-encoder.

The reranker is the whole of the live CPU cost, and dynamic quantization is the
one speedup that does not change the candidate pool: weights are stored as int8
and activations are quantized per batch at runtime, with no calibration set and
no retraining. Measured over 150 queries and 147,619 scored pairs, it costs
nothing detectable in ranking quality — every confidence interval spans zero.

Whether it is *faster*, though, is a property of the host and not of the model:

    fbgemm   (x86)   2.02x on the deployment VM — 45 -> 92 pairs/s
    qnnpack  (ARM)   0.24x on an Apple M-series laptop — 149 -> 36 pairs/s

That is a 8x spread in opposite directions, which is why `apply` defaults to
consulting the backend rather than trusting a boolean in a config file. A
deployment that moves from an E-series to an Ampere A1 would otherwise quietly
turn a 2x speedup into a 4x slowdown.
"""

import logging
import platform
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

# fbgemm is the x86 backend and the only one measured to be worth enabling;
# qnnpack is its ARM counterpart. Preference order matters on hosts shipping
# both, because fbgemm is substantially better optimised for this shape.
_ENGINE_PREFERENCE = ("fbgemm", "qnnpack")

# Machines fbgemm can actually run on. `supported_engines` reports what the
# wheel was *built* with, not what this CPU can execute, and the two differ on
# exactly the host we deploy to: the Linux aarch64 wheel lists fbgemm, so
# selecting by that list alone force-set an x86 backend on an Ampere A1 and the
# first prepack died with "RuntimeError: unknown architecure" — taking the
# service down at startup rather than merely running slowly. macOS arm64 lists
# only qnnpack, which is why this never showed up in development.
_X86_MACHINES = frozenset({"x86_64", "amd64", "x86", "i386", "i686"})

# Backends where dynamic quantization is known to pay off. "auto" enables
# quantization only here; everywhere else it is skipped rather than guessed at.
_FAST_ENGINES = frozenset({"fbgemm"})


def select_engine() -> Optional[str]:
    """
    Pick a quantized backend, or None if the build has none.

    PyTorch leaves `torch.backends.quantized.engine` unset on some builds, and
    an unset engine fails at the first prepack with a bare "NoQEngine" — so this
    sets it explicitly rather than relying on a default.
    """
    machine = platform.machine().lower()
    preference = _ENGINE_PREFERENCE if machine in _X86_MACHINES else ("qnnpack",)
    supported = list(torch.backends.quantized.supported_engines)
    for engine in preference:
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    logger.warning("No quantized backend available for %s (build supports: %s)",
                   machine, supported)
    return None


def quantize_cross_encoder(cross_encoder) -> bool:
    """
    Replace a `CrossEncoder`'s Linear layers with dynamic int8 equivalents.

    Returns True if the live module was actually converted.

    The assignment target is deliberately `transformer.model`, not the
    `auto_model` alias that reads more naturally. `auto_model` is a property, and
    `nn.Module.__setattr__` intercepts Module values before the property setter
    runs — so assigning there registers a *second, unused* child and silently
    leaves the module that forward() calls in fp32. That failure is invisible:
    inference still works and returns bit-identical scores, so it reads as
    "quantization made no difference" rather than as a bug. The post-condition
    below is what makes it loud instead.
    """
    engine = select_engine()
    if engine is None:
        return False

    # Quantized kernels are CPU-only. The app lets sentence-transformers pick a
    # device, which lands on MPS or CUDA on a development machine and CPU on the
    # deployment VM — so this is reachable locally and never in production.
    # Warn rather than raise: a speedup that cannot be applied is not a reason to
    # refuse to start, but it must not pass silently either.
    device = next(cross_encoder[0].auto_model.parameters()).device
    if device.type != "cpu":
        logger.warning(
            "Skipping int8 quantization: cross-encoder is on %s and quantized "
            "kernels are CPU-only", device,
        )
        return False

    transformer = cross_encoder[0]
    quantized = torch.ao.quantization.quantize_dynamic(
        transformer.auto_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    transformer.model = quantized

    import torch.ao.nn.quantized.dynamic as nnqd
    n_quantized = sum(
        1 for m in transformer.auto_model.modules() if isinstance(m, nnqd.Linear)
    )
    if n_quantized == 0:
        raise RuntimeError(
            "Quantization did not reach the live module — the CrossEncoder "
            "internals have changed and the assignment target needs revisiting"
        )

    logger.info("Quantized %d Linear layers to int8 (engine: %s)", n_quantized, engine)
    return True


def _try(cross_encoder) -> bool:
    """
    Quantize, or log loudly and carry on in fp32.

    Nothing here is worth an outage. This is a speed optimization on a model
    that runs correctly without it, so a backend that turns out to be unusable
    should cost latency, not availability — which is exactly what it cost when
    an x86 backend was selected on ARM and the process exited 1 on every
    restart. The traceback still reaches the log; only the crash is removed.
    """
    try:
        return quantize_cross_encoder(cross_encoder)
    except Exception:
        logger.exception("int8 quantization failed — continuing in fp32")
        return False


def apply(cross_encoder, setting: Union[bool, str]) -> bool:
    """
    Honour a config value of True, False or "auto". Returns True if quantized.

    "auto" is the sane default for a repo deployed to more than one instance
    shape: it quantizes on backends measured to benefit and leaves the model
    alone elsewhere, so the same config.yaml is correct on x86 and on ARM.
    Setting it to True forces quantization regardless — useful for measuring a
    backend before deciding whether to add it to `_FAST_ENGINES`.
    """
    if setting is False or setting is None:
        return False

    if setting is True:
        return _try(cross_encoder)

    if str(setting).lower() != "auto":
        raise ValueError(
            f"quantize_reranker must be true, false or 'auto', got {setting!r}"
        )

    engine = select_engine()
    if engine not in _FAST_ENGINES:
        logger.info(
            "Skipping int8 quantization: backend %s is not one that benefits "
            "(auto mode). Set quantize_reranker: true to force it.",
            engine or "none",
        )
        return False
    return _try(cross_encoder)
