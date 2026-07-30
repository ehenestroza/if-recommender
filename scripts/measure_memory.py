#!/usr/bin/env python
"""
Measure the app's resident memory, stage by stage.

Written to answer one deployment question: how much RAM does a VM need? The
numbers are not obvious from disk sizes — the precomputed ranking tables are
78 MB of Parquet that become roughly ten times that as Python dicts, and live
scoring grows the allocator arena well past what the objects themselves occupy.

Run it on the target machine. Measurements taken on a developer laptop do not
transfer: glibc's allocator fragments differently from macOS's, and the arena
growth measured here is exactly the part that differs.

Usage
-----
    uv run scripts/measure_memory.py                # ~5 min
    uv run scripts/measure_memory.py --queries 80   # more marginal-cost samples
    uv run scripts/measure_memory.py --json         # machine-readable

What it reports
---------------
startup     what the process costs before serving anything
marginal    what each new distinct `vibe` query adds, as it decays
retained    whether that growth is the result cache or reusable arena

The `retained` line is the one that matters: if clearing the cache gives the
memory back, the cache is the thing to bound. If it does not, the cache is
irrelevant to sizing and shrinking it saves nothing.
"""

import argparse
import gc
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rss_mb() -> float:
    """Resident set size in MB. /proc on Linux, ps elsewhere."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024      # kB → MB
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure app resident memory")
    parser.add_argument("--queries", type=int, default=40,
                        help="Distinct vibe queries per sampling block (default: 40)")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    report: dict = {"platform": sys.platform, "python": sys.version.split()[0]}
    log = (lambda *a: None) if args.json else print

    baseline = rss_mb()
    report["baseline_mb"] = round(baseline, 1)
    log(f"\n  {'bare python':<38}{baseline:8.1f} MB")

    import faiss, gradio, pandas, sentence_transformers, torch  # noqa: F401
    libs = rss_mb()
    report["libraries_mb"] = round(libs, 1)
    log(f"  {'+ torch / sentence-transformers / gradio':<38}{libs:8.1f} MB"
        f"  (+{libs - baseline:.1f})")

    # Importing the app loads the models, the FAISS index and the ranking tables.
    try:
        import app
    except FileNotFoundError as exc:
        print(f"\n  Cannot load the app: {exc}\n"
              f"  models/, outputs/ and data/ must be present.", file=sys.stderr)
        raise SystemExit(1)

    startup = rss_mb()
    report["startup_mb"] = round(startup, 1)
    log(f"  {'+ models, index, ranking tables':<38}{startup:8.1f} MB"
        f"  (+{startup - libs:.1f})")
    log(f"\n  startup total: {startup:.0f} MB\n")

    # Live scoring. Only `vibe` scores at request time, so it is the whole of the
    # dynamic cost; the other three modes are served from the lookup tables.
    systems = [s for _, s in app.SYSTEM_CHOICES[: max(8, args.queries // 4)]]
    tags = [t for _, t in app.TAG_CHOICES[: max(16, args.queries // 2)]]
    combos = list(itertools.product(systems, tags))
    if len(combos) < args.queries * 2:
        raise SystemExit(f"  Need {args.queries * 2} distinct queries, "
                         f"only {len(combos)} combinations available.")

    blocks = []
    previous = startup
    for index, label in ((0, "warm-up"), (1, "fresh"), (2, "fresh")):
        lo = index * args.queries
        for system, tag in combos[lo: lo + args.queries]:
            app._score_browse((system,), (tag,))
        gc.collect()
        now = rss_mb()
        per = (now - previous) / args.queries
        blocks.append({"label": label, "queries": lo + args.queries,
                       "rss_mb": round(now, 1), "per_query_mb": round(per, 2)})
        log(f"  {f'{label} queries {lo + 1}-{lo + args.queries}':<38}{now:8.1f} MB"
            f"  ({per:+.2f} MB/query)")
        previous = now
    report["blocks"] = blocks

    # Is the growth the cache, or the allocator? Clearing the cache answers it.
    before_clear = rss_mb()
    app._score_browse.cache_clear()
    gc.collect()
    after_clear = rss_mb()
    released = before_clear - after_clear
    report["cache_released_mb"] = round(released, 1)
    log(f"\n  {'released by clearing the cache':<38}{released:8.1f} MB")

    # The widest query the UI can produce: every system, many tags.
    widest_scored, _ = app._score_browse(tuple(systems), tuple(tags))
    gc.collect()
    peak = rss_mb()
    report["widest_query_candidates"] = len(widest_scored)
    report["peak_mb"] = round(peak, 1)
    log(f"  {f'widest query ({len(widest_scored):,} candidates)':<38}{peak:8.1f} MB")

    marginal = blocks[-1]["per_query_mb"]
    report["marginal_mb_per_query"] = marginal
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"\n  {'-' * 52}")
    print(f"  startup                {startup / 1024:5.2f} GB")
    print(f"  after {blocks[-1]['queries']} live queries  {peak / 1024:5.2f} GB")
    print(f"  marginal cost          {marginal:5.2f} MB per new distinct query")
    if released < 50:
        print("\n  The result cache is NOT what grows — clearing it released"
              f" {released:.0f} MB.\n  The growth is reusable allocator arena, so it"
              " flattens with use and\n  BROWSE_CACHE_SIZE is not a memory lever.")
    else:
        print(f"\n  The result cache holds {released:.0f} MB. Lowering"
              " BROWSE_CACHE_SIZE would cap this.")
    print("\n  Growth per new query decays; provision for roughly double the"
          "\n  'after N live queries' figure above and it will not be close.\n")


if __name__ == "__main__":
    main()
