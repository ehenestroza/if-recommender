#!/usr/bin/env python
"""Turn exp_vibe_depth.py's JSON into the quality / consistency / cost tables."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

_spec = importlib.util.spec_from_file_location("exp", _HERE / "exp_vibe_depth.py")
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

REFERENCE = "post:None"


def _frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("uid")


def _label(key: str) -> str:
    order, cap = key.split(":")
    if cap == "None":
        return "full pool (today)"
    return f"top {int(cap):,} ({order}-filter)"


def _sort_key(key: str):
    order, cap = key.split(":")
    return (order != "post", float("inf") if cap == "None" else int(cap))


def report(shape_result: dict, filters: list) -> None:
    shape = shape_result["shape"]
    per_query = shape_result["per_query"]
    keys = sorted(per_query, key=_sort_key)
    frames = {k: _frame(per_query[k]) for k in keys}
    ref = frames[REFERENCE]
    n = len(ref)

    print("\n" + "=" * 96)
    print(f"  SHAPE: {shape}  ({shape_result['n_systems']} system(s), "
          f"{shape_result['n_tags']} tags)   n = {n} held-out users")
    print("=" * 96)
    print(f"  raw pool (cosine >= floor):  median {shape_result['raw_pool']['p50']:.0f}   "
          f"p90 {shape_result['raw_pool']['p90']:.0f}   "
          f"p99 {shape_result['raw_pool']['p99']:.0f}   "
          f"max {shape_result['raw_pool']['max']:.0f}")
    print(f"  after tag pre-filter:        median {shape_result['tag_kept_pool']['p50']:.0f}   "
          f"p90 {shape_result['tag_kept_pool']['p90']:.0f}   "
          f"p99 {shape_result['tag_kept_pool']['p99']:.0f}   "
          f"max {shape_result['tag_kept_pool']['max']:.0f}")

    # ---- quality ---------------------------------------------------------
    print("\n  QUALITY vs held-out positives  (Δ = variant − full pool, 95% paired bootstrap CI)")
    print(f"  {'variant':<24} {'R@10':>7} {'R@25':>7} {'NDCG@10':>8} {'NDCG@25':>8} "
          f"{'MRR':>7}   {'ΔNDCG@10 [95% CI]':>26}")
    print("  " + "-" * 92)
    for k in keys:
        f = frames[k]
        d, lo, hi = exp.paired_bootstrap(f["ndcg@10"], ref["ndcg@10"])
        ci = "—" if k == REFERENCE else f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        star = "" if k == REFERENCE or (lo <= 0 <= hi) else "  *"
        print(f"  {_label(k):<24} {f['recall@10'].mean():>7.4f} {f['recall@25'].mean():>7.4f} "
              f"{f['ndcg@10'].mean():>8.4f} {f['ndcg@25'].mean():>8.4f} "
              f"{f['mrr'].mean():>7.4f}   {ci:>26}{star}")
    print("  * = interval excludes zero")

    # ---- consistency -----------------------------------------------------
    print("\n  CONSISTENCY vs the full-pool ranking (what today's users see)")
    print(f"  {'variant':<24} {'ovlp@10':>8} {'ovlp@25':>8} {'top-1 same':>11} "
          f"{'mean |Δrank|':>13} {'% ovlp@25=1.0':>14}")
    print("  " + "-" * 84)
    for k in keys:
        f = frames[k]
        print(f"  {_label(k):<24} {f['overlap@10'].mean():>8.3f} {f['overlap@25'].mean():>8.3f} "
              f"{f['top1_same'].mean():>11.3f} {f['displacement@25'].mean():>13.2f} "
              f"{(f['overlap@25'] >= 0.999).mean() * 100:>13.1f}%")

    # ---- consistency under filtering ------------------------------------
    print("\n  CONSISTENCY UNDER FILTERING  (overlap@25 vs full pool, and page depth)")
    header = f"  {'variant':<24}"
    for name in filters:
        header += f" {name[:20]:>22}"
    header += f" {'unfiltered depth':>17} {'% pages < 25':>13}"
    print(header)
    print("  " + "-" * (24 + 22 * len(filters) + 33))
    for k in keys:
        f = frames[k]
        line = f"  {_label(k):<24}"
        for name in filters:
            ov = f[f"foverlap@25::{name}"].mean()
            short = (f[f"fdepth::{name}"] < 25).mean() * 100
            line += f" {ov:>13.3f} /{short:>5.0f}%"
        line += f" {f['depth'].median():>17.0f} {(f['depth'] < 25).mean() * 100:>12.1f}%"
        print(line)
    print("    each filter cell is  overlap@25 / % of queries left with fewer than 25 results")

    # ---- cost ------------------------------------------------------------
    print("\n  COST  (cross-encoder pairs actually scored per query)")
    print(f"  {'variant':<24} {'median':>8} {'p90':>8} {'p99':>8} {'max':>8} "
          f"{'mean':>8}   {'vs full':>8}")
    print("  " + "-" * 80)
    ref_mean = ref["n_pairs"].mean()
    for k in keys:
        f = frames[k]
        m = f["n_pairs"].mean()
        print(f"  {_label(k):<24} {f['n_pairs'].median():>8.0f} "
              f"{np.percentile(f['n_pairs'], 90):>8.0f} "
              f"{np.percentile(f['n_pairs'], 99):>8.0f} {f['n_pairs'].max():>8.0f} "
              f"{m:>8.0f}   {m / ref_mean:>7.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/vibe_depth.json")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    filters = list(exp.FILTERS)
    for shape_result in results.values():
        report(shape_result, filters)
    print()


if __name__ == "__main__":
    main()
