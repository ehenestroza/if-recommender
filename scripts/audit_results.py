#!/usr/bin/env python
"""
Quality-check what the app actually returns for its most prominent inputs.

Spot-checking by hand finds the obvious cases and misses the rare ones. This
walks the top N entries of the game, author and reviewer pickers — the ones a
visitor is most likely to click first — collects the first page of results for
each, and reports anything that would look wrong on a card: missing fields, or
tags whose meaning says the entry is not a game someone can play.

It reports rather than fixes. Deciding that a tag means "exclude this" is a
judgement call, and a bad exclusion silently removes real games, so the output
is meant to be read before anything is added to `drop_non_games`.

Usage
-----
    uv run scripts/audit_results.py                 # top 50, first 20 results
    uv run scripts/audit_results.py --top 100
    uv run scripts/audit_results.py --json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Tags that, read plainly, say the entry is not a playable game. Kept separate
# from the "malformed" checks below because these are a judgement about meaning,
# not about data hygiene, and each one is reported with its games so a human can
# confirm before it becomes an exclusion rule.
SUSPECT_TAG_PATTERNS = {
    "not-a-game": r"^not a game$",
    "tool/utility": r"^(tool|utility|utilities|authoring system|development system)$",
    "library/extension": r"^(library|extension|code library|inform 7 extension)$",
    "index/catalogue": r"^(index|catalog|catalogue|directory|list of games)$",
    "review/essay": r"^(review|reviews|essay|article|criticism)$",
    "unplayable": r"^(unfinished|incomplete|abandoned|broken|non-working)$",
    "template/sample": r"^(template|sample|test|example|placeholder)$",
    "collection": r"^(collection|anthology|compilation)$",
}

# Values that read as absent even though the field is technically populated.
PLACEHOLDER = {"", "-", "--", "n/a", "na", "none", "unknown", "untitled", "?", "tbd"}


def _parts(value) -> list:
    return [p.strip() for p in str(value or "").split(",") if p.strip()]


def _blank(value) -> bool:
    return str(value or "").strip().lower() in PLACEHOLDER


def audit(app, modes, top_n, page_size):
    """Collect the first page for each top input and index every issue found."""
    choices = {
        "game": app.GAME_CHOICES,
        "author": app.AUTHOR_CHOICES,
        "reviewer": app.USER_CHOICES,
    }
    defaults = app.FILTER_DEFAULTS
    init = {"results": [], "scored": [], "relevance": {}, "query_key": None,
            "page": 0, "per_page": page_size}

    surfaced = {}            # gameid -> set of "mode:label" that surfaced it
    queries_run = 0
    empty_queries = []

    for mode in modes:
        for label, key in choices[mode][:top_n]:
            kw = {"game": None, "author": None, "user": None}
            kw[{"game": "game", "author": "author", "reviewer": "user"}[mode]] = key
            out = app.recommend(init, mode, kw["game"], kw["author"], kw["user"],
                                [], [], *defaults, page_size)
            queries_run += 1
            results = out[3].get("results", [])[:page_size]
            if not results:
                empty_queries.append(f"{mode}:{label[:40]}")
            for gid, _ in results:
                surfaced.setdefault(gid, set()).add(f"{mode}:{label[:34]}")

    issues = defaultdict(list)
    tag_counter = Counter()
    compiled = {name: re.compile(p, re.I) for name, p in SUSPECT_TAG_PATTERNS.items()}

    for gid, seen_in in surfaced.items():
        row = app.META.loc[gid].to_dict() if gid in app.META.index else {}
        title = str(row.get("title", ""))
        record = {"gameid": gid, "title": title,
                  "author": str(row.get("author", "")),
                  "seen_in": sorted(seen_in)[:3]}

        if _blank(title):
            issues["empty title"].append(record)
        if _blank(row.get("author")):
            issues["empty author"].append(record)
        if _blank(row.get("year")):
            issues["missing year"].append(record)
        if not _parts(row.get("tags")):
            issues["no tags"].append(record)
        if _blank(row.get("system")):
            issues["no system"].append(record)
        if _blank(row.get("description")):
            issues["no description"].append(record)

        for tag in _parts(row.get("tags")):
            tag_counter[tag.lower()] += 1
            for name, rx in compiled.items():
                if rx.match(tag.strip()):
                    issues[f"suspect tag: {name}"].append({**record, "tag": tag})

    return {"queries_run": queries_run, "distinct_games": len(surfaced),
            "empty_queries": empty_queries, "issues": issues,
            "tag_counter": tag_counter, "surfaced": surfaced}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--modes", default="game,author,reviewer")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    import app  # heavy: loads every artefact the web app uses

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    report = audit(app, modes, args.top, args.page_size)

    if args.json:
        print(json.dumps({
            "queries_run": report["queries_run"],
            "distinct_games": report["distinct_games"],
            "empty_queries": report["empty_queries"],
            "issues": {k: v for k, v in report["issues"].items()},
        }, indent=2))
        return

    print("\n" + "=" * 78)
    print(f"  Audited {report['queries_run']} queries "
          f"({args.top} per mode: {', '.join(modes)}), first {args.page_size} results each")
    print(f"  {report['distinct_games']} distinct games surfaced")
    print("=" * 78)

    if report["empty_queries"]:
        print(f"\n  QUERIES RETURNING NOTHING ({len(report['empty_queries'])}):")
        for q in report["empty_queries"][:10]:
            print(f"    {q}")

    if not report["issues"]:
        print("\n  No issues found.")
        return

    for name in sorted(report["issues"], key=lambda k: -len(report["issues"][k])):
        rows = report["issues"][name]
        unique = {r["gameid"]: r for r in rows}
        print(f"\n  {name.upper()} — {len(unique)} game(s)")
        for r in list(unique.values())[:8]:
            extra = f"  [tag: {r['tag']}]" if "tag" in r else ""
            print(f"    {r['title'][:44]:<46} by {r['author'][:24]:<26}{extra}")
            print(f"      surfaced by: {', '.join(r['seen_in'])}")
        if len(unique) > 8:
            print(f"    … and {len(unique) - 8} more")


if __name__ == "__main__":
    main()
