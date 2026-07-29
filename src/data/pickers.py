"""
Choice lists for the four search modes, shared by the web app and the CLI.

Both front-ends must offer the same things under the same labels — a game
reachable in one and not the other would be a bug nobody notices for months —
so the lists are built here rather than in either interface.

Every list is ordered by frequency, so the first entries are the ones most
people are looking for.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data.preprocessor import (
    SYSTEM_GENRE_SEPARATORS,
    build_display_map,
    profile_vocabulary,
)

# Catch-all credits and single-system studios. All three sort to the top by game
# count and all three give poor recommendations — an aggregate of 61 unrelated
# games, or a catalogue so system-specific that almost nothing outside it
# matches. A bad result from the most obvious first click is worse than the entry
# being absent. They remain available as *filters*, which is a different job.
EXCLUDED_AUTHORS = {
    "anonymous",
    "anonymous (first row software publishing inc.)",
    "failbetter games",
}


def game_choices(game_docs: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Games as "Title — Author (Year)", most-reviewed first.

    119 titles are shared by up to five different games, so a bare title would
    make all but one of each unreachable. Author and year disambiguate at no cost
    to typing, and help even where titles are unique.
    """
    frame = game_docs.sort_values("review_count", ascending=False)
    return [
        (f"{r.title} — {r.author}" + (f" ({r.year})" if str(r.year).strip() else ""), r.gameid)
        for r in frame.itertuples(index=False)
    ]


def author_choices(author_profiles: pd.DataFrame) -> List[Tuple[str, str]]:
    """Authors by catalogue size, excluding the catch-all credits."""
    frame = author_profiles[~author_profiles["authorid"].isin(EXCLUDED_AUTHORS)]
    frame = frame.sort_values("game_count", ascending=False)
    return [
        (f"{r.name}  ·  {r.game_count} game{'s' if r.game_count != 1 else ''}", r.authorid)
        for r in frame.itertuples(index=False)
    ]


def reviewer_choices(
    profile_map: Dict[str, str],
    name_map: Dict[str, str],
    reviews: Optional[pd.DataFrame] = None,
) -> List[Tuple[str, str]]:
    """Reviewers by review count, so recognisable names surface first."""
    counts = reviews.groupby("userid").size() if reviews is not None else pd.Series(dtype=int)
    rows = [(name_map.get(uid, uid), uid, int(counts.get(uid, 0))) for uid in profile_map]
    rows.sort(key=lambda r: -r[2])
    return [(f"{name}  ·  {n} reviews", uid) for name, uid, n in rows]


def vocab_choices(
    game_docs: pd.DataFrame,
    n_systems: int = 10_000,
    n_tags: int = 10_000,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Systems and tags for *building* a vibe query: canonically cased for display,
    normalised values underneath.

    Drawn from the `_clean` columns, so competition tags (XYZZY, IFComp, Spring
    Thing) are absent by design — preprocessing strips them, the encoders never
    saw them, and offering one here would send the model a token it cannot
    represent. Use the tag *filter* to narrow by those instead.
    """
    systems, tags = profile_vocabulary(game_docs, n_systems=n_systems, n_tags=n_tags)
    system_casing = build_display_map(game_docs, "system", SYSTEM_GENRE_SEPARATORS)
    tag_casing = build_display_map(game_docs, "tags")
    return (
        [(system_casing.get(s, (s, 0))[0], s) for s in systems],
        [(tag_casing.get(t, (t, 0))[0], t) for t in tags],
    )
