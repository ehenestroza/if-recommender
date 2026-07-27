"""
Naming convention for original vs. model-facing column variants.

IFDB values are normalised before they reach the encoders — lowercased, split and
deduplicated, competition tags dropped, version numbers stripped — but anything
shown to a user has to match what ifdb.org shows, or the results look wrong.

So originals keep their own names (`author`, `system`, `tags`) and every
normalised variant is written alongside under a `_clean` suffix.  Display code
reads the plain names; retrieval, ranking, filtering, and profile building read
the `_clean` ones.
"""

from typing import Iterable

CLEAN_SUFFIX = "_clean"

# Columns of `game_docs` that exist in both an original and a normalised form.
CLEANED_FIELDS = ("author", "system", "tags")


def clean_col(field: str) -> str:
    """Name of the normalised variant of `field`."""
    return f"{field}{CLEAN_SUFFIX}"


def clean_col_in(columns: Iterable[str], field: str) -> str:
    """
    Normalised variant of `field` if present, else `field` itself.

    The fallback keeps Parquet files written before this split usable.
    """
    columns = set(columns)
    candidate = clean_col(field)
    return candidate if candidate in columns else field


def clean_value(info: dict, field: str) -> str:
    """Read `field`'s normalised value from a game-info dict, original as fallback."""
    value = info.get(clean_col(field))
    if value is None:
        value = info.get(field, "")
    return "" if value is None else str(value)


def split_clean(info: dict, field: str) -> set:
    """`field`'s normalised value as a set of lowercase comma-separated parts."""
    return {part.strip().lower() for part in clean_value(info, field).split(",") if part.strip()}
