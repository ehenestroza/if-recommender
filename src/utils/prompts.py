"""
Typeahead prompts for the terminal front-end.

The web app picks games, authors and reviewers from dropdowns; the CLI used to
demand a raw ID pasted from a URL. These give the terminal the same affordance:
type a fragment, see matching entries filtered live, press Enter.

Matching is case-insensitive substring, the same rule the filters use, so
"photopia" and "cadre" both find the same game.

prompt_toolkit is a hard dependency, but every prompt degrades to plain input()
if it is missing, so the pipeline still runs in a bare environment.
"""

from typing import List, Optional, Sequence, Tuple

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.formatted_text import HTML
    HAS_PROMPT_TOOLKIT = True
except ImportError:  # pragma: no cover - exercised only in bare environments
    HAS_PROMPT_TOOLKIT = False

# How many matches the fallback prompt lists before asking people to narrow down.
FALLBACK_MATCHES = 12

# Navigation words, honoured at every prompt so that going back and getting out
# never depend on which prompt you happen to be standing in. Both are checked
# before the choice list, so a game called "Back" is unreachable by typing its
# whole title alone — "back —" or picking it from the menu still works, and a
# reliable way out is worth more than that one title.
BACK_WORDS = {"back", "b"}
QUIT_WORDS = {"quit", "exit", "q"}

Choice = Tuple[str, str]   # (label shown, value returned)


class Cancelled(Exception):
    """Raised to go up one level: 'back', Ctrl-C or Ctrl-D."""


class Quit(Exception):
    """Raised to leave the program entirely, from any depth."""


def _navigate(text: str) -> None:
    """Raise if `text` is a navigation word rather than a choice."""
    lowered = text.strip().lower()
    if lowered in BACK_WORDS:
        raise Cancelled
    if lowered in QUIT_WORDS:
        raise Quit


def _resolve(text: str, choices: Sequence[Choice]) -> Optional[str]:
    """
    Map typed text to a value: exact label first, then unique-ish substring.

    Falling back to the first substring match matters because the completion menu
    lets people accept a highlighted entry *or* keep typing past it — and someone
    who typed enough to see one obvious candidate should not be told it is
    unrecognised. Choices arrive most-popular-first, so the first match is the
    one they almost certainly meant.
    """
    text = text.strip()
    if not text:
        return None
    lowered = text.lower()
    for label, value in choices:
        if label.lower() == lowered:
            return value
    for label, value in choices:
        if lowered in label.lower():
            return value
    return None


def _session() -> "PromptSession":
    # One session per prompt keeps history from leaking between a game picker and
    # a tag picker, where the previous entry is never a useful suggestion.
    return PromptSession()


def _ask(label: str, choices: Sequence[Choice]) -> str:
    """Read one line, with live completion when prompt_toolkit is available."""
    if not HAS_PROMPT_TOOLKIT:
        return input(f"{label} > ").strip()

    completer = WordCompleter(
        [c[0] for c in choices],
        ignore_case=True,
        # Labels contain spaces, so the whole line is the term being completed,
        # and a fragment may land anywhere inside it ("cadre" in "Photopia —
        # Adam Cadre"). Both are off by default.
        sentence=True,
        match_middle=True,
    )
    return _session().prompt(
        HTML(f"<b>{label}</b> > "),
        completer=completer,
        complete_while_typing=True,
    ).strip()


def hint_for(choices: Sequence[Choice]) -> str:
    """The one-line instruction shown above a picker."""
    return (f"  {len(choices):,} to choose from — type any part of the name, "
            f"'back' to go back, 'quit' to exit")


def pick_one(
    label: str,
    choices: Sequence[Choice],
    *,
    allow_blank: bool = False,
    hint: bool = True,
) -> Optional[str]:
    """
    Prompt until the user picks something from `choices`, and return its value.

    Returns None on a blank line when `allow_blank`. Raises Cancelled on Ctrl-C.

    The hint is printed rather than shown in a bottom toolbar: the toolbar needs
    cursor-position support that not every terminal offers, and silently losing
    the only instruction on how to use the picker is worse than a plain line.
    """
    if hint:
        print(hint_for(choices))
    while True:
        try:
            text = _ask(label, choices)
        except (EOFError, KeyboardInterrupt):
            raise Cancelled from None

        if not text:
            if allow_blank:
                return None
            continue

        _navigate(text)

        value = _resolve(text, choices)
        if value is not None:
            return value

        matches = [c for c in choices if text.lower() in c[0].lower()]
        if not matches:
            print(f"  nothing matches {text!r}.")
        else:  # only reachable without prompt_toolkit, where nothing is filtered live
            print(f"  {len(matches)} matches — did you mean one of these?")
            for lbl, _ in matches[:FALLBACK_MATCHES]:
                print(f"    {lbl}")
            if len(matches) > FALLBACK_MATCHES:
                print(f"    … and {len(matches) - FALLBACK_MATCHES:,} more")


def pick_many(label: str, choices: Sequence[Choice]) -> List[str]:
    """
    Collect several values one per line, ending on a blank line.

    A line at a time rather than a comma-separated list because completion works
    on the whole line: with several values on it, every suggestion after the
    first would be filtered against text that includes the previous entries.
    """
    picked: List[str] = []
    print(hint_for(choices) + ", blank line when done")
    # 'back' abandons the whole list rather than undoing one entry: it means the
    # same thing here as everywhere else, and silently discarding some picks but
    # not others would be the surprising reading.
    while True:
        # Numbers the entry being typed, not the ones already in hand — an
        # unnumbered first prompt followed by "[1]" reads as though the second
        # entry were the first.
        shown = f"{label} [{len(picked) + 1}]"
        # Hint printed once above, not before every entry.
        value = pick_one(shown, choices, allow_blank=True, hint=False)
        if value is None:
            return picked
        if value in picked:
            print(f"  already added {value!r}.")
            continue
        picked.append(value)


def choose(label: str, options: Sequence[Choice]) -> str:
    """Pick from a handful of options by number or name — for short menus."""
    for i, (text, _) in enumerate(options, 1):
        print(f"    {i}. {text}")
    while True:
        try:
            raw = input(f"{label} > ").strip()
        except (EOFError, KeyboardInterrupt):
            raise Cancelled from None
        _navigate(raw)
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        value = _resolve(raw, options)
        if value is not None:
            return value
        print(f"  enter 1-{len(options)}, or the name.")


def read_line(label: str, words: Optional[Sequence[str]] = None) -> str:
    """
    A plain line of input, optionally completing `words` (used for filter keys).

    Ctrl-C raises Cancelled, not KeyboardInterrupt, so interrupting here goes up
    one level exactly as it does in a picker instead of killing the program.
    """
    try:
        if not HAS_PROMPT_TOOLKIT:
            return input(f"{label} > ").strip()
        completer = WordCompleter(list(words), ignore_case=True) if words else None
        return _session().prompt(
            HTML(f"<b>{label}</b> > "),
            completer=completer,
            complete_while_typing=bool(words),
        ).strip()
    except (EOFError, KeyboardInterrupt):
        raise Cancelled from None
