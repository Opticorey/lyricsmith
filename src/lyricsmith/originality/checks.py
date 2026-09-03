"""Originality/plagiarism-safety checks for lyricsmith.

Two independent signals:

1. ``cliche_flags`` -- a hand-curated list of generic, overused lyric
   phrases/images that read as "AI slop" rather than specific songwriter
   craft (ARCHITECTURE.md section 0). These are our own generic-phrase
   judgments about *categories* of stock imagery common across pop lyric
   writing in general -- none of them are drawn from, or quote, any
   specific copyrighted song. See ARCHITECTURE.md section 7.

2. ``ngram_overlap`` -- a caller-supplied-corpus word n-gram overlap
   check. No corpus ships with this package; callers opt in with their
   own text if they want near-duplication detection against something.

Both are pure, offline, deterministic, and depend on nothing outside the
standard library plus ``lyricsmith.core``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from lyricsmith.core import Song

# ---------------------------------------------------------------------------
# Cliche list
# ---------------------------------------------------------------------------
# Hand-curated stock phrases/images that show up over and over in generic
# (often AI-generated) pop/rock lyric writing. These are broad categories of
# overused imagery we are naming ourselves -- not quotations from any real,
# identifiable, copyrighted song. Matching is case-insensitive substring
# matching against a normalized (whitespace-collapsed) version of the input.
CLICHE_PHRASES: tuple[str, ...] = (
    "shadows of my mind",
    "chasing the light",
    "broken pieces of my heart",
    "pieces of my heart",
    "dancing in the rain",
    "screaming into the void",
    "paint the sky",
    "whispers in the wind",
    "diamond in the rough",
    "burning like a fire",
    "chains that bind me",
    "tears like rain",
    "fire in my soul",
    "ashes of yesterday",
    "ghost of who i was",
    "puzzle pieces falling",
    "shattered like glass",
    "heart of stone",
    "storm inside my head",
    "demons in the dark",
    "light at the end of the tunnel",
    "wildfire in my veins",
    "castle made of sand",
    "walls i built around",
    "prisoner of my own mind",
    "drowning in the silence",
    "dancing with the devil",
    "written in the stars",
    "beautiful disaster",
    "broken wings",
    "empty room, empty heart",
    "puppet on a string",
    "mirror of my soul",
    "howling at the moon",
    "bleeding out these words",
    "echoes of a memory",
    "rise from the ashes",
    "colors of a fading dream",
    "hourglass running out",
    "battle scars i hide",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def cliche_flags(text: str) -> list[str]:
    """Return the list of stock phrases from ``CLICHE_PHRASES`` found as
    case-insensitive substrings of ``text``. Order follows ``CLICHE_PHRASES``;
    duplicates in the source list are never produced (each phrase appears at
    most once even if it occurs multiple times in ``text``)."""
    normalized = _normalize(text)
    return [phrase for phrase in CLICHE_PHRASES if phrase in normalized]


# ---------------------------------------------------------------------------
# N-gram overlap
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(text: str, corpus: list[str], n: int = 5) -> float:
    """Fraction (0..1) of ``text``'s word n-grams that also appear
    somewhere in ``corpus``'s combined n-gram set.

    Returns 0.0 if ``corpus`` is empty, or if ``text`` has fewer than
    ``n`` words (no n-grams to compare)."""
    if not corpus:
        return 0.0
    text_ngrams = _ngrams(_tokenize(text), n)
    if not text_ngrams:
        return 0.0

    corpus_ngrams: set[tuple[str, ...]] = set()
    for entry in corpus:
        corpus_ngrams |= _ngrams(_tokenize(entry), n)
    if not corpus_ngrams:
        return 0.0

    hits = sum(1 for gram in text_ngrams if gram in corpus_ngrams)
    return hits / len(text_ngrams)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

OVERLAP_FLAG_THRESHOLD = 0.5


@dataclass
class OriginalityReport:
    """Aggregate originality/plagiarism-safety result for a `Song`."""

    cliche_hits: dict[str, list[str]] = field(default_factory=dict)
    max_ngram_overlap: float = 0.0
    overlap_flagged_lines: list[str] = field(default_factory=list)
    clean: bool = True
    summary: str = ""


def check(song: Song, corpus: list[str] | None = None) -> OriginalityReport:
    """Run cliche and n-gram overlap checks over every line of every
    section in ``song``.

    ``corpus`` defaults to an empty list when omitted: per the project's
    licensing policy (ARCHITECTURE.md section 7), no lyrics corpus ships
    with this package, so ``ngram_overlap`` is always 0.0 unless the
    caller explicitly supplies their own comparison text.
    """
    if corpus is None:
        corpus = []

    cliche_hits: dict[str, list[str]] = {}
    overlap_flagged_lines: list[str] = []
    max_overlap = 0.0

    for line in song.all_lines():
        if not line:
            continue

        hits = cliche_flags(line)
        if hits:
            cliche_hits[line] = hits

        overlap = ngram_overlap(line, corpus)
        if overlap > max_overlap:
            max_overlap = overlap
        if overlap > OVERLAP_FLAG_THRESHOLD:
            overlap_flagged_lines.append(line)

    clean = not cliche_hits and not overlap_flagged_lines

    if clean:
        summary = "Clean: no cliche phrases and no n-gram overlap above threshold."
    else:
        parts = []
        if cliche_hits:
            total_hits = sum(len(v) for v in cliche_hits.values())
            parts.append(f"{total_hits} cliche hit(s) across {len(cliche_hits)} line(s)")
        if overlap_flagged_lines:
            parts.append(
                f"{len(overlap_flagged_lines)} line(s) with n-gram overlap "
                f"above {OVERLAP_FLAG_THRESHOLD:.0%} (max {max_overlap:.0%})"
            )
        summary = "Flagged: " + "; ".join(parts)

    return OriginalityReport(
        cliche_hits=cliche_hits,
        max_ngram_overlap=max_overlap,
        overlap_flagged_lines=overlap_flagged_lines,
        clean=clean,
        summary=summary,
    )
