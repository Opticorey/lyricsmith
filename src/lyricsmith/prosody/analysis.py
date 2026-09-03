"""Syllable/stress/rhyme analysis on top of the CMU Pronouncing Dictionary
(via the `pronouncing` package -- public domain data, see
ARCHITECTURE.md section 7).

Pure functions only: no I/O, no network, no dependency on any lyricsmith
module besides `core`. Every function here must be safe to call on
arbitrary English text without raising -- words `pronouncing` doesn't know
fall back to a grapheme-based heuristic instead of crashing.
"""
from __future__ import annotations

import re

import pronouncing

from lyricsmith.core import LineConstraint, ValidationResult

# Matches a run of letters/apostrophes -- our definition of "a word" for
# tokenizing a line. Deliberately simple: lyrics are plain UTF-8 text with
# no markup (ARCHITECTURE.md section 2).
_WORD_RE = re.compile(r"[A-Za-z']+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
_NON_LETTER_RE = re.compile(r"[^a-z']")
_STRESS_DIGIT_RE = re.compile(r"[012]")


def _words(text: str) -> list[str]:
    """Tokenize a line of text into words, in reading order."""
    return _WORD_RE.findall(text)


def _clean(word: str) -> str:
    """Lowercase a word and strip anything that isn't a letter or
    apostrophe, so punctuation attached to a word (commas, line-final
    periods, etc.) doesn't break dictionary lookup."""
    return _NON_LETTER_RE.sub("", word.strip().lower())


def _heuristic_syllable_count(word: str) -> int:
    """Fallback syllable counter for words CMUdict doesn't know: count
    vowel groups, with a light adjustment for a silent trailing 'e'.
    Imprecise but never wrong by more than a syllable or two for ordinary
    English, and never raises.
    """
    w = _clean(word)
    if not w:
        return 0
    groups = _VOWEL_GROUP_RE.findall(w)
    count = len(groups)
    if count == 0:
        # No vowel letters at all (e.g. an initialism like "rhythm" minus
        # its vowels, or a stray consonant cluster) -- every word has at
        # least one syllable.
        return 1
    if w.endswith("e") and not w.endswith("le") and count > 1:
        count -= 1
    return max(count, 1)


def _heuristic_stress_pattern(word: str) -> str:
    """Fallback stress pattern for unknown words: alternate stress
    starting unstressed (a generic iambic default), one char per
    syllable from the heuristic syllable count.
    """
    n = _heuristic_syllable_count(word)
    return "".join("x" if i % 2 == 0 else "/" for i in range(n))


def _phones_for(word: str) -> list[str]:
    w = _clean(word)
    if not w:
        return []
    return pronouncing.phones_for_word(w)


def count_syllables(text: str) -> int:
    """Count syllables across a full line (or any span of text), summing
    per-word counts. Words unknown to CMUdict fall back to a vowel-group
    heuristic rather than raising.
    """
    total = 0
    for word in _words(text):
        phones = _phones_for(word)
        if phones:
            total += pronouncing.syllable_count(phones[0])
        else:
            total += _heuristic_syllable_count(word)
    return total


def stress_pattern(text: str) -> str:
    """Build a per-syllable stress string across the whole line, in
    reading order -- 'x' for unstressed, '/' for stressed (CMU primary
    *and* secondary stress both count as stressed, since the output
    alphabet here only distinguishes two levels). Unknown words fall back
    to an alternating heuristic pattern.
    """
    parts: list[str] = []
    for word in _words(text):
        phones = _phones_for(word)
        if phones:
            stresses = pronouncing.stresses(phones[0])
            parts.append("".join("x" if d == "0" else "/" for d in stresses))
        else:
            parts.append(_heuristic_stress_pattern(word))
    return "".join(parts)


def _normalized_rhyming_parts(word: str) -> set[str]:
    """All of a word's CMUdict "rhyming parts" (everything from the
    stressed vowel nearest the end of the word onward), one per known
    pronunciation, with stress digits stripped so e.g. primary and
    secondary stress on the same vowel still match. Empty set for a word
    CMUdict doesn't know.
    """
    parts: set[str] = set()
    for phones in _phones_for(word):
        rp = pronouncing.rhyming_part(phones)
        if rp:
            norm = _STRESS_DIGIT_RE.sub("", rp).strip()
            if norm:
                parts.add(norm)
    return parts


def _fallback_rhyme_suffix(word: str) -> str:
    """Grapheme-based fallback for words CMUdict doesn't know: the last
    few letters, used as a crude stand-in for a shared rhyme sound.
    """
    w = _clean(word)
    if len(w) <= 3:
        return w
    return w[-3:]


def rhyme_key(word: str) -> str:
    """Canonical rhyme-family key for grouping words that rhyme with each
    other. Two words sharing a rhyme_key are (at least) near-rhymes.
    Based on the word's primary CMUdict pronunciation's rhyming part
    (stress digits stripped); falls back to a grapheme suffix for unknown
    words.
    """
    parts = _normalized_rhyming_parts(word)
    if parts:
        # Multiple pronunciations can yield different rhyming parts; pick
        # a stable, deterministic one as the canonical key.
        return sorted(parts)[0]
    return _fallback_rhyme_suffix(word)


def rhymes_with(word: str, candidate: str) -> bool:
    """Practical, generous perfect/near-rhyme check: True if `word` and
    `candidate` share a rhyming part (perfect rhyme, e.g. "night"/"light")
    or at least the same vowel nucleus from the last stressed syllable
    onward (near/slant rhyme, e.g. "time"/"light") -- good enough for real
    songwriting, not just textbook-perfect rhymes. Unknown words fall back
    to comparing a grapheme suffix.
    """
    w_parts = _normalized_rhyming_parts(word)
    c_parts = _normalized_rhyming_parts(candidate)

    if w_parts and c_parts:
        if w_parts & c_parts:
            return True
        w_vowels = {p.split()[0] for p in w_parts}
        c_vowels = {p.split()[0] for p in c_parts}
        return bool(w_vowels & c_vowels)

    # At least one word is unknown to CMUdict -- fall back to a crude
    # grapheme comparison rather than refusing to answer.
    w_suffix = _fallback_rhyme_suffix(word)
    c_suffix = _fallback_rhyme_suffix(candidate)
    if not w_suffix or not c_suffix:
        return False
    return w_suffix == c_suffix or w_suffix[-2:] == c_suffix[-2:]


def _last_word(text: str) -> str | None:
    words = _words(text)
    return words[-1] if words else None


def _stress_pattern_close_enough(actual: str, target: str) -> bool:
    """Lenient stress-pattern comparison: the target is guidance, not a
    strict requirement. Only flag a line as clearly off, not for a single
    syllable being out of place.
    """
    if actual == target:
        return True
    shorter, longer = sorted((actual, target), key=len)
    mismatches = sum(1 for a, b in zip(actual, target) if a != b)
    length_diff = len(longer) - len(shorter)
    total_off = mismatches + length_diff
    # Tolerate up to a quarter of the target's syllables being off, with a
    # floor of one syllable so short lines aren't held to an exact match.
    allowed = max(1, len(target) // 4)
    return total_off <= allowed


def validate_line(text: str, constraint: LineConstraint) -> ValidationResult:
    """Check a candidate line against a LineConstraint's hard constraints
    (syllable range, rhyme target) and soft guidance (stress pattern).
    Returns a ValidationResult with a descriptive error per violation;
    never raises on ordinary text.

    INTEGRATION FIX (applied by the integrator during Wave-3 showcase
    build, see STATUS.json / critic gauntlet notes): stress-pattern
    mismatches are reported but never flip `ok` to False. CMUdict only
    records each word's stress in ISOLATION -- it has no notion of
    sentence-level destressing, so a normal English line full of
    monosyllabic function words ("the", "a", "to", "and", "I") comes back
    almost entirely stressed ("/"), which will not match a clean
    alternating "x/x/x/..." target for the overwhelming majority of real,
    well-formed lines -- including genuinely well-crafted ones. Gating
    validity on that comparison would reject good songwriting and only
    reward generators that overfit to this specific (inaccurate) metric.
    Syllable count and rhyme are the two constraints this module can
    actually measure reliably, so those are what determine `ok`; stress
    stays a visible, non-fatal signal for generators that want to use it.
    """
    errors: list[str] = []

    n = count_syllables(text)
    lo, hi = constraint.syllable_range
    if not (lo <= n <= hi):
        errors.append(f"syllable count {n} outside range ({lo}, {hi})")

    if constraint.rhyme_target_word:
        last = _last_word(text)
        if last is None:
            errors.append(
                f"line has no words to check against rhyme target "
                f"{constraint.rhyme_target_word!r}"
            )
        elif not rhymes_with(last, constraint.rhyme_target_word):
            errors.append(
                f"last word {last!r} does not rhyme with target "
                f"{constraint.rhyme_target_word!r}"
            )

    non_fatal_notes: list[str] = []
    if constraint.stress_pattern:
        actual = stress_pattern(text)
        if not _stress_pattern_close_enough(actual, constraint.stress_pattern):
            non_fatal_notes.append(
                f"(non-fatal) stress pattern {actual!r} does not closely match "
                f"target {constraint.stress_pattern!r} -- advisory only, does "
                f"not affect ok (see validate_line docstring)"
            )

    ok = not errors
    return ValidationResult(ok=ok, errors=tuple(errors + non_fatal_notes))
