from lyricsmith.core import LineConstraint
from lyricsmith.prosody import (
    count_syllables,
    rhyme_key,
    rhymes_with,
    stress_pattern,
    validate_line,
)


# -- count_syllables ---------------------------------------------------

def test_count_syllables_single_known_word():
    assert count_syllables("literally") == 4


def test_count_syllables_monosyllable():
    assert count_syllables("dog") == 1


def test_count_syllables_sums_across_a_line():
    # "the" 1 + "long" 1 + "and" 1 + "winding" 2 + "road" 1 = 6
    assert count_syllables("the long and winding road") == 6


def test_count_syllables_ignores_punctuation():
    assert count_syllables("hello, world!") == count_syllables("hello world")


def test_count_syllables_empty_string_is_zero():
    assert count_syllables("") == 0


def test_count_syllables_never_raises_on_unknown_words():
    # Nonsense tokens with no CMUdict entry must fall back, not crash.
    n = count_syllables("zzxqvthfrobnicate splunkotronic")
    assert isinstance(n, int)
    assert n > 0


def test_count_syllables_unknown_word_heuristic_is_reasonable():
    # Not a real dictionary word, but the vowel-group heuristic should
    # land in the right ballpark (2 vowel groups -> ~2 syllables).
    assert count_syllables("blorpazoid") in (2, 3, 4)


# -- stress_pattern ------------------------------------------------------

def test_stress_pattern_uses_x_and_slash_only():
    pattern = stress_pattern("she walks in beauty like the night")
    assert set(pattern) <= {"x", "/"}


def test_stress_pattern_length_matches_syllable_count():
    line = "hello world"
    assert len(stress_pattern(line)) == count_syllables(line)


def test_stress_pattern_known_word():
    # "hello" is stressed on the second syllable per CMUdict.
    assert stress_pattern("hello") == "x/"


def test_stress_pattern_unknown_word_does_not_raise():
    pattern = stress_pattern("zzxqvthfrobnicate")
    assert set(pattern) <= {"x", "/"}
    assert len(pattern) > 0


# -- rhymes_with -----------------------------------------------------------

def test_rhymes_with_perfect_rhyme_positive():
    assert rhymes_with("night", "light") is True


def test_rhymes_with_perfect_rhyme_symmetric():
    assert rhymes_with("light", "night") is True


def test_rhymes_with_common_rhyme_family():
    assert rhymes_with("love", "above") is True
    assert rhymes_with("love", "dove") is True


def test_rhymes_with_near_rhyme_generous_case():
    # Slant rhyme: same vowel nucleus, different trailing consonant --
    # should still count as a workable songwriting rhyme.
    assert rhymes_with("time", "light") is True


def test_rhymes_with_negative_case():
    assert rhymes_with("cat", "dog") is False
    assert rhymes_with("orange", "banana") is False


def test_rhymes_with_unknown_word_fallback_does_not_raise():
    # Neither word is a real CMUdict entry; the fallback heuristic must
    # still return a plain bool rather than raising.
    result = rhymes_with("splunkotronic", "bunkotronic")
    assert isinstance(result, bool)


def test_rhymes_with_unknown_word_fallback_matches_shared_suffix():
    assert rhymes_with("splunk", "chunk") is True


# -- rhyme_key -------------------------------------------------------------

def test_rhyme_key_groups_rhyming_words():
    assert rhyme_key("night") == rhyme_key("light")
    assert rhyme_key("night") == rhyme_key("delight")


def test_rhyme_key_distinguishes_non_rhymes():
    assert rhyme_key("night") != rhyme_key("banana")


def test_rhyme_key_never_raises_on_unknown_word():
    key = rhyme_key("zzxqvthfrobnicate")
    assert isinstance(key, str)
    assert key != ""


# -- validate_line -----------------------------------------------------

def test_validate_line_passes_within_syllable_range():
    constraint = LineConstraint(role="v1l1", syllable_range=(4, 6))
    result = validate_line("she walks tonight", constraint)
    assert result.ok is True
    assert result.errors == ()


def test_validate_line_fails_outside_syllable_range():
    constraint = LineConstraint(role="v1l1", syllable_range=(2, 3))
    result = validate_line("she walks alone tonight", constraint)
    assert result.ok is False
    assert any("syllable count" in e for e in result.errors)


def test_validate_line_passes_rhyme_target():
    constraint = LineConstraint(
        role="v1l1", syllable_range=(1, 20), rhyme_target_word="light"
    )
    result = validate_line("she walks tonight", constraint)
    assert result.ok is True


def test_validate_line_fails_rhyme_target_mismatch():
    constraint = LineConstraint(
        role="v1l1", syllable_range=(1, 20), rhyme_target_word="dog"
    )
    result = validate_line("she walks tonight", constraint)
    assert result.ok is False
    assert any("does not rhyme" in e for e in result.errors)


def test_validate_line_reports_multiple_errors():
    constraint = LineConstraint(
        role="v1l1", syllable_range=(1, 2), rhyme_target_word="dog"
    )
    result = validate_line("she walks tonight", constraint)
    assert result.ok is False
    assert len(result.errors) == 2


def test_validate_line_no_rhyme_target_skips_rhyme_check():
    constraint = LineConstraint(role="v1l1", syllable_range=(1, 20))
    result = validate_line("completely unconstrained line of text", constraint)
    assert result.ok is True


def test_validate_line_lenient_on_close_stress_pattern():
    # A target that differs from the line's actual pattern by exactly one
    # syllable (same length, one position flipped) should not fail the
    # line -- stress guidance is soft, not a hard requirement.
    line = "she walks in beauty like the night"
    actual = stress_pattern(line)
    assert actual == "//x/x/x/"
    close_target = "x/x/x/x/"  # first syllable flipped from the actual
    constraint = LineConstraint(
        role="v1l1", syllable_range=(1, 20), stress_pattern=close_target
    )
    result = validate_line(line, constraint)
    assert result.ok is True


def test_validate_line_flags_clearly_wrong_stress_pattern_as_non_fatal():
    # INTEGRATION FIX (see validate_line docstring): stress-pattern mismatches
    # are reported for visibility but never make `ok` False. CMUdict gives
    # per-word stress in isolation with no sentence-level destressing, so a
    # normal English line reads as almost entirely stressed -- gating on a
    # clean alternating target would reject most real, well-formed lines
    # (including well-crafted ones), not just bad ones. Syllable count and
    # rhyme are what this module can measure reliably, so those alone gate
    # `ok`; a clearly-wrong stress match still surfaces as a visible note.
    constraint = LineConstraint(
        role="v1l1", syllable_range=(1, 20), stress_pattern="/x/x/x/x/x/x/x/x"
    )
    # A single unstressed monosyllable line is wildly off a 16-syllable
    # alternating target.
    result = validate_line("no", constraint)
    assert result.ok is True
    assert any("stress pattern" in e and "non-fatal" in e for e in result.errors)


def test_validate_line_unknown_last_word_does_not_raise():
    constraint = LineConstraint(
        role="v1l1", syllable_range=(1, 20), rhyme_target_word="light"
    )
    result = validate_line("zzxqvthfrobnicate splunkotronic", constraint)
    assert isinstance(result.ok, bool)
