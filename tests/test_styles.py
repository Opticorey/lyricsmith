import pytest

from lyricsmith.core import ConstraintError, SectionRole
from lyricsmith.styles import GENRE_PROFILES, GenreProfile, get_profile


ALL_GENRES = ["pop", "hip_hop", "country", "folk_ballad", "rock"]


def test_genre_profiles_has_exactly_the_five_expected_keys():
    assert set(GENRE_PROFILES) == set(ALL_GENRES)


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_get_profile_returns_genre_profile_with_correct_structure(genre):
    profile = get_profile(genre)
    assert isinstance(profile, GenreProfile)
    assert profile.name == genre
    assert isinstance(profile.section_order, list)
    assert len(profile.section_order) > 0
    assert all(isinstance(role, SectionRole) for role in profile.section_order)
    assert isinstance(profile.rhyme_scheme_by_role, dict)
    assert isinstance(profile.syllable_range_by_role, dict)
    assert isinstance(profile.stress_enforced_by_role, dict)
    assert isinstance(profile.imagery_registers, list)
    # imagery registers are soft guidance: short, non-empty, real strings
    assert 4 <= len(profile.imagery_registers) <= 8
    assert all(isinstance(item, str) and item for item in profile.imagery_registers)


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_section_order_roles_covered_by_rhyme_and_syllable_dicts(genre):
    """Every role that actually appears in section_order must have an entry
    in both rhyme_scheme_by_role and syllable_range_by_role (and
    stress_enforced_by_role) -- otherwise a scaffold builder can't build a
    constraint for that section."""
    profile = get_profile(genre)
    roles_used = set(profile.section_order)
    assert roles_used <= set(profile.rhyme_scheme_by_role)
    assert roles_used <= set(profile.syllable_range_by_role)
    assert roles_used <= set(profile.stress_enforced_by_role)


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_syllable_ranges_are_valid_inclusive_ranges(genre):
    profile = get_profile(genre)
    for role, (lo, hi) in profile.syllable_range_by_role.items():
        assert isinstance(lo, int) and isinstance(hi, int)
        assert lo > 0
        assert lo <= hi


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_rhyme_scheme_values_are_uppercase_or_empty(genre):
    profile = get_profile(genre)
    for role, scheme in profile.rhyme_scheme_by_role.items():
        assert scheme == "" or scheme.isupper()


def test_get_profile_unknown_genre_raises_constraint_error():
    with pytest.raises(ConstraintError):
        get_profile("emo_rap")


def test_get_profile_unknown_genre_error_message_lists_valid_genres():
    with pytest.raises(ConstraintError) as exc_info:
        get_profile("not_a_genre")
    message = str(exc_info.value)
    for genre in ALL_GENRES:
        assert genre in message


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_get_profile_with_mood_returns_same_structure_as_without(genre):
    """mood doesn't change structure -- just validated/ignored for now."""
    without_mood = get_profile(genre)
    with_mood = get_profile(genre, mood="wistful")
    assert with_mood.section_order == without_mood.section_order
    assert with_mood.rhyme_scheme_by_role == without_mood.rhyme_scheme_by_role
    assert with_mood.syllable_range_by_role == without_mood.syllable_range_by_role
    assert with_mood.stress_enforced_by_role == without_mood.stress_enforced_by_role


def test_get_profile_rejects_empty_string_mood():
    with pytest.raises(ConstraintError):
        get_profile("pop", mood="")


def test_get_profile_rejects_whitespace_only_mood():
    with pytest.raises(ConstraintError):
        get_profile("pop", mood="   ")


def test_get_profile_accepts_none_mood():
    # Should not raise -- None means "no mood given".
    profile = get_profile("pop", mood=None)
    assert profile.name == "pop"


def test_hip_hop_verse_does_not_enforce_stress_but_chorus_does():
    """Per ARCHITECTURE.md: hip-hop prioritizes rhythmic density over a
    fixed stress grid for verses, but the hook/chorus should still be
    tight and repeatable."""
    profile = get_profile("hip_hop")
    assert profile.stress_enforced_by_role[SectionRole.VERSE] is False
    assert profile.stress_enforced_by_role[SectionRole.CHORUS] is True


def test_hip_hop_verse_rhyme_scheme_is_free():
    profile = get_profile("hip_hop")
    assert profile.rhyme_scheme_by_role[SectionRole.VERSE] == ""


def test_hip_hop_verses_are_denser_than_folk_ballad_verses():
    """Genre differentiation sanity check: hip-hop lines should run
    noticeably longer/denser than folk-ballad lines, not identical."""
    hip_hop = get_profile("hip_hop")
    folk = get_profile("folk_ballad")
    hh_lo, hh_hi = hip_hop.syllable_range_by_role[SectionRole.VERSE]
    fb_lo, fb_hi = folk.syllable_range_by_role[SectionRole.VERSE]
    assert hh_lo > fb_hi


def test_folk_ballad_has_no_pre_chorus_and_more_verses_than_pop():
    folk = get_profile("folk_ballad")
    pop = get_profile("pop")
    assert SectionRole.PRE_CHORUS not in folk.section_order
    folk_verse_count = folk.section_order.count(SectionRole.VERSE)
    pop_verse_count = pop.section_order.count(SectionRole.VERSE)
    assert folk_verse_count > pop_verse_count


def test_profiles_are_not_all_identical_across_genres():
    """Guard against five copy-pasted profiles with only the name changed."""
    syllable_signatures = {
        genre: tuple(sorted(get_profile(genre).syllable_range_by_role.items(), key=str))
        for genre in ALL_GENRES
    }
    assert len(set(syllable_signatures.values())) == len(ALL_GENRES)

    rhyme_signatures = {
        genre: tuple(sorted(get_profile(genre).rhyme_scheme_by_role.items(), key=str))
        for genre in ALL_GENRES
    }
    assert len(set(rhyme_signatures.values())) == len(ALL_GENRES)


def test_genre_profiles_dict_values_are_get_profile_results():
    for genre in ALL_GENRES:
        assert GENRE_PROFILES[genre] is get_profile(genre)
