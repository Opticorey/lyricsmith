import dataclasses

import pytest

from lyricsmith.core import ConstraintError, SectionRole, Song
from lyricsmith.constraints import build_scaffold
from lyricsmith.styles import get_profile

ALL_GENRES = ["pop", "hip_hop", "country", "folk_ballad", "rock"]


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_returns_song_with_correct_top_level_fields(genre):
    song = build_scaffold(genre=genre, mood="wistful")
    assert isinstance(song, Song)
    assert song.title == ""
    assert song.theme == ""
    assert song.genre == genre
    assert song.mood == "wistful"


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_default_structure_matches_profile_section_order(genre):
    profile = get_profile(genre)
    song = build_scaffold(genre=genre, mood="happy")
    assert [s.role for s in song.sections] == profile.section_order


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_section_indices_are_0_based_per_role_occurrence(genre):
    song = build_scaffold(genre=genre, mood="happy")
    seen: dict[SectionRole, int] = {}
    for section in song.sections:
        expected_index = seen.get(section.role, 0)
        assert section.index == expected_index
        seen[section.role] = expected_index + 1


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_sections_have_empty_lines(genre):
    song = build_scaffold(genre=genre, mood="happy")
    for section in song.sections:
        assert section.lines == []


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_rhyme_slots_match_profile_scheme_per_section(genre):
    profile = get_profile(genre)
    song = build_scaffold(genre=genre, mood="happy")
    for section in song.sections:
        scheme = profile.rhyme_scheme_by_role[section.role]
        actual_slots = [c.rhyme_slot for c in section.constraints]
        if scheme:
            assert actual_slots == list(scheme)
        else:
            # free/unrhymed: every slot is None (checked more thoroughly below)
            assert all(slot is None for slot in actual_slots)


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_syllable_ranges_match_profile(genre):
    profile = get_profile(genre)
    song = build_scaffold(genre=genre, mood="happy")
    for section in song.sections:
        expected_range = profile.syllable_range_by_role[section.role]
        for c in section.constraints:
            assert c.syllable_range == expected_range


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_stress_pattern_set_iff_enforced(genre):
    profile = get_profile(genre)
    song = build_scaffold(genre=genre, mood="happy")
    for section in song.sections:
        enforced = profile.stress_enforced_by_role[section.role]
        for c in section.constraints:
            if enforced:
                assert c.stress_pattern is not None
                assert set(c.stress_pattern) <= {"x", "/"}
            else:
                assert c.stress_pattern is None


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_rhyme_target_word_always_none(genre):
    song = build_scaffold(genre=genre, mood="happy")
    for section in song.sections:
        for c in section.constraints:
            assert c.rhyme_target_word is None


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_line_constraint_roles_unique_within_song(genre):
    song = build_scaffold(genre=genre, mood="happy")
    roles = [c.role for section in song.sections for c in section.constraints]
    assert len(roles) == len(set(roles))


def test_build_scaffold_free_verse_hip_hop_has_reasonable_line_count():
    """Hip-hop verses/intro/outro have rhyme_scheme == "" -- they should
    still get a sensible non-trivial number of lines, not zero."""
    song = build_scaffold(genre="hip_hop", mood="defiant")
    for section in song.sections:
        if section.role in (SectionRole.VERSE, SectionRole.INTRO, SectionRole.OUTRO):
            assert len(section.constraints) > 0
            assert all(c.rhyme_slot is None for c in section.constraints)


def test_build_scaffold_custom_structure_overrides_default_order():
    custom = [SectionRole.VERSE, SectionRole.CHORUS]
    song = build_scaffold(genre="pop", mood="happy", structure=custom)
    assert [s.role for s in song.sections] == custom
    assert len(song.sections) == 2


def test_build_scaffold_custom_structure_repeated_role_gets_incrementing_index():
    custom = [SectionRole.VERSE, SectionRole.VERSE, SectionRole.VERSE]
    song = build_scaffold(genre="pop", mood="happy", structure=custom)
    assert [s.index for s in song.sections] == [0, 1, 2]


def test_build_scaffold_custom_structure_role_with_no_profile_data_raises():
    """pop's profile has no PRE_CHORUS... wait, pop *does* have pre_chorus.
    Use hip_hop, whose profile never defines PRE_CHORUS at all."""
    custom = [SectionRole.PRE_CHORUS]
    with pytest.raises(ConstraintError):
        build_scaffold(genre="hip_hop", mood="defiant", structure=custom)


def test_build_scaffold_unknown_genre_raises_constraint_error():
    with pytest.raises(ConstraintError):
        build_scaffold(genre="emo_rap", mood="happy")


def test_build_scaffold_unknown_genre_propagates_from_styles_not_swallowed():
    with pytest.raises(ConstraintError) as exc_info:
        build_scaffold(genre="not_a_genre", mood="happy")
    assert "not_a_genre" in str(exc_info.value)


def test_build_scaffold_invalid_mood_raises_constraint_error():
    with pytest.raises(ConstraintError):
        build_scaffold(genre="pop", mood="")


@pytest.mark.parametrize("genre", ALL_GENRES)
@pytest.mark.parametrize("seed", [None, 0, 42, 12345])
def test_build_scaffold_deterministic_given_same_inputs_and_seed(genre, seed):
    song_a = build_scaffold(genre=genre, mood="happy", seed=seed)
    song_b = build_scaffold(genre=genre, mood="happy", seed=seed)
    assert dataclasses.asdict(song_a) == dataclasses.asdict(song_b)


def test_build_scaffold_deterministic_across_different_seeds_too():
    """Nothing in this version of build_scaffold actually randomizes on
    seed, so different seeds currently produce identical scaffolds too --
    this pins that behavior down explicitly."""
    song_a = build_scaffold(genre="rock", mood="angry", seed=1)
    song_b = build_scaffold(genre="rock", mood="angry", seed=999)
    assert dataclasses.asdict(song_a) == dataclasses.asdict(song_b)


def test_build_scaffold_seed_is_keyword_only_friendly_default_none():
    # seed should be optional and default to None
    song = build_scaffold(genre="pop", mood="happy")
    assert isinstance(song, Song)


@pytest.mark.parametrize("genre", ALL_GENRES)
def test_build_scaffold_section_count_matches_default_structure_length(genre):
    profile = get_profile(genre)
    song = build_scaffold(genre=genre, mood="happy")
    assert len(song.sections) == len(profile.section_order)
