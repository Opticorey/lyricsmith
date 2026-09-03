"""Per-line scaffold builder for lyricsmith (ARCHITECTURE.md section 3,
`constraints`).

`constraints` depends on `styles` (for `GenreProfile`/`get_profile`) and
`core` (data model + `ConstraintError`). Per the dependency graph in
ARCHITECTURE.md section 1 it also sits "above" `prosody`, but the actual
logic here never needs to call into `prosody`: building a scaffold only
needs the *shape* prosody would later validate against (syllable ranges,
rhyme-slot letters, a soft stress-pattern target string), not prosody's
analysis functions themselves. See `_alternating_stress_pattern` below for
how a soft stress target is synthesized without calling `prosody`.

Public API:
    build_scaffold(genre, mood, structure=None, seed=None) -> Song

Returns a `Song` with empty `lines` but fully populated `constraints` per
section -- ready for `generation.fill_song` to fill in actual line text.
"""
from __future__ import annotations

from lyricsmith.core import ConstraintError, LineConstraint, Section, SectionRole, Song
from lyricsmith.styles import GenreProfile, get_profile

__all__ = ["build_scaffold"]

# Free/unrhymed sections (rhyme_scheme == "") don't get a line count from
# scheme-letter counting, so we need a fallback line count per role. These
# defaults are chosen to feel like a real section of that shape (a hip-hop
# verse is long and dense; an intro/outro is short framing) -- see
# ARCHITECTURE.md's hip-hop profile note ("rhyme scheme is left free ... not
# because hip-hop doesn't rhyme, but because real flow leans on internal/
# multisyllabic rhyme"). Genuinely unlisted roles fall back to 4 lines, a
# reasonable default section length.
_DEFAULT_FREE_LINE_COUNT_BY_ROLE: dict[SectionRole, int] = {
    SectionRole.INTRO: 4,
    SectionRole.VERSE: 8,
    SectionRole.PRE_CHORUS: 4,
    SectionRole.CHORUS: 4,
    SectionRole.BRIDGE: 4,
    SectionRole.OUTRO: 4,
}
_DEFAULT_FREE_LINE_COUNT = 4


def _alternating_stress_pattern(syllable_range: tuple[int, int]) -> str:
    """Synthesize a simple canonical alternating stress pattern ('x/x/...')
    as a soft target, sized from the midpoint of the syllable range. This
    is deliberately unsophisticated -- ARCHITECTURE.md notes
    `prosody.validate_line` already treats `stress_pattern` leniently (see
    `_stress_pattern_close_enough` in prosody/analysis.py), so a plain
    alternating grid is a fine default target rather than anything derived
    from real word stress.
    """
    lo, hi = syllable_range
    n = (lo + hi) // 2
    n = max(n, 1)
    return "".join("x" if i % 2 == 0 else "/" for i in range(n))


def _line_constraints_for_role(
    role: SectionRole, section_index: int, profile: GenreProfile
) -> list[LineConstraint]:
    try:
        scheme = profile.rhyme_scheme_by_role[role]
        syllable_range = profile.syllable_range_by_role[role]
        stress_enforced = profile.stress_enforced_by_role[role]
    except KeyError:
        raise ConstraintError(
            f"genre profile {profile.name!r} has no constraint data for "
            f"section role {role.value!r}"
        ) from None

    if scheme:
        rhyme_slots: list[str | None] = list(scheme)
    else:
        # Free/unrhymed section: no scheme letters to count, so fall back
        # to a sensible per-role default line count (see module docstring).
        n_lines = _DEFAULT_FREE_LINE_COUNT_BY_ROLE.get(role, _DEFAULT_FREE_LINE_COUNT)
        rhyme_slots = [None] * n_lines

    stress_pattern = (
        _alternating_stress_pattern(syllable_range) if stress_enforced else None
    )

    constraints: list[LineConstraint] = []
    for line_index, slot in enumerate(rhyme_slots):
        constraints.append(
            LineConstraint(
                role=f"{role.value}_{section_index}_line_{line_index}",
                syllable_range=syllable_range,
                rhyme_slot=slot,
                rhyme_target_word=None,
                stress_pattern=stress_pattern,
            )
        )
    return constraints


def build_scaffold(
    genre: str,
    mood: str,
    structure: list[SectionRole] | None = None,
    seed: int | None = None,
) -> Song:
    """Build an empty (unfilled-lyrics) `Song` scaffold for a genre/mood.

    Looks up the genre's `GenreProfile` via `styles.get_profile(genre,
    mood)` (unknown genre/mood errors from `styles` propagate unchanged --
    they're already `ConstraintError`s). `structure`, if given, overrides
    the profile's default `section_order` as the sequence of section roles
    to build; each occurrence of a role gets a 0-based `index` (the second
    VERSE in the song gets index=1, etc).

    Each section's `constraints` are built from the profile's
    `rhyme_scheme_by_role` (determines line count + each line's
    `rhyme_slot` letter; `""` means free/unrhymed -- a default line count
    is used instead, see `_DEFAULT_FREE_LINE_COUNT_BY_ROLE`),
    `syllable_range_by_role` (used verbatim), and `stress_enforced_by_role`
    (a soft alternating `stress_pattern` target is generated only when
    True, else left `None`). `rhyme_target_word` always stays `None` here
    -- that's filled in later, once `generation` has an actual word for
    that rhyme slot. `title` and `theme` are left as `""` for the caller to
    fill in.

    `seed`: reserved for future randomized scaffold choices (e.g. varying
    section counts within a genre-appropriate range). Nothing in this
    version of `build_scaffold` makes a random choice, so `seed` is
    currently unused and the function is deterministic regardless of its
    value -- it exists now so a future version has a well-defined place to
    plug randomization in without changing the public signature (see
    ARCHITECTURE.md section 5, determinism rules: "build_scaffold(...,
    seed=N) is fully deterministic: same inputs + seed -> byte-identical
    scaffold").

    Raises `lyricsmith.core.ConstraintError` if `genre`/`mood` are invalid
    (propagated from `styles.get_profile`), or if an explicitly-passed
    `structure` contains a section role the profile has no constraint data
    for (shouldn't happen with any of the 5 built-in genre profiles, which
    each cover every role in their own `section_order`, but a custom
    `structure` can name a role a given genre's profile never uses --
    e.g. `PRE_CHORUS` against the `hip_hop` profile -- so this is guarded
    rather than left to raise a raw `KeyError`).
    """
    profile = get_profile(genre, mood)
    roles = structure if structure is not None else profile.section_order

    sections: list[Section] = []
    occurrence_counts: dict[SectionRole, int] = {}
    for role in roles:
        section_index = occurrence_counts.get(role, 0)
        occurrence_counts[role] = section_index + 1
        constraints = _line_constraints_for_role(role, section_index, profile)
        sections.append(Section(role=role, index=section_index, constraints=constraints))

    return Song(title="", theme="", genre=genre, mood=mood, sections=sections)
