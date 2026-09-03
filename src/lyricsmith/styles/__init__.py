"""Genre/mood profiles for lyricsmith (ARCHITECTURE.md section 3, `styles`).

`styles` depends only on `core` (for `SectionRole` and `ConstraintError`).
It defines the real `GenreProfile` dataclass that satisfies core's
structural `GenreProfileLike` protocol (name, section_order,
rhyme_scheme_by_role, syllable_range_by_role) and adds the extra fields
`constraints` (a later wave) needs to build a per-line scaffold:
`stress_enforced_by_role` and `imagery_registers`.

Every profile below is hand-tuned per genre -- section shape, rhyme
tightness, syllable density, and whether a fixed stress grid is enforced
all genuinely differ genre-to-genre, because that's what actually makes a
generated pop chorus feel different from a generated folk verse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lyricsmith.core import ConstraintError, SectionRole

__all__ = ["GenreProfile", "GENRE_PROFILES", "get_profile"]


@dataclass(frozen=True)
class GenreProfile:
    """A genre's structural + soft-guidance profile.

    `name`, `section_order`, `rhyme_scheme_by_role`, and
    `syllable_range_by_role` satisfy `core.model.GenreProfileLike`.
    """

    name: str
    section_order: list[SectionRole]
    rhyme_scheme_by_role: dict[SectionRole, str]
    syllable_range_by_role: dict[SectionRole, tuple[int, int]]
    stress_enforced_by_role: dict[SectionRole, bool]
    # Soft flavor guidance only -- never a required-content check. A valid
    # song in this genre does not have to use any of these registers.
    imagery_registers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# POP
#
# Tight, hooky, built for repetition. Chorus lines run short and punchy
# (the part a listener sings back), pre-chorus builds tension with a
# tighter couplet rhyme, verses are the loosest section. Stress is enforced
# throughout because pop melodies are rhythmically strict.
# ---------------------------------------------------------------------------
_POP = GenreProfile(
    name="pop",
    section_order=[
        SectionRole.VERSE,
        SectionRole.PRE_CHORUS,
        SectionRole.CHORUS,
        SectionRole.VERSE,
        SectionRole.PRE_CHORUS,
        SectionRole.CHORUS,
        SectionRole.BRIDGE,
        SectionRole.CHORUS,
    ],
    rhyme_scheme_by_role={
        SectionRole.VERSE: "ABAB",
        SectionRole.PRE_CHORUS: "AABB",
        SectionRole.CHORUS: "AABB",
        SectionRole.BRIDGE: "ABCB",
    },
    syllable_range_by_role={
        SectionRole.VERSE: (7, 10),
        SectionRole.PRE_CHORUS: (6, 8),
        SectionRole.CHORUS: (5, 7),
        SectionRole.BRIDGE: (6, 9),
    },
    stress_enforced_by_role={
        SectionRole.VERSE: True,
        SectionRole.PRE_CHORUS: True,
        SectionRole.CHORUS: True,
        SectionRole.BRIDGE: True,
    },
    imagery_registers=[
        "heartbreak and reconciliation",
        "city lights and late-night drives",
        "dancing to forget",
        "texts left on read",
        "a summer that won't last",
        "mirrors and self-doubt",
        "young love against the odds",
    ],
)

# ---------------------------------------------------------------------------
# HIP-HOP
#
# Verses are long, dense, and rhythm-first: rhyme scheme is left free (""),
# not because hip-hop doesn't rhyme, but because real flow leans on
# internal/multisyllabic rhyme and pocket-placement that a fixed end-rhyme
# letter scheme can't capture, and stress is NOT enforced for verses --
# per ARCHITECTURE.md, hip-hop prioritizes rhythmic density over a fixed
# stress grid. The chorus/hook flips that: it needs to be tight, repeatable,
# and stress-enforced so it lands as a singable hook.
# ---------------------------------------------------------------------------
_HIP_HOP = GenreProfile(
    name="hip_hop",
    section_order=[
        SectionRole.INTRO,
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.OUTRO,
    ],
    rhyme_scheme_by_role={
        SectionRole.INTRO: "",
        SectionRole.VERSE: "",
        SectionRole.CHORUS: "AABB",
        SectionRole.OUTRO: "",
    },
    syllable_range_by_role={
        SectionRole.INTRO: (8, 14),
        SectionRole.VERSE: (13, 20),
        SectionRole.CHORUS: (6, 10),
        SectionRole.OUTRO: (6, 10),
    },
    stress_enforced_by_role={
        SectionRole.INTRO: False,
        SectionRole.VERSE: False,
        SectionRole.CHORUS: True,
        SectionRole.OUTRO: False,
    },
    imagery_registers=[
        "hustle and the come-up",
        "loyalty and betrayal",
        "the block, the corner store",
        "self-made success",
        "money as proof of survival",
        "late nights in the studio",
        "family sacrifice",
        "receipts and comeuppance",
    ],
)

# ---------------------------------------------------------------------------
# COUNTRY
#
# Storytelling verses: longer, conversational, narrative rhyme (ABAB) that
# reads like a sentence. Chorus tightens into AABB for the memorable hook.
# Stress is enforced but loosely -- country phrasing tolerates more
# conversational stretch than pop while still riding a clear meter.
# ---------------------------------------------------------------------------
_COUNTRY = GenreProfile(
    name="country",
    section_order=[
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.BRIDGE,
        SectionRole.CHORUS,
    ],
    rhyme_scheme_by_role={
        SectionRole.VERSE: "ABAB",
        SectionRole.CHORUS: "AABB",
        SectionRole.BRIDGE: "ABAB",
    },
    syllable_range_by_role={
        SectionRole.VERSE: (8, 11),
        SectionRole.CHORUS: (6, 9),
        SectionRole.BRIDGE: (7, 10),
    },
    stress_enforced_by_role={
        SectionRole.VERSE: True,
        SectionRole.CHORUS: True,
        SectionRole.BRIDGE: True,
    },
    imagery_registers=[
        "small towns and dirt roads",
        "pickup trucks",
        "weather as mood",
        "family land passed down",
        "leaving vs. staying",
        "front porches and Sunday mornings",
        "faith",
        "first loves and old flames",
    ],
)

# ---------------------------------------------------------------------------
# FOLK BALLAD
#
# No pre-chorus, more verses -- the story carries the song. Verse rhyme is
# the classic ballad quatrain (ABCB: only lines 2 and 4 rhyme), sparser and
# more prose-like than pop/country. Lines run short and plain-spoken.
# Intro/outro are spoken-word-adjacent framing, unrhymed and unenforced.
# ---------------------------------------------------------------------------
_FOLK_BALLAD = GenreProfile(
    name="folk_ballad",
    section_order=[
        SectionRole.INTRO,
        SectionRole.VERSE,
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.VERSE,
        SectionRole.OUTRO,
    ],
    rhyme_scheme_by_role={
        SectionRole.INTRO: "",
        SectionRole.VERSE: "ABCB",
        SectionRole.CHORUS: "AABB",
        SectionRole.OUTRO: "",
    },
    syllable_range_by_role={
        SectionRole.INTRO: (5, 8),
        SectionRole.VERSE: (6, 8),
        SectionRole.CHORUS: (5, 8),
        SectionRole.OUTRO: (5, 8),
    },
    stress_enforced_by_role={
        SectionRole.INTRO: False,
        SectionRole.VERSE: True,
        SectionRole.CHORUS: True,
        SectionRole.OUTRO: False,
    },
    imagery_registers=[
        "rivers and mountains",
        "ghosts and old letters",
        "seasons turning",
        "hand-me-down stories",
        "graves and homecomings",
        "wandering and exile",
        "work-worn hands",
    ],
)

# ---------------------------------------------------------------------------
# ROCK
#
# Short, punchy, riff-driven lines built to ride a hard backbeat -- the
# shortest lines of any genre here, verse and chorus alike, with tight
# couplet rhymes (AABB) that shout back easily. Stress is enforced hard
# throughout the sung sections since the rhythm section won't bend for a
# ragged lyric.
# ---------------------------------------------------------------------------
_ROCK = GenreProfile(
    name="rock",
    section_order=[
        SectionRole.INTRO,
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.BRIDGE,
        SectionRole.CHORUS,
        SectionRole.OUTRO,
    ],
    rhyme_scheme_by_role={
        SectionRole.INTRO: "",
        SectionRole.VERSE: "AABB",
        SectionRole.CHORUS: "AABB",
        SectionRole.BRIDGE: "ABAB",
        SectionRole.OUTRO: "",
    },
    syllable_range_by_role={
        SectionRole.INTRO: (4, 7),
        SectionRole.VERSE: (5, 8),
        SectionRole.CHORUS: (4, 7),
        SectionRole.BRIDGE: (6, 9),
        SectionRole.OUTRO: (4, 7),
    },
    stress_enforced_by_role={
        SectionRole.INTRO: False,
        SectionRole.VERSE: True,
        SectionRole.CHORUS: True,
        SectionRole.BRIDGE: True,
        SectionRole.OUTRO: False,
    },
    imagery_registers=[
        "rebellion and defiance",
        "open roads, fast cars",
        "fire and adrenaline",
        "breaking free",
        "electric nights",
        "scars worn proud",
        "us against the world",
    ],
)

GENRE_PROFILES: dict[str, GenreProfile] = {
    "pop": _POP,
    "hip_hop": _HIP_HOP,
    "country": _COUNTRY,
    "folk_ballad": _FOLK_BALLAD,
    "rock": _ROCK,
}


def get_profile(genre: str, mood: str | None = None) -> GenreProfile:
    """Look up a genre's `GenreProfile`.

    `mood` doesn't change the returned profile's structure -- moods affect
    content generation (word choice, tone), not section/rhyme/syllable
    structure -- so this is intentionally a light touch for now: if given,
    it must be a non-empty string, and is otherwise ignored. A future
    version may branch structure on mood (e.g. a "somber" bridge dropping
    the enforced stress grid); nothing here forecloses that.
    """
    if mood is not None and not mood.strip():
        raise ConstraintError("mood, if provided, must be a non-empty string")

    try:
        return GENRE_PROFILES[genre]
    except KeyError:
        valid = ", ".join(sorted(GENRE_PROFILES))
        raise ConstraintError(
            f"unknown genre {genre!r}; valid genres are: {valid}"
        ) from None
