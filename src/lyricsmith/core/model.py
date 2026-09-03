"""Shared data model for lyricsmith. No internal dependencies on other
lyricsmith modules — everything else depends on this, never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


class SectionRole(str, Enum):
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"


@dataclass(frozen=True)
class Syllable:
    """A phonetic syllable. stress follows CMU dict convention:
    0 = unstressed, 1 = primary stress, 2 = secondary stress."""
    text: str
    stress: int  # 0, 1, or 2


@dataclass(frozen=True)
class Word:
    text: str
    syllables: tuple[Syllable, ...]
    phonemes: str = ""

    @property
    def syllable_count(self) -> int:
        return len(self.syllables)


@dataclass(frozen=True)
class LineConstraint:
    """A hard constraint a generated line must satisfy."""
    role: str  # e.g. "verse_1_line_3" -- unique within a song, used for logging/debugging
    syllable_range: tuple[int, int]  # inclusive (min, max)
    rhyme_slot: Optional[str] = None  # e.g. "A"; None = unconstrained rhyme
    rhyme_target_word: Optional[str] = None  # filled in once the first line of the slot exists
    stress_pattern: Optional[str] = None  # e.g. "x/x/x/x/"; None = not enforced


@dataclass
class Section:
    role: SectionRole
    index: int  # 0-based occurrence of this role within the song (e.g. verse #0, verse #1)
    constraints: list[LineConstraint] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)  # filled in by generation; empty until then

    @property
    def is_filled(self) -> bool:
        return len(self.lines) == len(self.constraints) and all(self.lines)


@dataclass
class Song:
    title: str
    theme: str
    genre: str
    mood: str
    sections: list[Section] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return bool(self.sections) and all(s.is_filled for s in self.sections)

    def all_lines(self) -> list[str]:
        return [line for s in self.sections for line in s.lines]

    def rhyme_scheme_str(self, section: Section) -> str:
        return "".join(c.rhyme_slot or "-" for c in section.constraints)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()


class GenreProfileLike(Protocol):
    """Structural shape `core` expects from styles.GenreProfile, declared
    here to avoid styles -> core -> styles circularity. styles.py defines
    the real dataclass; this is just the contract."""
    name: str
    section_order: list[SectionRole]
    rhyme_scheme_by_role: dict
    syllable_range_by_role: dict
