"""Per-module showcase mode convention (every lyricsmith module follows this
shape): `python -m lyricsmith.<module> --demo` prints a small, human-readable
demonstration of that module's behavior using only its own public API, with
no network calls and no dependency on modules later in the build order.

This file is the reference implementation other module builders should
copy the shape of.
"""
import argparse

from lyricsmith.core import LineConstraint, Section, SectionRole, Song


def demo() -> None:
    c1 = LineConstraint(role="verse_1_line_1", syllable_range=(7, 9), rhyme_slot="A")
    c2 = LineConstraint(role="verse_1_line_2", syllable_range=(7, 9), rhyme_slot="B")
    c3 = LineConstraint(role="verse_1_line_3", syllable_range=(7, 9), rhyme_slot="A")
    c4 = LineConstraint(role="verse_1_line_4", syllable_range=(7, 9), rhyme_slot="B")
    section = Section(role=SectionRole.VERSE, index=0, constraints=[c1, c2, c3, c4])
    song = Song(title="Untitled", theme="leaving a small town for good",
                genre="country", mood="bittersweet", sections=[section])

    print("=== core module demo ===")
    print(f"Song shell: {song.title!r} ({song.genre}/{song.mood}) -- theme: {song.theme!r}")
    print(f"Section: {section.role.value} #{section.index}, "
          f"rhyme scheme {song.rhyme_scheme_str(section)!r}, "
          f"{len(section.constraints)} lines, filled={section.is_filled}")
    for c in section.constraints:
        print(f"  - {c.role}: {c.syllable_range[0]}-{c.syllable_range[1]} syllables, "
              f"rhyme slot {c.rhyme_slot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="core module showcase")
    parser.add_argument("--demo", action="store_true", help="run the demo")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
