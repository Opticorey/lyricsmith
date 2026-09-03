"""Per-module showcase mode convention (every lyricsmith module follows this
shape): `python -m lyricsmith.<module> --demo` prints a small, human-readable
demonstration of that module's behavior using only its own public API, with
no network calls and no dependency on modules later in the build order.

Shape copied from `lyricsmith.core.__main__` (the reference implementation).
This demo runs `fill_song` end to end with `TemplateLineGenerator` --
`ClaudeLineGenerator` is deliberately not exercised here: this sandbox has
no `ANTHROPIC_API_KEY` and no network path to the Anthropic API (see
ARCHITECTURE.md section 9), so it is only described below.
"""
import argparse

from lyricsmith.constraints import build_scaffold
from lyricsmith.core import SectionRole
from lyricsmith.generation import ClaudeLineGenerator, TemplateLineGenerator, fill_song


def demo() -> None:
    print("=== generation module demo ===")

    scaffold = build_scaffold(
        genre="pop",
        mood="bittersweet",
        structure=[SectionRole.VERSE, SectionRole.CHORUS, SectionRole.VERSE],
    )
    scaffold.theme = "driving away from a hometown that stopped feeling like home"

    generator = TemplateLineGenerator(seed=7)
    result = fill_song(scaffold, generator, max_retries=2)
    song = result.song

    print(f"\nTitle: {song.title!r}")
    print(f"Theme: {song.theme!r}  Genre: {song.genre}  Mood: {song.mood}")
    print(f"Fully filled: {song.is_filled}")

    for section in song.sections:
        scheme = song.rhyme_scheme_str(section)
        print(f"\n-- {section.role.value} #{section.index} (rhyme scheme {scheme!r}) --")
        for line, constraint in zip(section.lines, section.constraints):
            n = len(line.split())
            print(f"  [{constraint.rhyme_slot or '-'}] {line}")

    print(f"\nWarnings ({len(result.warnings)}):")
    if result.warnings:
        for w in result.warnings:
            print(f"  - {w}")
    else:
        print("  (none -- every line satisfied its constraints on first or retried attempt)")

    print(
        "\nNote: ClaudeLineGenerator (LLM-backed) is not demoed live here -- "
        "it requires the ANTHROPIC_API_KEY environment variable and network "
        "access to the Anthropic API, neither of which is available in this "
        "sandbox. Its class is importable and fully implemented "
        f"({ClaudeLineGenerator!r}); constructing it without a key raises "
        "lyricsmith.core.GenerationError, as verified in tests/test_generation.py."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generation module showcase")
    parser.add_argument("--demo", action="store_true", help="run the demo")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
