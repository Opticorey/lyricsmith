"""Per-module showcase mode convention (every lyricsmith module follows this
shape): `python -m lyricsmith.<module> --demo` prints a small, human-readable
demonstration of that module's behavior using only its own public API, with
no network calls and no dependency on modules later in the build order.

Shape copied from `lyricsmith/core/__main__.py` (the reference
implementation).
"""
import argparse

from lyricsmith.constraints import build_scaffold


def demo() -> None:
    print("=== constraints module demo ===")
    for genre, mood in [("pop", "euphoric"), ("hip_hop", "defiant"), ("folk_ballad", "grieving")]:
        song = build_scaffold(genre=genre, mood=mood)
        print(f"\n--- {genre}/{mood} scaffold ({len(song.sections)} sections) ---")
        for section in song.sections:
            scheme = song.rhyme_scheme_str(section)
            syll_ranges = sorted({c.syllable_range for c in section.constraints})
            syll_str = ", ".join(f"{lo}-{hi}" for lo, hi in syll_ranges)
            stressed = any(c.stress_pattern for c in section.constraints)
            print(
                f"  {section.role.value} #{section.index}: "
                f"scheme={scheme!r} ({len(section.constraints)} lines), "
                f"syllables={syll_str}, stress_enforced={stressed}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="constraints module showcase")
    parser.add_argument("--demo", action="store_true", help="run the demo")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
