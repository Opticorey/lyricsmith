"""Per-module showcase mode convention (every lyricsmith module follows this
shape): `python -m lyricsmith.<module> --demo` prints a small, human-readable
demonstration of that module's behavior using only its own public API, with
no network calls and no dependency on modules later in the build order.

Shape copied from `lyricsmith.core.__main__` (the reference implementation).
"""
import argparse

from lyricsmith.core import LineConstraint
from lyricsmith.prosody import (
    count_syllables,
    rhyme_key,
    rhymes_with,
    stress_pattern,
    validate_line,
)


def demo() -> None:
    print("=== prosody module demo ===")

    lines = [
        "the long and winding road",
        "she walks in beauty like the night",
        "we could be heroes just for one day",
    ]
    print("\n-- syllable counts --")
    for line in lines:
        print(f"  {line!r}: {count_syllables(line)} syllables")

    print("\n-- stress patterns --")
    for line in lines:
        print(f"  {line!r}: {stress_pattern(line)}")

    print("\n-- rhyme checks --")
    pairs = [
        ("night", "light"),
        ("time", "light"),
        ("love", "above"),
        ("orange", "banana"),
    ]
    for word, candidate in pairs:
        result = rhymes_with(word, candidate)
        print(f"  {word!r} / {candidate!r}: {'rhymes' if result else 'no rhyme'}")

    print("\n-- rhyme keys --")
    for word in ["night", "light", "delight", "banana"]:
        print(f"  {word!r}: {rhyme_key(word)!r}")

    print("\n-- validate_line --")
    constraint_ok = LineConstraint(
        role="verse_1_line_1",
        syllable_range=(6, 8),
        rhyme_target_word="light",
    )
    line_ok = "she walks alone tonight"
    result_ok = validate_line(line_ok, constraint_ok)
    print(f"  {line_ok!r} against {constraint_ok.syllable_range} syllables, "
          f"rhyme target {constraint_ok.rhyme_target_word!r}:")
    print(f"    ok={result_ok.ok}, errors={list(result_ok.errors)}")

    constraint_bad = LineConstraint(
        role="verse_1_line_2",
        syllable_range=(4, 5),
        rhyme_target_word="light",
    )
    line_bad = "we could be heroes just for one more day"
    result_bad = validate_line(line_bad, constraint_bad)
    print(f"  {line_bad!r} against {constraint_bad.syllable_range} syllables, "
          f"rhyme target {constraint_bad.rhyme_target_word!r}:")
    print(f"    ok={result_bad.ok}, errors={list(result_bad.errors)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prosody module showcase")
    parser.add_argument("--demo", action="store_true", help="run the demo")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
