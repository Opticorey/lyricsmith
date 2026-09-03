"""Per-module showcase mode convention (every lyricsmith module follows this
shape): `python -m lyricsmith.<module> --demo` prints a small, human-readable
demonstration of that module's behavior using only its own public API, with
no network calls and no dependency on modules later in the build order.

Shape copied from lyricsmith.core.__main__.
"""
import argparse

from lyricsmith.core import LineConstraint, Section, SectionRole, Song
from lyricsmith.originality import check, cliche_flags, ngram_overlap


def demo() -> None:
    print("=== originality module demo ===")

    # -- cliche_flags: a deliberately cliche-written example line (written by
    # us for this demo, not lifted from any real song) vs. a clean original
    # line.
    cliche_line = "I'm dancing in the rain, chasing the light, screaming into the void"
    clean_line = "The porch light hums while the screen door counts my footsteps home"

    print("\n-- cliche_flags --")
    print(f"  line: {cliche_line!r}")
    print(f"  flags: {cliche_flags(cliche_line)}")
    print(f"  line: {clean_line!r}")
    print(f"  flags: {cliche_flags(clean_line)}")

    # -- ngram_overlap: a small invented (non-copyrighted) demo corpus, made
    # up entirely for this demo.
    demo_corpus = [
        "the bus left at seven and I counted every streetlight on the way",
        "you kept your coat on like you might still leave without a word",
    ]
    near_duplicate = "the bus left at seven and I counted every streetlight going by"
    unrelated = "quiet static hums beneath the kitchen radio at noon"

    print("\n-- ngram_overlap (n=5) --")
    print(f"  corpus: {demo_corpus}")
    print(f"  near-duplicate line: {near_duplicate!r}")
    print(f"  overlap: {ngram_overlap(near_duplicate, demo_corpus):.2f}")
    print(f"  unrelated line: {unrelated!r}")
    print(f"  overlap: {ngram_overlap(unrelated, demo_corpus):.2f}")

    # -- check(): aggregate report over a small demo Song.
    section = Section(
        role=SectionRole.VERSE,
        index=0,
        constraints=[
            LineConstraint(role="verse_1_line_1", syllable_range=(6, 12)),
            LineConstraint(role="verse_1_line_2", syllable_range=(6, 12)),
        ],
        lines=[cliche_line, clean_line],
    )
    song = Song(
        title="Demo Song",
        theme="a quiet night at home",
        genre="folk_ballad",
        mood="wistful",
        sections=[section],
    )

    report = check(song, corpus=demo_corpus)
    print("\n-- check(song, corpus=demo_corpus) --")
    print(f"  clean: {report.clean}")
    print(f"  cliche_hits: {report.cliche_hits}")
    print(f"  max_ngram_overlap: {report.max_ngram_overlap:.2f}")
    print(f"  overlap_flagged_lines: {report.overlap_flagged_lines}")
    print(f"  summary: {report.summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="originality module showcase")
    parser.add_argument("--demo", action="store_true", help="run the demo")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
