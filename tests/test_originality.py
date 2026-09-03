from lyricsmith.core import LineConstraint, Section, SectionRole, Song
from lyricsmith.originality import (
    CLICHE_PHRASES,
    OriginalityReport,
    check,
    cliche_flags,
    ngram_overlap,
)


# ---------------------------------------------------------------------------
# cliche_flags
# ---------------------------------------------------------------------------

def test_cliche_flags_catches_known_bad_phrase():
    text = "Tonight I'm dancing in the rain all alone"
    flags = cliche_flags(text)
    assert "dancing in the rain" in flags


def test_cliche_flags_catches_multiple_distinct_phrases():
    text = "Chasing the light while I'm screaming into the void again"
    flags = cliche_flags(text)
    assert "chasing the light" in flags
    assert "screaming into the void" in flags
    assert len(flags) == 2


def test_cliche_flags_is_case_insensitive():
    text = "SHADOWS OF MY MIND keep pulling me back down"
    flags = cliche_flags(text)
    assert "shadows of my mind" in flags


def test_cliche_flags_does_not_duplicate_repeated_phrase():
    text = "paint the sky, then paint the sky again"
    flags = cliche_flags(text)
    assert flags.count("paint the sky") == 1


def test_cliche_flags_passes_clean_original_text():
    text = "The bus pulls out at six and the porch light flickers twice"
    assert cliche_flags(text) == []


def test_cliche_flags_returns_empty_list_for_empty_string():
    assert cliche_flags("") == []


def test_cliche_list_is_reasonably_sized_and_lowercase():
    # Sanity check on the curated list itself: substantial, and stored
    # lowercase so the substring match in cliche_flags is meaningful.
    assert 25 <= len(CLICHE_PHRASES) <= 60
    assert all(p == p.lower() for p in CLICHE_PHRASES)
    assert len(set(CLICHE_PHRASES)) == len(CLICHE_PHRASES)  # no duplicates


# ---------------------------------------------------------------------------
# ngram_overlap
# ---------------------------------------------------------------------------

def test_ngram_overlap_zero_for_empty_corpus():
    assert ngram_overlap("some line with plenty of words in it", []) == 0.0


def test_ngram_overlap_zero_for_no_overlap():
    corpus = ["a completely different sentence about kitchen radios humming"]
    text = "the mountain trail wound past a frozen creek bed slowly"
    assert ngram_overlap(text, corpus) == 0.0


def test_ngram_overlap_high_for_near_duplicate_text():
    corpus = ["the bus left at seven and I counted every streetlight on the way"]
    near_duplicate = "the bus left at seven and I counted every streetlight going by"
    overlap = ngram_overlap(near_duplicate, corpus)
    assert overlap > 0.5


def test_ngram_overlap_one_for_exact_duplicate():
    corpus = ["you kept your coat on like you might still leave without a word"]
    assert ngram_overlap(corpus[0], corpus) == 1.0


def test_ngram_overlap_zero_when_text_shorter_than_n():
    corpus = ["some long sentence that definitely has enough words in it"]
    assert ngram_overlap("too short", corpus, n=5) == 0.0


def test_ngram_overlap_respects_custom_n():
    corpus = ["quiet static hums beneath the kitchen radio at noon today"]
    text = "quiet static hums beneath the porch instead of the radio"
    # With n=3, "quiet static hums" and "static hums beneath" overlap.
    overlap_n3 = ngram_overlap(text, corpus, n=3)
    overlap_n7 = ngram_overlap(text, corpus, n=7)
    assert overlap_n3 > 0.0
    assert overlap_n7 <= overlap_n3


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------

def _song_with_lines(lines: list[str]) -> Song:
    section = Section(
        role=SectionRole.VERSE,
        index=0,
        constraints=[
            LineConstraint(role=f"v1l{i+1}", syllable_range=(1, 20)) for i in range(len(lines))
        ],
        lines=lines,
    )
    return Song(title="t", theme="x", genre="pop", mood="happy", sections=[section])


def test_check_returns_originality_report():
    song = _song_with_lines(["a clean original line about a kitchen table"])
    report = check(song)
    assert isinstance(report, OriginalityReport)


def test_check_clean_true_when_no_cliches_and_no_corpus():
    song = _song_with_lines([
        "the porch light hums while the screen door counts my footsteps home",
        "a quiet kettle ticks against the window in the dark",
    ])
    report = check(song)
    assert report.clean is True
    assert report.cliche_hits == {}
    assert report.overlap_flagged_lines == []
    assert report.max_ngram_overlap == 0.0


def test_check_flags_cliche_lines_and_sets_clean_false():
    song = _song_with_lines([
        "I keep dancing in the rain and chasing the light",
        "a plain original line about a bicycle wheel",
    ])
    report = check(song)
    assert report.clean is False
    assert "I keep dancing in the rain and chasing the light" in report.cliche_hits
    hits = report.cliche_hits["I keep dancing in the rain and chasing the light"]
    assert "dancing in the rain" in hits
    assert "chasing the light" in hits
    assert "a plain original line about a bicycle wheel" not in report.cliche_hits


def test_check_flags_overlap_lines_against_supplied_corpus():
    corpus = ["the bus left at seven and I counted every streetlight on the way"]
    song = _song_with_lines([
        "the bus left at seven and I counted every streetlight going by",
        "a totally unrelated line about a rusted garden gate",
    ])
    report = check(song, corpus=corpus)
    assert report.clean is False
    assert "the bus left at seven and I counted every streetlight going by" in report.overlap_flagged_lines
    assert report.max_ngram_overlap > 0.5


def test_check_no_overlap_flags_when_corpus_omitted():
    # Per licensing policy: corpus defaults to [] so ngram checks never
    # fire unless the caller explicitly opts in.
    line = "the bus left at seven and I counted every streetlight on the way"
    song = _song_with_lines([line])
    report = check(song)
    assert report.overlap_flagged_lines == []
    assert report.max_ngram_overlap == 0.0


def test_check_ignores_empty_lines():
    song = _song_with_lines(["", "a clean line about an open window"])
    report = check(song)
    assert report.clean is True


def test_check_summary_is_nonempty_string():
    song = _song_with_lines(["a clean original line about a kitchen table"])
    report = check(song)
    assert isinstance(report.summary, str)
    assert len(report.summary) > 0
