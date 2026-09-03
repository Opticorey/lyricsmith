from lyricsmith.core import LineConstraint, Section, SectionRole, Song


def test_section_is_filled_false_when_empty():
    c = LineConstraint(role="v1l1", syllable_range=(6, 9), rhyme_slot="A")
    s = Section(role=SectionRole.VERSE, index=0, constraints=[c])
    assert s.is_filled is False


def test_section_is_filled_true_when_lines_match_constraints():
    c = LineConstraint(role="v1l1", syllable_range=(6, 9), rhyme_slot="A")
    s = Section(role=SectionRole.VERSE, index=0, constraints=[c], lines=["a line of text here"])
    assert s.is_filled is True


def test_rhyme_scheme_str():
    cs = [
        LineConstraint(role="a", syllable_range=(1, 9), rhyme_slot="A"),
        LineConstraint(role="b", syllable_range=(1, 9), rhyme_slot="B"),
        LineConstraint(role="c", syllable_range=(1, 9), rhyme_slot="A"),
        LineConstraint(role="d", syllable_range=(1, 9), rhyme_slot="B"),
    ]
    s = Section(role=SectionRole.VERSE, index=0, constraints=cs)
    song = Song(title="t", theme="x", genre="pop", mood="happy", sections=[s])
    assert song.rhyme_scheme_str(s) == "ABAB"
