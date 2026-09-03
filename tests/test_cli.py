"""Unit tests for lyricsmith.cli (ARCHITECTURE.md section 3, `cli`).

Built against the FROZEN generation API documented in ARCHITECTURE.md, per
the Wave 3 parallel-build rule -- these tests exercise the real `cli.main`
module. Tests that actually run `generate` end-to-end depend on
`lyricsmith.generation` (TemplateLineGenerator / fill_song) being finished
by the other Wave 3 builder; if that module isn't done yet (or has a bug),
those specific tests may fail -- that's an integration-timing issue, not a
bug in this test file or in `cli` itself. Tests that only exercise `cli`'s
own argument parsing/validation (--help, invalid --genre, invalid
--structure) don't depend on `generation` at all and should always pass.
"""
from __future__ import annotations

import os

from click.testing import CliRunner

from lyricsmith.cli.main import cli, format_scaffold_explain, format_song, parse_structure
from lyricsmith.core import SectionRole


def test_help_works():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.output


def test_generate_help_works():
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--theme" in result.output
    assert "--genre" in result.output
    assert "--mood" in result.output


def test_parse_structure_valid():
    roles = parse_structure("verse,chorus,verse,chorus,bridge,chorus")
    assert roles == [
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.VERSE,
        SectionRole.CHORUS,
        SectionRole.BRIDGE,
        SectionRole.CHORUS,
    ]


def test_parse_structure_invalid_token():
    import click
    import pytest

    with pytest.raises(click.BadParameter) as excinfo:
        parse_structure("verse,not_a_role,chorus")
    msg = str(excinfo.value)
    assert "not_a_role" in msg
    # should list valid tokens for the user
    assert "verse" in msg
    assert "chorus" in msg


def test_generate_template_engine_succeeds():
    """Full `generate` run with the offline TemplateLineGenerator. Depends
    on `lyricsmith.generation` (TemplateLineGenerator + fill_song) being
    implemented per ARCHITECTURE.md's contract -- may fail if that module
    isn't finished yet in the other Wave 3 track."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "--theme",
            "a summer that won't last",
            "--genre",
            "pop",
            "--mood",
            "hopeful",
            "--engine",
            "template",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() != ""


def test_generate_invalid_genre_gives_clean_error():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "--theme",
            "anything",
            "--genre",
            "not_a_real_genre",
            "--mood",
            "sad",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "not_a_real_genre" in result.output


def test_generate_invalid_structure_token_gives_clean_error():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "--theme",
            "anything",
            "--genre",
            "pop",
            "--mood",
            "sad",
            "--structure",
            "verse,bogus_role,chorus",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "bogus_role" in result.output


def test_explain_prints_scaffold_info():
    """--explain should print section-by-section rhyme scheme / syllable
    range / line count info. Depends on `generation` to complete the run
    (see module docstring); if it fails purely on the generation step the
    --explain output -- printed before generation runs -- should still have
    been emitted."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "--theme",
            "leaving home",
            "--genre",
            "folk_ballad",
            "--mood",
            "wistful",
            "--engine",
            "template",
            "--seed",
            "1",
            "--explain",
        ],
    )
    combined = result.output
    assert "Scaffold plan" in combined
    assert "rhyme scheme" in combined


def test_out_file_writes_formatted_lyrics(tmp_path):
    out_file = tmp_path / "song.txt"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "--theme",
            "small towns and dirt roads",
            "--genre",
            "country",
            "--mood",
            "nostalgic",
            "--engine",
            "template",
            "--seed",
            "3",
            "--out",
            str(out_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert content.strip() != ""
    # stdout should be quiet (or at least not contain the lyric body) since
    # output went to the file instead
    assert content not in result.output or result.output.strip() == ""


def test_generate_claude_engine_without_api_key_gives_clean_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "--theme",
            "anything",
            "--genre",
            "pop",
            "--mood",
            "sad",
            "--engine",
            "claude",
        ],
    )
    assert result.exit_code != 0
    # must not be a raw traceback bubbling out uncaught -- either a clean
    # SystemExit (click's own handling) or no exception at all
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_format_song_shape():
    from lyricsmith.constraints import build_scaffold

    scaffold = build_scaffold("rock", "angry", seed=1)
    scaffold.theme = "breaking free"
    scaffold.title = "Untitled Anthem"
    for section in scaffold.sections:
        section.lines = [f"line {i}" for i in range(len(section.constraints))]

    text = format_song(scaffold)
    assert text.startswith("Untitled Anthem")
    assert "[Verse 1]" in text
    assert "[Chorus 1]" in text


def test_format_scaffold_explain_shape():
    from lyricsmith.constraints import build_scaffold

    scaffold = build_scaffold("hip_hop", "confident", seed=1)
    text = format_scaffold_explain(scaffold)
    assert "Scaffold plan" in text
    assert "Verse 1" in text
