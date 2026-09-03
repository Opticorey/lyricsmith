"""lyricsmith CLI (ARCHITECTURE.md section 3, `cli`).

Built against the FROZEN PUBLIC API documented in ARCHITECTURE.md section 3
for `constraints` and `generation` -- not against `generation`'s actual
implementation, which may not exist yet or may be mid-write in parallel
(Wave 3: `generation` and `cli` build concurrently). See the lazy import of
`lyricsmith.generation` inside `_make_generator` for how this module stays
usable (in particular, `--help` and argument validation) even if
`generation` is broken or absent.

Depends only on: `lyricsmith.core` (model + errors), `lyricsmith.styles`
(genre validation), `lyricsmith.constraints` (scaffold building) -- all
already-stable modules -- plus `lyricsmith.generation` imported lazily at
call time.
"""
from __future__ import annotations

import sys

import click

from lyricsmith.constraints import build_scaffold
from lyricsmith.core import LyricsmithError, Section, SectionRole, Song
from lyricsmith.styles import GENRE_PROFILES

__all__ = ["cli", "generate", "format_song", "parse_structure", "format_scaffold_explain"]


def parse_structure(raw: str) -> list[SectionRole]:
    """Parse a comma-separated `--structure` string into `list[SectionRole]`.

    Raises `click.BadParameter` (a clean CLI-facing error, not a traceback)
    naming the bad token and the valid tokens if any token doesn't match a
    `SectionRole` value.
    """
    valid_tokens = [role.value for role in SectionRole]
    roles: list[SectionRole] = []
    for raw_token in raw.split(","):
        token = raw_token.strip()
        try:
            roles.append(SectionRole(token))
        except ValueError:
            raise click.BadParameter(
                f"unknown structure token {token!r}; valid tokens are: "
                f"{', '.join(valid_tokens)}"
            ) from None
    return roles


def format_scaffold_explain(song: Song) -> str:
    """Render a human-readable summary of a scaffold's constraint plan --
    section by section, rhyme scheme, syllable ranges, line counts -- so a
    user can see the plan before lines are filled in (the `--explain`
    flag)."""
    lines = [f"Scaffold plan for {song.genre}/{song.mood} ({len(song.sections)} sections):"]
    role_counts: dict[SectionRole, int] = {}
    for section in song.sections:
        occurrence = role_counts.get(section.role, 0) + 1
        role_counts[section.role] = occurrence
        label = section.role.value.replace("_", " ").title()
        scheme = song.rhyme_scheme_str(section) or "(free/unrhymed)"
        syllable_ranges = sorted({c.syllable_range for c in section.constraints})
        range_str = ", ".join(f"{lo}-{hi}" for lo, hi in syllable_ranges) or "n/a"
        lines.append(
            f"  [{label} {occurrence}] {len(section.constraints)} lines, "
            f"rhyme scheme {scheme!r}, syllables {range_str}"
        )
    return "\n".join(lines)


def _section_label(section: Section, occurrence: int) -> str:
    return f"{section.role.value.replace('_', ' ').title()} {occurrence}"


def format_song(song: Song) -> str:
    """Render a filled `Song` as a plain-text lyric sheet: title, blank
    line, then each section labeled (e.g. "[Verse 1]", "[Chorus]") with its
    lines, blank line between sections."""
    parts = [song.title or "Untitled", ""]
    role_counts: dict[SectionRole, int] = {}
    for section in song.sections:
        occurrence = role_counts.get(section.role, 0) + 1
        role_counts[section.role] = occurrence
        parts.append(f"[{_section_label(section, occurrence)}]")
        parts.extend(section.lines)
        parts.append("")
    # Trim the trailing blank line from the loop, keep a single final newline.
    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts) + "\n"


def _make_generator(engine: str):
    """Instantiate a `LineGenerator` for `--engine`. Imports
    `lyricsmith.generation` lazily (inside this function, not at module
    top-level) so that `lyricsmith.cli.main` -- and therefore `--help` and
    all argument validation -- keeps working even if `generation` has an
    import error while it's mid-write in a parallel build wave. The import
    only happens once a user actually runs `generate`, at which point a
    broken `generation` module surfaces as a normal, catchable error instead
    of taking down the whole CLI.
    """
    from lyricsmith import generation

    if engine == "claude":
        return generation.ClaudeLineGenerator()
    return generation.TemplateLineGenerator()


@click.group()
def cli() -> None:
    """lyricsmith -- generate song lyrics with structure, rhyme, and meter
    enforced as hard constraints."""


@cli.command()
@click.option("--theme", required=True, type=str, help="The song's theme/subject.")
@click.option(
    "--genre",
    required=True,
    type=str,
    help=f"Genre. One of: {', '.join(sorted(GENRE_PROFILES))}.",
)
@click.option("--mood", required=True, type=str, help="The song's mood/tone.")
@click.option(
    "--structure",
    default=None,
    type=str,
    help="Comma-separated section roles, e.g. verse,chorus,verse,chorus,bridge,chorus. "
    "Omit to use the genre's default structure.",
)
@click.option(
    "--engine",
    type=click.Choice(["template", "claude"]),
    default="template",
    show_default=True,
    help=(
        "Line generation engine. 'template' is an experimental, offline, "
        "zero-dependency fallback -- see the printed disclaimer when used. "
        "'claude' is the intended full-quality path (requires ANTHROPIC_API_KEY)."
    ),
)
@click.option("--seed", type=int, default=None, help="Seed for deterministic scaffold building.")
@click.option(
    "--explain",
    is_flag=True,
    default=False,
    help="Print the scaffold's constraint plan to stderr before generating.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the formatted lyrics to this file instead of stdout.",
)
def generate(
    theme: str,
    genre: str,
    mood: str,
    structure: str | None,
    engine: str,
    seed: int | None,
    explain: bool,
    out: str | None,
) -> None:
    """Generate a full song's lyrics."""
    if genre not in GENRE_PROFILES:
        valid = ", ".join(sorted(GENRE_PROFILES))
        raise click.BadParameter(f"unknown genre {genre!r}; valid genres are: {valid}")

    # ARCHITECTURE.md section 8 (failure isolation): catch LyricsmithError
    # (and subclasses -- ConstraintError from build_scaffold, GenerationError
    # from a broken/misconfigured generator, ValidationError, or anything
    # `generation.fill_song` raises per the documented contract) at this
    # command boundary and turn it into a clean one-line stderr message plus
    # a nonzero exit, instead of a raw traceback. Truly unexpected exceptions
    # (programmer errors, bugs) are NOT caught here and propagate normally.
    try:
        parsed_structure = parse_structure(structure) if structure else None

        scaffold = build_scaffold(genre, mood, structure=parsed_structure, seed=seed)
        scaffold.theme = theme

        if explain:
            click.echo(format_scaffold_explain(scaffold), err=True)

        if engine == "template":
            # Critic gauntlet verdict (STATUS.json, generation module, round
            # 3/3 -- see ARCHITECTURE.md section 9): TemplateLineGenerator
            # scored 3-4/10 (FAIL/low-MEH) against a professional-songwriter
            # bar after three build-and-retry rounds, and is shipped BLOCKED
            # from that quality claim rather than silently passed off as
            # equivalent to the LLM-backed path. This disclaimer is the
            # ship-with-caveat step of that process, not optional flavor text.
            click.echo(
                "Note: --engine template is an experimental offline fallback. "
                "Output may contain unclear or nonsensical phrasing, unresolved "
                "references, and imagery that doesn't cohere into a clear "
                "narrative. Quality is not comparable to the standard "
                "(--engine claude) generation path. See STATUS.json for the "
                "full gauntlet history.",
                err=True,
            )

        generator = _make_generator(engine)

        from lyricsmith.generation import fill_song

        result = fill_song(scaffold, generator)
    except LyricsmithError as exc:
        raise click.ClickException(str(exc)) from exc

    if result.warnings:
        click.echo(
            f"{len(result.warnings)} line(s) needed a fallback after retries -- see below:",
            err=True,
        )
        for warning in result.warnings:
            click.echo(f"  - {warning}", err=True)

    output = format_song(result.song)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        click.echo(output, nl=False)


if __name__ == "__main__":
    sys.exit(cli())
