"""Thin web wrapper around the real lyricsmith engine, so the Lyricsmith
Console (web/static/index.html) can generate ACTUAL songs live instead of
replaying captured samples.

Deliberately minimal: one POST endpoint that calls the exact same
`build_scaffold` + `fill_song` pipeline the CLI uses, plus static file
serving for the console page itself. No database, no auth, no queueing --
this is meant to be the fastest honest path from "working local package" to
"something with a URL", not a production service.

ANTHROPIC_API_KEY is read server-side from the environment (Render's env
var config, in the deployed case) and NEVER sent to or accepted from the
browser -- the whole point of moving generation server-side instead of
running it in client-side JS.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lyricsmith.constraints import build_scaffold
from lyricsmith.core import LyricsmithError, Section, SectionRole, Song
from lyricsmith.styles import GENRE_PROFILES

app = FastAPI(title="lyricsmith web")

# Permissive CORS: this is a small demo service, not a multi-tenant API --
# the console page is the only intended caller, but there's no session/auth
# boundary here worth protecting with a stricter origin list.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class GenerateRequest(BaseModel):
    theme: str = Field(..., min_length=1, max_length=300)
    genre: str
    mood: str = Field(..., min_length=1, max_length=100)
    structure: Optional[str] = None  # comma-separated SectionRole tokens, same as CLI --structure
    engine: str = "template"  # "template" | "claude"
    seed: Optional[int] = None


def _parse_structure(raw: str) -> list[SectionRole]:
    valid_tokens = [role.value for role in SectionRole]
    roles: list[SectionRole] = []
    for raw_token in raw.split(","):
        token = raw_token.strip()
        try:
            roles.append(SectionRole(token))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"unknown structure token {token!r}; valid tokens are: {', '.join(valid_tokens)}",
            ) from None
    return roles


def _section_label(section: Section, occurrence: int) -> str:
    return f"{section.role.value.replace('_', ' ').title()} {occurrence}"


def _section_scheme(song: Song, section: Section) -> str:
    raw = song.rhyme_scheme_str(section)
    return "free" if set(raw) <= {"-"} else raw


def _section_syll(section: Section) -> str:
    ranges = sorted({c.syllable_range for c in section.constraints})
    return ", ".join(f"{lo}–{hi}" for lo, hi in ranges) or "n/a"


def _serialize(song: Song, warnings: list[str]) -> dict:
    role_counts: dict[SectionRole, int] = {}
    sections_out = []
    explain_out = []
    for section in song.sections:
        occurrence = role_counts.get(section.role, 0) + 1
        role_counts[section.role] = occurrence
        label = _section_label(section, occurrence)
        scheme = _section_scheme(song, section)
        sections_out.append({"role": label, "scheme": scheme, "lines": list(section.lines)})
        explain_out.append(
            {
                "role": label,
                "scheme": scheme,
                "syll": _section_syll(section),
            }
        )
    return {
        "title": song.title or "Untitled",
        "genre": song.genre,
        "mood": song.mood,
        "theme": song.theme,
        "warnings": len(warnings),
        "total": len(song.all_lines()),
        "sections": sections_out,
        "explain": explain_out,
    }


def _make_generator(engine: str):
    # Imported lazily, same reasoning as the CLI's `_make_generator`: keep
    # the rest of this module usable even if `generation`/`anthropic` has a
    # problem, and only pay the cost of constructing the (possibly
    # key-requiring) generator once a request actually asks for it.
    from lyricsmith import generation

    if engine == "claude":
        return generation.ClaudeLineGenerator()
    return generation.TemplateLineGenerator()


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    if req.genre not in GENRE_PROFILES:
        valid = ", ".join(sorted(GENRE_PROFILES))
        raise HTTPException(status_code=400, detail=f"unknown genre {req.genre!r}; valid genres are: {valid}")
    if req.engine not in ("template", "claude"):
        raise HTTPException(status_code=400, detail="engine must be 'template' or 'claude'")

    try:
        parsed_structure = _parse_structure(req.structure) if req.structure else None
        scaffold = build_scaffold(req.genre, req.mood, structure=parsed_structure, seed=req.seed)
        scaffold.theme = req.theme

        generator = _make_generator(req.engine)

        from lyricsmith.generation import fill_song

        result = fill_song(scaffold, generator)
    except LyricsmithError as exc:
        # Same failure-isolation contract as the CLI: a clean 4xx with the
        # real error text, never a raw traceback. This is also what a
        # missing/broken ANTHROPIC_API_KEY surfaces as (GenerationError is a
        # LyricsmithError subclass, raised from ClaudeLineGenerator.__init__).
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _serialize(result.song, result.warnings)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
