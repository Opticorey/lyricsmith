# lyricsmith — ARCHITECTURE

## 0. Goal and bar

Generate song lyrics that read as **professional songwriter craft**:
consistent imagery and POV, natural prosody that actually sings, emotional
specificity, hooks that land. Never AI-slop: forced rhymes, cliché imagery,
inconsistent tense/POV, meter-breaking filler.

Engine strategy: **hybrid**. An algorithmic scaffold (this package's own
code) enforces structure, rhyme scheme, syllable count, and stress pattern
as hard constraints. An LLM (or a template fallback) fills the actual line
content within those constraints, validated and retried against the same
hard constraints.

## 1. Module boundaries and dependency graph

```
core            (data model + shared enums/units — no internal deps)
  |
  +-- prosody       (syllable/stress/rhyme analysis — depends on core)
  +-- styles        (genre/mood profiles — depends on core)
  +-- originality   (cliché + n-gram overlap checks — depends on core, prosody)
        |
        +-- constraints   (per-line scaffold builder — depends on prosody, styles)
              |
              +-- generation   (LineGenerator impls + orchestration — depends on constraints, prosody, originality)
              +-- cli          (argument parsing + I/O — depends on the PUBLIC API of generation, styles, constraints; built in parallel against this doc, not against generation's internals)
                    |
                    +-- showcase   (harness fixtures + example gallery — depends on everything; built last by the integrator)
```

Build waves (dependency order):
- **Wave 0** (integrator, before fan-out): `core`, verification harness skeleton
- **Wave 1** (parallel): `prosody`, `styles`, `originality`
- **Wave 2**: `constraints`
- **Wave 3** (parallel): `generation`, `cli` (cli codes against the frozen public API below, not against generation's actual code, so it can proceed in parallel)
- **Integrator**: `showcase`, full harness run

Rule: a builder may only write inside `src/lyricsmith/<their module>/` and
its matching `tests/test_<module>.py`. Any change to `core` or to another
module's public API goes through the integrator, never edited directly by
another module's builder.

## 2. Shared data model (`core`)

All cross-module data flows through these types (dataclasses, frozen where
practical, defined in `src/lyricsmith/core/model.py`):

- `Syllable` — a phonetic syllable with its stress marker (`0`=unstressed,
  `1`=primary, `2`=secondary), per CMU dict convention.
- `Word` — text + `list[Syllable]` + phoneme string.
- `LineConstraint` — `syllable_range: tuple[int,int]`, `rhyme_slot: str |
  None` (e.g. `"A"`), `stress_pattern: str | None` (e.g. `"x/x/x/x/"` —
  optional, only enforced when the style profile specifies one), `role: str`
  (e.g. `"verse_1_line_3"`).
- `SectionRole` — enum: `VERSE`, `PRE_CHORUS`, `CHORUS`, `BRIDGE`, `OUTRO`,
  `INTRO`.
- `Section` — `role: SectionRole`, `index: int`, `lines: list[str]`,
  `constraints: list[LineConstraint]`.
- `Song` — `title: str`, `theme: str`, `genre: str`, `mood: str`,
  `sections: list[Section]`, `metadata: dict`.
- `RhymeScheme` — a string like `"ABAB"` or `"AABB"`, one letter per line
  within a section; `""` (empty string) means unrhymed/free verse.
- `GenreProfile` — see `styles` below; imported as a type here, defined
  there, to avoid a circular import (styles depends on core, not the other
  way around — core only declares the `Protocol` shape it expects).

Units/conventions:
- Syllable counts and stress markers always follow CMU Pronouncing
  Dictionary conventions (via `pronouncing`).
- All text is plain UTF-8; no markup in `Line`/`Section` content.
- Section/line indices are 0-based internally, 1-based only in
  human-facing CLI/showcase output.
- `RhymeScheme` letters are always uppercase ASCII; lowercase is reserved
  (unused for now) for "near-rhyme" scoring in a future version.

## 3. Public API per module

### `prosody`
```python
def count_syllables(text: str) -> int
def stress_pattern(text: str) -> str  # e.g. "x/x/x/x/" (x=unstressed, /=stressed)
def rhymes_with(word: str, candidate: str) -> bool
def rhyme_key(word: str) -> str  # canonical rhyme-family key for grouping
def validate_line(text: str, constraint: LineConstraint) -> ValidationResult
```
`ValidationResult = (ok: bool, errors: list[str])`. Pure functions, no I/O,
no network — this module must be usable offline and is unit-testable
without any external service.

### `styles`
```python
GENRE_PROFILES: dict[str, GenreProfile]  # "pop", "hip_hop", "country", "folk_ballad", "rock"
def get_profile(genre: str, mood: str | None = None) -> GenreProfile
```
`GenreProfile` carries: default section order (e.g.
`[VERSE, CHORUS, VERSE, CHORUS, BRIDGE, CHORUS]`), default rhyme scheme per
section role, target syllable range per section role, whether a strict
stress pattern is enforced (pop/country: loosely; hip-hop: no, prioritizes
rhythmic density over a fixed stress grid instead — see `constraints`),
and a short list of genre-appropriate imagery registers (used as generation
guidance, never as hard constraints — a chorus is not required to use them).

### `originality`
```python
def cliche_flags(text: str) -> list[str]           # named stock phrases found
def ngram_overlap(text: str, corpus: list[str], n: int = 5) -> float  # 0..1
def check(song: Song, corpus: list[str] | None = None) -> OriginalityReport
```
No bundled copyrighted lyrics ship in this package. `corpus` is caller-
supplied (optional); the cliché list is hand-curated generic AI-lyric
phrases (not copyrighted text) and is always checked.

### `constraints`
```python
def build_scaffold(genre: str, mood: str, structure: list[SectionRole] | None = None) -> Song
```
Returns a `Song` with empty `lines` but fully populated
`constraints`/`RhymeScheme` per section, ready for a `LineGenerator` to
fill. Deterministic given the same `(genre, mood, structure, seed)`.

### `generation`
```python
class LineGenerator(Protocol):
    def generate_line(self, constraint: LineConstraint, context: GenerationContext) -> str: ...

class ClaudeLineGenerator(LineGenerator):   # needs ANTHROPIC_API_KEY
    ...

class TemplateLineGenerator(LineGenerator): # offline fallback, no deps
    ...

def fill_song(scaffold: Song, generator: LineGenerator, max_retries: int = 2) -> FilledSongResult
```
`GenerationContext` carries theme, prior lines (for coherence/POV/tense
consistency), rhyme-slot target word (once the first line in a rhyme
family is written, later lines in that family receive its actual rhyme
target), and an explicit anti-cliché instruction list. Every generated line
is validated via `prosody.validate_line`; on failure the generator is
retried with the specific violation fed back as feedback, up to
`max_retries`, then the best-scoring attempt is kept with a flagged
warning (failure isolation — one bad line never crashes the whole song).

### `cli`
```python
lyricsmith generate --theme TEXT --genre GENRE --mood MOOD
                     [--structure verse,chorus,verse,chorus,bridge,chorus]
                     [--engine template|claude] [--seed INT] [--explain]
                     [--out FILE]
```
Built against this document's `constraints`/`generation` signatures, not
against their implementations — this is what lets `cli` and `generation`
build in the same wave without blocking each other.

### `showcase`
Fixture generator + gallery, built by the integrator once every module
above is integrated. Produces `showcase/gallery/*.json` (full song + scaffold
+ validation report) across the demo matrix, used by both the harness and
the critic gauntlet.

## 4. Events / logging

No pub/sub event bus (not warranted at this scale). Instead, every
generation run emits a structured `RunLog` (JSON) capturing: input params,
the frozen scaffold, every candidate line per slot (including rejected
retries and why), final song, prosody validation report, originality
report, and wall-clock timing per stage. This is what the verification
harness and critics actually read — nothing may be claimed as working
without a `RunLog` to point to.

## 5. Determinism rules

- `constraints.build_scaffold(..., seed=N)` is fully deterministic: same
  inputs + seed → byte-identical scaffold.
- `prosody` and `originality` are pure functions: fully deterministic.
- `generation` with `TemplateLineGenerator` is deterministic given a seed.
- `generation` with `ClaudeLineGenerator` is **not** claimed deterministic
  (temperature fixed at 0.7, documented, not reproducible bit-for-bit) —
  but every run's full prompt/response transcript is captured in `RunLog`
  so any output is *auditable* even though it isn't *reproducible*.

## 6. Performance budget

- Scaffold build: < 50ms.
- `prosody.validate_line`: < 10ms/line.
- Full song generation, `TemplateLineGenerator`: < 500ms end-to-end.
- Full song generation, `ClaudeLineGenerator`: < 45s end-to-end (dominated
  by LLM round-trips; budget assumes ~30-40 lines with retries).
- CLI cold start: < 300ms before generation begins.

## 7. Resource / licensing policy

- Rhyme/syllable/stress data: CMU Pronouncing Dictionary via `pronouncing`
  (public domain).
- No copyrighted song lyrics are bundled, trained on, or fetched at build
  or run time. `originality.check()` only compares against a corpus the
  caller explicitly supplies.
- `ClaudeLineGenerator` reads `ANTHROPIC_API_KEY` from the environment;
  never hardcoded, never logged (RunLog redacts the key if present in any
  captured env dump).
- No network calls anywhere in the package except the optional Anthropic
  API call in `ClaudeLineGenerator`, which is opt-in via `--engine claude`.

## 8. Failure isolation

- `prosody`/`styles`/`constraints`/`originality` are pure/offline; their
  failures are unit-test-catchable before integration and cannot crash a
  live generation run.
- A `LineGenerator` exception (e.g. API timeout) is caught per-line in
  `fill_song`; the song is still returned with the failed slot marked and
  a warning, never a hard crash — a user gets a usable partial song, not a
  stack trace.
- The CLI catches all `LyricsmithError` subclasses at the top level and
  prints a clear message; only truly unexpected exceptions propagate (and
  are asked to be reported).

## 9. Build-time note on the generation engine (read before Wave 3)

This sandbox has no `ANTHROPIC_API_KEY` and no network path to call the
Anthropic API from inside the Python process. `ClaudeLineGenerator` must
still be built as real, correct, testable code (unit tests mock the API
call). For the showcase/gauntlet fixtures generated *during this build*,
the integrator stands in for the live API call by direct reasoning (acting
as the constrained songwriter) rather than an HTTP request — this is
recorded in `RunLog.metadata["stand_in"] = true` and is never silently
presented as `ClaudeLineGenerator` output. `TemplateLineGenerator` fixtures
are real, run the actual shipped code, and are honestly MEH-tier per the
gauntlet anchors below.
