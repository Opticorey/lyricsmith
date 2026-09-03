# Build prompt — lyricsmith (filled in)

# Goal
Build **lyricsmith, a Python library + CLI for generating professional-quality
song lyrics** using **Python 3.11+, a hybrid rule/LLM generation engine
(`pronouncing`/CMU dict for rhyme+syllable+stress analysis, the `anthropic`
SDK for the creative line-fill step, `click` for the CLI, `pytest` for
tests)** from **an empty repo (greenfield)**.
The bar is **professional human songwriter craft** — lyrics with precise,
consistent imagery, natural conversational prosody that actually sings,
emotional specificity, and hooks that land — not another AI tool's output.
Never **AI-slop lyrics**: forced rhymes (moon/June), generic cliché imagery
("shadows of my mind," "chasing the light"), inconsistent POV/tense,
meter-breaking filler lines, or a structure that technically satisfies a
rhyme scheme but says nothing.

# How to work
1. **Architecture first.** Before feature code, write ARCHITECTURE.md:
   module boundaries, shared data model, public API per module, events,
   units/conventions, determinism rules, performance budget, licensing
   policy, failure isolation.
2. **Build the verification loop before the product.**
   A headless harness that **generates songs across a genre × mood ×
   structure matrix and captures full transcripts + prosody-validation
   reports**, plus a per-module `--demo` showcase mode. No agent may claim
   anything it hasn't captured and inspected.
3. **Fan out.** One builder per module, owning only its folder.
   Waves ordered by dependency: **[core (foundational, built directly) →
   Wave 1: prosody, styles, originality] → [Wave 2: constraints] →
   [Wave 3: generation, cli] → [integrator: showcase + full harness run]**.
   Between waves, one integrator applies core changes and fixes seams.
4. **Gauntlet.** A separate critic that writes no code scores each module
   0–10 against **professional-songwriter craft**, with anchors at
   **10 = indistinguishable from a working pro's polished draft / PASS ≥7 =
   solid demo-quality draft, real craft, minor rough edges / MEH 4–6 =
   functional but generic and mechanical, reads as AI-written / FAIL <4 =
   broken meter or rhyme, incoherent, unmistakable slop**.
   Pass = ≥7 with zero errors. Below that: ranked issues, retry,
   max **3** rounds. At round 3 failure: **mark the module BLOCKED in
   STATUS.json with the critic's full ranked issue list, ship the
   best-scoring attempt with a visible caveat, and escalate rather than
   loop forever.**
5. **Final gate.** Whole-system critic, then blind pairwise comparison
   against **reference songwriting excerpts (public-domain/user-supplied,
   never scraped copyrighted lyrics)** by **an independent judge subagent
   with no memory of the build, given two unlabeled samples**.
6. **Loop until pass.** Persist scores and open issues to STATUS.json;
   resume from the weakest module.

# Rules
- Never inflate scores. Report real numbers, failed rounds, what's missing.
- Never edit another module's folder. Core changes go through the integrator.
- Keep **the harness's ability to run an end-to-end `generate` call**
  live at all times; others depend on it.
- Do not ask questions. Decide, state assumptions, keep going.
- Budget: stop at **one full pass through all waves + one critic round per
  module (up to 3 for any module scoring below PASS) + the final gate, or
  ~3 hours of wall-clock build time — whichever comes first**, and report
  status.
Start now.

---

## Notes on the two decisions this template couldn't make for itself

**No live Anthropic API key is available in this build sandbox** (checked:
`ANTHROPIC_API_KEY` is unset, no network path to call the Anthropic API from
inside the sandboxed Python process). This matters because the generation
engine is a **hybrid**: an algorithmic scaffold enforces structure/rhyme/
meter as hard constraints, and an LLM fills the actual line content within
them. Two consequences, both documented in ARCHITECTURE.md and STATUS.json
rather than papered over:

1. The shipped library ships a real `ClaudeLineGenerator` (uses your own
   `ANTHROPIC_API_KEY` at runtime — this is the path that can actually reach
   the professional-songwriter bar) and a zero-dependency
   `TemplateLineGenerator` fallback so the CLI still runs offline/free, at
   noticeably lower (MEH-tier) quality.
2. For *this build's* showcase and critic gauntlet, I stand in for the live
   API using my own reasoning directly (acting as the constrained
   songwriter) rather than an HTTP call, so the gauntlet can honestly score
   hybrid-quality output. That stand-in is clearly labeled in the harness
   output and is not part of the shipped `src/` package.

**Originality/licensing policy**: the `originality` module never bundles or
trains on real copyrighted lyrics (that would itself be a licensing
problem). It does two things instead: (a) generic n-gram overlap checking
against a corpus you optionally supply yourself, and (b) a built-in
cliché-phrase detector (generic AI-lyric stock phrases, not copyrighted
text) that doubles as an anti-slop quality gate. Rhyme/syllable/stress data
comes from the CMU Pronouncing Dictionary (public domain) via the
`pronouncing` package.
