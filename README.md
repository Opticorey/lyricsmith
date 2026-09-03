# lyricsmith

A song lyrics generator built to hit a **professional-songwriter** quality
bar, not "AI lyric generator" slop. See `ARCHITECTURE.md` for the full
design and `STATUS.json` for current build/gauntlet status.

## Quickstart

```bash
pip install -e ".[llm,dev]"

# Full-quality hybrid generation -- this is the intended path (needs your own
# Anthropic API key):
export ANTHROPIC_API_KEY=sk-...
lyricsmith generate --theme "leaving a small town for good" --genre country --mood bittersweet --engine claude

# Offline, zero-dependency fallback (--engine template, the default when no
# key is set): EXPERIMENTAL. After three gauntlet rounds it still scores
# 3-4/10 against a professional-songwriter bar -- grammatical sentences, but
# output can be incoherent, contain unresolved references, or not cohere
# into a clear narrative. See STATUS.json for the full scoring history
# before relying on it for anything but a quick structural preview.
lyricsmith generate --theme "leaving a small town for good" --genre country --mood bittersweet
```

## Per-module showcase mode

Every module has a headless demo you can run directly, e.g.:

```bash
python -m lyricsmith.prosody --demo
python -m lyricsmith.styles --demo
python -m lyricsmith.constraints --demo
```

## Verification harness

```bash
python harness/run_harness.py
```

Generates songs across a genre × mood × structure matrix and writes full
transcripts + prosody-validation reports to `harness/runs/<timestamp>/`.
