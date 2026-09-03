#!/usr/bin/env python3
"""Headless verification harness for lyricsmith.

Generates songs across a genre x mood x structure matrix, validates every
line against its constraints, and writes a full RunLog per song to
harness/runs/<timestamp>/. No agent may claim a module works without
pointing at a RunLog this harness produced.

This file is written against the PUBLIC API frozen in ARCHITECTURE.md
(constraints.build_scaffold, generation.fill_song/LineGenerator,
prosody.validate_line) before those modules exist, same as `cli`. It will
raise ImportError until Wave 2/3 land -- that's expected; run
`python harness/run_harness.py --selftest` to exercise only what already
exists (currently: core).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "harness" / "runs"

DEMO_MATRIX = [
    {"genre": "pop", "mood": "euphoric", "theme": "falling for someone at the worst possible time"},
    {"genre": "country", "mood": "bittersweet", "theme": "leaving a small town for good"},
    {"genre": "hip_hop", "mood": "defiant", "theme": "proving people wrong after being counted out"},
    {"genre": "folk_ballad", "mood": "grieving", "theme": "a grandparent's house being sold"},
    {"genre": "rock", "mood": "angry", "theme": "a friendship that quietly ended"},
]


def _json_default(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return str(obj)


def run_selftest() -> dict:
    """Exercises whatever modules currently exist. Safe to run at any point
    in the build -- this is how the integrator checks a wave landed cleanly
    without waiting for the whole pipeline to be buildable."""
    report = {"selftest_ran_at": datetime.now(timezone.utc).isoformat(), "checks": []}

    try:
        from lyricsmith.core import LineConstraint, Section, SectionRole, Song
        c = LineConstraint(role="x", syllable_range=(1, 9), rhyme_slot="A")
        s = Section(role=SectionRole.VERSE, index=0, constraints=[c])
        Song(title="t", theme="t", genre="pop", mood="happy", sections=[s])
        report["checks"].append({"module": "core", "ok": True})
    except Exception as e:
        report["checks"].append({"module": "core", "ok": False, "error": repr(e)})

    for modname in ["prosody", "styles", "originality", "constraints", "generation", "cli"]:
        try:
            __import__(f"lyricsmith.{modname}")
            report["checks"].append({"module": modname, "ok": True, "note": "imports cleanly"})
        except ImportError as e:
            report["checks"].append({"module": modname, "ok": False, "note": f"not yet built: {e}"})
        except Exception as e:
            report["checks"].append({"module": modname, "ok": False, "note": f"import ERROR: {e!r}"})

    return report


def run_full_matrix(stand_in: bool = False) -> Path:
    """Full end-to-end harness run. Requires constraints + generation to be
    integrated. `stand_in=True` uses the build-time songwriter stand-in
    documented in ARCHITECTURE.md section 9 instead of a live API call."""
    from lyricsmith.constraints import build_scaffold
    from lyricsmith.generation import TemplateLineGenerator, fill_song
    from lyricsmith.originality import check as originality_check
    from lyricsmith.prosody import validate_line

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for spec in DEMO_MATRIX:
        t0 = time.monotonic()
        scaffold = build_scaffold(genre=spec["genre"], mood=spec["mood"], structure=None)
        scaffold.theme = spec["theme"]
        generator = TemplateLineGenerator()
        result = fill_song(scaffold, generator, max_retries=2)
        elapsed = time.monotonic() - t0

        validation_reports = []
        for section in result.song.sections:
            for line, constraint in zip(section.lines, section.constraints):
                vr = validate_line(line, constraint)
                validation_reports.append({
                    "role": constraint.role, "line": line, "ok": vr.ok, "errors": list(vr.errors),
                })

        orig_report = originality_check(result.song)

        run_log = {
            "spec": spec,
            "elapsed_seconds": elapsed,
            "engine": "TemplateLineGenerator",
            "stand_in": stand_in,
            "song": dataclasses.asdict(result.song),
            "warnings": result.warnings,
            "validation_reports": validation_reports,
            "originality_report": dataclasses.asdict(orig_report),
        }
        fname = f"{spec['genre']}_{spec['mood']}".replace(" ", "_") + ".json"
        (out_dir / fname).write_text(json.dumps(run_log, indent=2, default=_json_default))
        index.append({"file": fname, "genre": spec["genre"], "mood": spec["mood"],
                       "ok": all(v["ok"] for v in validation_reports), "elapsed_seconds": elapsed})

    (out_dir / "_index.json").write_text(json.dumps(index, indent=2))
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lyricsmith verification harness")
    parser.add_argument("--selftest", action="store_true",
                         help="check which modules currently import cleanly")
    parser.add_argument("--full", action="store_true",
                         help="run the full genre x mood matrix (requires all modules)")
    args = parser.parse_args()

    if args.selftest or not args.full:
        report = run_selftest()
        print(json.dumps(report, indent=2))
        n_ok = sum(1 for c in report["checks"] if c["ok"])
        print(f"\n{n_ok}/{len(report['checks'])} checks passed", file=sys.stderr)
    if args.full:
        out_dir = run_full_matrix()
        print(f"Full harness run written to {out_dir}")
