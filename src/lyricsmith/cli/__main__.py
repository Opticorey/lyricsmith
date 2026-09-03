"""Per-module showcase mode convention (see `lyricsmith.core.__main__` for
the reference shape): `python -m lyricsmith.cli --demo` prints a small,
human-readable demonstration of this module's behavior using only its own
public API.

For `cli` specifically, a live human demo of a CLI *is* the CLI itself --
so "demo" here means invoking the `generate` command in-process via click's
`CliRunner`, with `--engine template` (the offline engine, always available,
no `ANTHROPIC_API_KEY` needed) and printing whatever it produces, including
its exit code. This deliberately goes through the exact same `cli` Group
object real users invoke -- it is not a separate demo code path.
"""
import argparse

from click.testing import CliRunner

from lyricsmith.cli.main import cli


def demo() -> None:
    print("=== cli module demo ===")
    print("Running: lyricsmith generate --theme ... --genre pop --mood hopeful "
          "--engine template --explain")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "--theme",
            "chasing a dream against long odds",
            "--genre",
            "pop",
            "--mood",
            "hopeful",
            "--engine",
            "template",
            "--seed",
            "7",
            "--explain",
        ],
    )

    print(f"--- exit code: {result.exit_code} ---")
    print(result.output)
    if result.exception and not isinstance(result.exception, SystemExit):
        print(f"--- unexpected exception: {result.exception!r} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cli module showcase")
    parser.add_argument("--demo", action="store_true", help="run the demo")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
