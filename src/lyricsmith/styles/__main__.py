"""Per-module showcase mode convention (every lyricsmith module follows this
shape): `python -m lyricsmith.<module> --demo` prints a small, human-readable
demonstration of that module's behavior using only its own public API, with
no network calls and no dependency on modules later in the build order.

Shape copied from lyricsmith/core/__main__.py.
"""
import argparse

from lyricsmith.styles import GENRE_PROFILES, get_profile


def demo() -> None:
    print("=== styles module demo ===")
    print(f"{len(GENRE_PROFILES)} genre profiles: {', '.join(GENRE_PROFILES)}")
    print()

    for genre_name in GENRE_PROFILES:
        profile = get_profile(genre_name, mood="bittersweet")
        order = " -> ".join(role.value for role in profile.section_order)
        print(f"--- {profile.name} ---")
        print(f"  section order ({len(profile.section_order)}): {order}")
        for role in dict.fromkeys(profile.section_order):  # unique, in order
            rhyme = profile.rhyme_scheme_by_role[role] or "(free)"
            lo, hi = profile.syllable_range_by_role[role]
            stress = "enforced" if profile.stress_enforced_by_role[role] else "free rhythm"
            print(f"    {role.value:<11} rhyme={rhyme:<6} syllables={lo}-{hi:<3} stress={stress}")
        print(f"  imagery: {', '.join(profile.imagery_registers)}")
        print()

    print("--- unknown genre error handling ---")
    try:
        get_profile("emo_rap_but_not_a_real_bucket")
    except Exception as exc:  # noqa: BLE001 -- demo only, show the message
        print(f"  get_profile('emo_rap_but_not_a_real_bucket') raised: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="styles module showcase")
    parser.add_argument("--demo", action="store_true", help="run the demo")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
