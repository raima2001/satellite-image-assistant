"""Bonus 3.B demo - show the skill loading decision for two turns.

No live model needed: calls the same matching function agent.graph uses
(agent.skills.match_skill) on two sample analyst messages, and shows that the
skill's body is added to the prompt only for the turn that matches it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.skills import load_skill_index, match_skill

TURNS = [
    "Which Sentinel-2 bands should I use to compute a vegetation index for this scene?",
    "What's the cloud cover on the second scene in my shortlist?",
]


def main() -> None:
    skills = load_skill_index()
    print(f"Skill index (frontmatter only, loaded once): {[s.name for s in skills]}\n")

    for message in TURNS:
        matched = match_skill(message, skills)
        print(f"Analyst: {message}")
        if matched:
            print(f"  -> loads skill '{matched.name}' for this turn only")
            print(f"  skill body added to prompt:\n{matched.body()}\n")
        else:
            print("  -> no skill matches; prompt unchanged\n")


if __name__ == "__main__":
    main()
