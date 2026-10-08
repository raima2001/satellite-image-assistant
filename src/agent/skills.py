"""Bonus 3.B - Agent Skills.

A skill is a directory under skills/ with a SKILL.md: frontmatter (name,
description) followed by a body. At startup only the frontmatter is read, so
the resident index is cheap - the body is read and added to the prompt only
on a turn whose message matches a skill's description, never on every turn.
"""

from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"
MIN_TRIGGER_WORD_LEN = 4
_STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "for", "are", "use",
    "uses", "used", "when", "does", "which", "asks", "asked", "like",
    "load", "into",
    # Generic to this assistant's domain, so they would match almost any turn.
    "about", "analyst", "asset", "assets", "band", "bands", "compute",
    "correspond", "explains", "names", "sentinel2",
}


@dataclass
class Skill:
    name: str
    description: str
    path: Path

    def body(self) -> str:
        _, _, rest = self.path.read_text().split("---", 2)
        return rest.strip()


def _parse_frontmatter(text: str) -> dict:
    _, frontmatter, _ = text.split("---", 2)
    fields = {}
    for line in frontmatter.strip().splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def load_skill_index() -> list[Skill]:
    """Read only the frontmatter of every skill - name and description."""
    skills = []
    if not SKILLS_DIR.is_dir():
        return skills
    for skill_file in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        fields = _parse_frontmatter(skill_file.read_text())
        skills.append(
            Skill(
                name=fields["name"],
                description=fields["description"],
                path=skill_file,
            )
        )
    return skills


def _words(text: str) -> set[str]:
    return {
        "".join(char for char in word if char.isalnum())
        for word in text.lower().split()
    }


def _trigger_words(description: str) -> set[str]:
    words = _words(description)
    return {word for word in words if len(word) >= MIN_TRIGGER_WORD_LEN and word not in _STOPWORDS}


def match_skill(user_text: str, skills: list[Skill]) -> Skill | None:
    """Pick the first skill whose description shares a whole word with the message.

    A plain keyword overlap on the frontmatter description decides whether to
    load a skill - no extra trigger config, no model call just to route it.
    """
    words = _words(user_text)
    for skill in skills:
        if words & _trigger_words(skill.description):
            return skill
    return None
