"""Audit .claude/ for orphans, broken references, and duplicate names.

Adapted from Garry Tan's `check-resolvable` pattern (gbrain) to our slash-command
based architecture (no RESOLVER.md / AGENTS.md natural-language dispatcher).

Checks:
  1. Every agent referenced by a command exists.
  2. Every skill referenced by an agent's `skills:` frontmatter exists.
  3. No two agents share the same `name`.
  4. No orphan skills (skills not referenced by any agent or command).

Exit code: 0 if clean, 1 if any error.

Run from repo root: `python scripts/check_resolvable.py`
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_DIR = REPO_ROOT / ".claude"

AGENT_REF_RE = re.compile(r"@([a-z][a-z0-9-]*)")


@dataclass
class Agent:
    name: str
    path: Path
    skills: list[str] = field(default_factory=list)


@dataclass
class Command:
    name: str
    path: Path
    body: str = ""


@dataclass
class Skill:
    name: str
    path: Path


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Return (frontmatter_dict, body). Empty dict if no frontmatter."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    fm_raw = text[4:end]
    body = text[end + 5 :]
    parsed = yaml.safe_load(fm_raw) or {}
    if not isinstance(parsed, dict):
        return {}, body
    return parsed, body


def _load_agents(agents_dir: Path) -> list[Agent]:
    agents: list[Agent] = []
    for path in sorted(agents_dir.glob("*.md")):
        fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
        name = str(fm.get("name") or path.stem)
        raw_skills = fm.get("skills") or []
        skills = [str(s) for s in raw_skills] if isinstance(raw_skills, list) else []
        agents.append(Agent(name=name, path=path, skills=skills))
    return agents


def _load_commands(commands_dir: Path) -> list[Command]:
    commands: list[Command] = []
    for path in sorted(commands_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        fm, body = _split_frontmatter(text)
        name = str(fm.get("name") or path.stem)
        commands.append(Command(name=name, path=path, body=body or text))
    return commands


def _load_skills(skills_dir: Path) -> list[Skill]:
    """Skills are either flat (.claude/skills/foo.md) or nested (.claude/skills/foo/SKILL.md).

    Files inside a directory that already contains a SKILL.md are treated as
    sub-documents of that multi-file skill, NOT standalone skills.
    Top-level files whose name is ALL_CAPS (e.g. SUPERPOWERS_PROVENANCE.md)
    are treated as documentation, not skills.
    """
    multi_file_dirs = {p.parent for p in skills_dir.rglob("SKILL.md")}
    skills: list[Skill] = []
    for path in sorted(skills_dir.rglob("*.md")):
        if path.name == "SKILL.md":
            skills.append(Skill(name=path.parent.name, path=path))
            continue
        if path.parent in multi_file_dirs:
            continue
        if path.parent == skills_dir and path.stem.isupper():
            continue
        skills.append(Skill(name=path.stem, path=path))
    return skills


def _find_skill_references(text: str, skill_names: set[str]) -> set[str]:
    """Return skill names mentioned in `text` (word-boundary match)."""
    found: set[str] = set()
    for name in skill_names:
        if re.search(rf"\b{re.escape(name)}\b", text):
            found.add(name)
    return found


def _agents_referenced_by(command: Command, known_agents: set[str]) -> set[str]:
    """Return the subset of @agent-name tokens in the command body that match known agents."""
    refs: set[str] = set()
    for match in AGENT_REF_RE.finditer(command.body):
        token = match.group(1)
        if token in known_agents:
            refs.add(token)
    return refs


def main() -> int:
    if not CLAUDE_DIR.is_dir():
        sys.stderr.write(f"ERROR: {CLAUDE_DIR} does not exist\n")
        return 1

    agents = _load_agents(CLAUDE_DIR / "agents")
    commands = _load_commands(CLAUDE_DIR / "commands")
    skills = _load_skills(CLAUDE_DIR / "skills")

    known_agent_names = {a.name for a in agents}
    known_skill_names = {s.name for s in skills}

    errors: list[str] = []
    warnings: list[str] = []

    name_to_agents: dict[str, list[Agent]] = defaultdict(list)
    for a in agents:
        name_to_agents[a.name].append(a)
    for name, group in name_to_agents.items():
        if len(group) > 1:
            paths = ", ".join(str(g.path.relative_to(REPO_ROOT)) for g in group)
            errors.append(f"Duplicate agent name '{name}' in: {paths}")

    skill_referenced_by: dict[str, list[str]] = defaultdict(list)
    for agent in agents:
        agent_text = agent.path.read_text(encoding="utf-8")
        for skill_name in agent.skills:
            if skill_name not in known_skill_names:
                errors.append(
                    f"Agent '{agent.name}' ({agent.path.relative_to(REPO_ROOT)}) "
                    f"references missing skill '{skill_name}' in frontmatter"
                )
            else:
                skill_referenced_by[skill_name].append(f"agent:{agent.name}")
        for found in _find_skill_references(agent_text, known_skill_names):
            skill_referenced_by[found].append(f"agent-body:{agent.name}")

    for command in commands:
        for found in _find_skill_references(command.body, known_skill_names):
            skill_referenced_by[found].append(f"command:{command.name}")

    for extra_path in (REPO_ROOT / "CLAUDE.md", *(REPO_ROOT / ".claude" / "rules").glob("*.md")):
        if extra_path.is_file():
            text = extra_path.read_text(encoding="utf-8")
            for found in _find_skill_references(text, known_skill_names):
                skill_referenced_by[found].append(f"doc:{extra_path.name}")

    for command in commands:
        refs = _agents_referenced_by(command, known_agent_names)
        if not refs:
            unknown_refs = {
                m.group(1)
                for m in AGENT_REF_RE.finditer(command.body)
                if m.group(1) not in known_agent_names
                and not m.group(1).startswith(("http", "https", "v"))
            }
            if unknown_refs:
                warnings.append(
                    f"Command '{command.name}' "
                    f"({command.path.relative_to(REPO_ROOT)}) "
                    f"mentions @-tokens that are not known agents: {sorted(unknown_refs)}"
                )

    orphan_skills = sorted(known_skill_names - set(skill_referenced_by.keys()))

    print(f"Agents:    {len(agents)} found")
    print(f"Commands:  {len(commands)} found")
    print(f"Skills:    {len(skills)} found")
    print()
    if orphan_skills:
        print(f"Orphan skills (no agent references them): {len(orphan_skills)}")
        for s in orphan_skills:
            print(f"  - {s}")
        print()

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
        print()

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        print()
        print("Status: FAIL")
        return 1

    print("Status: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
