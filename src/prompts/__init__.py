"""Versioned prompt loader.

Prompts are stored as markdown files in this directory using the convention
`{stage}_v{N}.md` (e.g. `verify_v1.md`, `extract_v1.md`). Each file contains
ONLY the prompt body — `PROMPT_INJECTION_GUARD` is prepended at load time.

Why split: the guard is a security primitive that lives in `src/prompt_guard.py`.
The body is creative content that `@prompt-smith` iterates on. Keeping them
separate prevents accidental guard mutation during prompt tuning.

Behavior contract: for any prompt that was previously a Python constant of
the form `PROMPT_INJECTION_GUARD + "\\n" + BODY`, the migrated equivalent
`load_prompt(name)` MUST return a byte-identical string.
"""

from __future__ import annotations

from pathlib import Path

from src.prompt_guard import PROMPT_INJECTION_GUARD

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str, *, with_guard: bool = True) -> str:
    """Load a prompt by name without the .md extension.

    Args:
        name: prompt filename stem (e.g. "verify_v1", "extract_v1")
        with_guard: if True (default), prepend PROMPT_INJECTION_GUARD + "\\n"

    Returns:
        The assembled prompt string.

    Raises:
        FileNotFoundError: if the markdown file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    body = path.read_text(encoding="utf-8")
    if with_guard:
        return PROMPT_INJECTION_GUARD + "\n" + body
    return body


__all__ = ["load_prompt"]
