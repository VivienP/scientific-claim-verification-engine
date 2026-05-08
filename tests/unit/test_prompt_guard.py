"""Unit tests for src/prompt_guard.py.

The guard is a defensive prefix injected into every system prompt the
pipeline sends to the LLM. These tests check (1) the guard contains the
key phrases an audit reviewer expects (so a future "tightening" doesn't
silently remove them), and (2) every callable surface that talks to the
LLM has the guard prepended.
"""

from __future__ import annotations

from src.prompt_guard import PROMPT_INJECTION_GUARD


class TestGuardContent:
    """The guard must contain every phrase below; missing any indicates
    a defensive layer was inadvertently weakened."""

    def test_marks_user_content_as_untrusted(self) -> None:
        assert "untrusted user data" in PROMPT_INJECTION_GUARD

    def test_explicitly_lists_known_injection_patterns(self) -> None:
        # Normalise newlines so multi-line wrapped phrases still match.
        normalised = " ".join(PROMPT_INJECTION_GUARD.split())
        for pattern in [
            "ignore previous instructions",
            "respond only with",
            "you are now a different assistant",
        ]:
            assert pattern in normalised, f"guard missing injection pattern: {pattern!r}"

    def test_instructs_to_use_prescribed_failure_sentinel(self) -> None:
        # If the LLM gets confused, it must NOT emit a refusal that breaks
        # the JSON parser downstream — it must use the schema-prescribed
        # failure sentinel.
        normalised = " ".join(PROMPT_INJECTION_GUARD.split())
        assert "not_addressed" in normalised
        assert "schema-prescribed failure sentinel" in normalised


class TestEveryPromptHasTheGuard:
    """Every LLM-facing system prompt in the codebase must include the guard."""

    def test_extract_prompt_has_guard(self) -> None:
        from src.extract import _SYSTEM_PROMPT

        assert PROMPT_INJECTION_GUARD in _SYSTEM_PROMPT

    def test_abstract_verifier_prompt_has_guard(self) -> None:
        from src.verify_prompts import _SYSTEM_PROMPT

        assert PROMPT_INJECTION_GUARD in _SYSTEM_PROMPT

    def test_fulltext_verifier_prompt_has_guard(self) -> None:
        from src.verify_prompts import _FULLTEXT_SYSTEM_PROMPT

        assert PROMPT_INJECTION_GUARD in _FULLTEXT_SYSTEM_PROMPT

    def test_title_only_verifier_prompt_has_guard(self) -> None:
        from src.verify_prompts import _TITLE_ONLY_SYSTEM_PROMPT

        assert PROMPT_INJECTION_GUARD in _TITLE_ONLY_SYSTEM_PROMPT

    def test_citing_context_verifier_prompt_has_guard(self) -> None:
        from src.verify_prompts import _CITING_CONTEXT_SYSTEM_PROMPT

        assert PROMPT_INJECTION_GUARD in _CITING_CONTEXT_SYSTEM_PROMPT

    def test_numeric_extractor_prompt_has_guard(self) -> None:
        from src.numeric.extract import _SYSTEM_PROMPT

        assert PROMPT_INJECTION_GUARD in _SYSTEM_PROMPT
