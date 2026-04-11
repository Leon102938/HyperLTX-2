from __future__ import annotations

from agent_core.adapters.base import StubAdapter


class StoryboardStubAdapter(StubAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="storyboard_stub",
            kind="storyboard",
            reason="Storyboard planning hook reserved for future phases; not enabled in Phase 1.",
        )
