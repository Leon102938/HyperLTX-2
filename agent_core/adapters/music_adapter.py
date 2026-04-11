from __future__ import annotations

from agent_core.adapters.base import StubAdapter


class MusicAdapter(StubAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="music_stub",
            kind="music",
            reason="Music backend interface reserved for future phases; not enabled in Phase 1.",
        )
