from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from agent_core.adapters import HiDreamStoryboardAdapter, LTX2Adapter, MusicAdapter, QwenTTSAdapter, StoryboardStubAdapter
from agent_core.adapters.base import BackendAdapter
from agent_core.schemas import BackendCapabilities


class BackendRegistry:
    def __init__(self, adapters: Iterable[BackendAdapter] | None = None) -> None:
        self._adapters: dict[str, BackendAdapter] = {}
        self._kind_index: dict[str, list[str]] = defaultdict(list)
        for adapter in adapters or ():
            self.register(adapter)

    def register(self, adapter: BackendAdapter) -> None:
        self._adapters[adapter.name] = adapter
        self._kind_index[adapter.kind].append(adapter.name)

    def get(self, name: str) -> BackendAdapter:
        return self._adapters[name]

    def adapters_for_kind(self, kind: str) -> list[BackendAdapter]:
        return [self._adapters[name] for name in self._kind_index.get(kind, [])]

    def primary(self, kind: str) -> BackendAdapter | None:
        for adapter in self.adapters_for_kind(kind):
            capability = adapter.capabilities()
            if capability.available and capability.phase1_enabled:
                return adapter
        return None

    def primary_capability(self, kind: str) -> BackendCapabilities | None:
        adapter = self.primary(kind)
        return adapter.capabilities() if adapter else None

    def snapshot(self) -> dict[str, BackendCapabilities]:
        return {name: adapter.capabilities() for name, adapter in self._adapters.items()}

    def supports_video_pipeline(self, pipeline_name: str) -> bool:
        adapter = self.primary("video")
        if adapter is None:
            return False
        capability = adapter.capabilities()
        return pipeline_name in capability.supported_pipelines


def build_default_registry(base_url: str = "http://127.0.0.1:8000") -> BackendRegistry:
    return BackendRegistry(
        adapters=[
            QwenTTSAdapter(base_url=base_url),
            LTX2Adapter(base_url=base_url),
            HiDreamStoryboardAdapter(base_url=base_url),
            MusicAdapter(base_url=base_url),
            StoryboardStubAdapter(),
        ]
    )
