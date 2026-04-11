from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from agent_core.schemas import BackendCapabilities, ExecutionResult, JobInput, ProductionPlan


class BackendAdapter(ABC):
    name = "base"
    kind = "base"
    phase1_enabled = False

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError


class VoiceAdapter(BackendAdapter):
    kind = "voice"
    phase1_enabled = True

    @abstractmethod
    def generate_voice(self, job: JobInput, plan: ProductionPlan, workspace: Path) -> ExecutionResult:
        raise NotImplementedError


class VideoAdapter(BackendAdapter):
    kind = "video"
    phase1_enabled = True

    @abstractmethod
    def generate_video(
        self,
        job: JobInput,
        plan: ProductionPlan,
        workspace: Path,
        voice_result: ExecutionResult | None = None,
    ) -> ExecutionResult:
        raise NotImplementedError


class StubAdapter(BackendAdapter):
    phase1_enabled = False

    def __init__(self, *, name: str, kind: str, reason: str):
        self.name = name
        self.kind = kind
        self.reason = reason

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind=self.kind,  # type: ignore[arg-type]
            available=False,
            phase1_enabled=False,
            transport="stub",
            notes=[self.reason],
        )
