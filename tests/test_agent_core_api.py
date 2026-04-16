import threading
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from agent_core.schemas import JobInput, ResultSummary
from agent_core.state_store import StateStore
from app import agent_core_api
from app.main import app


class FakeBridgeAgent:
    def __init__(
        self,
        state_store: StateStore,
        *,
        fail: bool = False,
        started_event: threading.Event | None = None,
        release_event: threading.Event | None = None,
    ) -> None:
        self.state_store = state_store
        self.fail = fail
        self.started_event = started_event
        self.release_event = release_event

    def load_job(self, source) -> JobInput:
        job = source if isinstance(source, JobInput) else JobInput.model_validate(source)
        if not job.job_id:
            job.job_id = "bridge-generated-job"
        return job

    def run_job(self, source, *, raise_on_error: bool = False) -> ResultSummary:  # noqa: ARG002
        job = self.load_job(source)

        state = self.state_store.initialize(job)
        self.state_store.transition(state, "validated", "bridge validated job")
        self.state_store.transition(state, "planned", "bridge created plan")

        if self.started_event is not None:
            self.started_event.set()
        if self.release_event is not None:
            self.release_event.wait(timeout=5)

        job_dir = self.state_store.job_dir(job.job_id)
        final_path = job_dir / "final.mp4"
        final_path.write_bytes(b"fake-mp4")

        if self.fail:
            self.state_store.fail(state, "forced bridge failure")
            result = ResultSummary(
                job_id=job.job_id,
                success=False,
                final_phase="failed",
                message="forced bridge failure",
                output_final_path=None,
                output_video_path=None,
                output_audio_path=None,
            )
        else:
            self.state_store.transition(state, "assembled", "bridge assembled result")
            self.state_store.transition(state, "done", "bridge finished job")
            result = ResultSummary(
                job_id=job.job_id,
                success=True,
                final_phase="assembled",
                message="bridge success",
                planned_duration_sec=4.0,
                actual_final_duration_sec=4.0,
                output_final_path=str(final_path),
                output_video_path=str(final_path),
                output_audio_path=None,
            )
        self.state_store.save_result(state, result)
        return result


class AgentCoreApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = StateStore(Path(self.tmpdir.name) / "runs")
        self.runner = agent_core_api.BackgroundJobRunner()
        app.dependency_overrides.clear()

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.tmpdir.cleanup()

    def _client(
        self,
        *,
        fail: bool = False,
        started_event: threading.Event | None = None,
        release_event: threading.Event | None = None,
    ) -> TestClient:
        agent = FakeBridgeAgent(
            self.store,
            fail=fail,
            started_event=started_event,
            release_event=release_event,
        )
        app.dependency_overrides[agent_core_api.get_state_store] = lambda: self.store
        app.dependency_overrides[agent_core_api.get_video_agent] = lambda: agent
        app.dependency_overrides[agent_core_api.get_job_runner] = lambda: self.runner
        return TestClient(app)

    def _wait_for_status(self, client: TestClient, job_id: str, expected_statuses: set[str]) -> dict:
        for _ in range(80):
            payload = client.get(f"/agent-core/jobs/{job_id}").json()
            if payload["status"] in expected_statuses:
                return payload
            time.sleep(0.05)
        self.fail(f"job {job_id} did not reach statuses {sorted(expected_statuses)}")

    def _wait_for_payload(self, client: TestClient, job_id: str, predicate, description: str) -> dict:
        for _ in range(80):
            payload = client.get(f"/agent-core/jobs/{job_id}").json()
            if predicate(payload):
                return payload
            time.sleep(0.05)
        self.fail(f"job {job_id} did not reach expected payload condition: {description}")

    def test_run_endpoint_returns_success_contract_and_refs(self) -> None:
        client = self._client()

        response = client.post(
            "/agent-core/run",
            json={
                "job": {
                    "job_id": "bridge-success",
                    "idea": "A modular bridge accepts a job.",
                    "duration_sec": 4,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["success"])
        self.assertEqual(payload["job_id"], "bridge-success")
        self.assertEqual(payload["status"], "assembled")
        self.assertTrue(payload["is_terminal"])
        self.assertFalse(payload["should_poll"])
        self.assertIsNone(payload["retry_after_sec"])
        self.assertTrue(payload["artifacts_ready"])
        self.assertTrue(payload["final_mp4_ready"])
        self.assertTrue(payload["result_json_ready"])
        self.assertEqual(payload["result"]["output_final_path"], payload["refs"]["final_mp4_path"])
        self.assertEqual(payload["public_refs"]["final_mp4_url"], payload["refs"]["final_mp4_url"])
        self.assertTrue(payload["refs"]["result_json_path"].endswith("/result.json"))
        self.assertTrue(payload["refs"]["state_json_url"].endswith("/agent-runs/bridge-success/state.json"))

    def test_async_submit_returns_accepted_contract_and_poll_url(self) -> None:
        release_event = threading.Event()
        client = self._client(release_event=release_event)

        response = client.post(
            "/agent-core/jobs",
            json={
                "job": {
                    "job_id": "bridge-accepted",
                    "idea": "A modular bridge accepts async jobs.",
                    "duration_sec": 4,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                }
            },
        )
        release_event.set()

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["job_id"], "bridge-accepted")
        self.assertEqual(payload["status"], "accepted")
        self.assertEqual(payload["current_phase"], "received")
        self.assertIsNone(payload["success"])
        self.assertFalse(payload["is_terminal"])
        self.assertTrue(payload["should_poll"])
        self.assertEqual(payload["retry_after_sec"], 2)
        self.assertFalse(payload["artifacts_ready"])
        self.assertFalse(payload["final_mp4_ready"])
        self.assertFalse(payload["result_json_ready"])
        self.assertIn("accepted", payload["status_summary"].lower())
        self.assertTrue(payload["poll_url"].endswith("/agent-core/jobs/bridge-accepted"))
        self._wait_for_status(client, "bridge-accepted", {"done"})

    def test_async_status_endpoint_reports_running_then_done(self) -> None:
        started_event = threading.Event()
        release_event = threading.Event()
        client = self._client(started_event=started_event, release_event=release_event)

        submit_response = client.post(
            "/agent-core/jobs",
            json={
                "job": {
                    "job_id": "bridge-status",
                    "idea": "A modular bridge exposes async status.",
                    "duration_sec": 4,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                }
            },
        )

        self.assertEqual(submit_response.status_code, 202)
        self.assertTrue(started_event.wait(timeout=2))

        response = client.get("/agent-core/jobs/bridge-status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["job_id"], "bridge-status")
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["current_phase"], "planned")
        self.assertIsNone(payload["success"])
        self.assertFalse(payload["is_terminal"])
        self.assertTrue(payload["should_poll"])
        self.assertEqual(payload["retry_after_sec"], 3)
        self.assertFalse(payload["artifacts_ready"])
        self.assertFalse(payload["final_mp4_ready"])
        self.assertFalse(payload["result_json_ready"])
        self.assertEqual(payload["public_refs"]["final_mp4_url"], payload["refs"]["final_mp4_url"])

        release_event.set()
        done_payload = self._wait_for_status(client, "bridge-status", {"done"})
        self.assertTrue(done_payload["success"])
        self.assertTrue(done_payload["is_terminal"])
        self.assertFalse(done_payload["should_poll"])
        self.assertIsNone(done_payload["retry_after_sec"])
        self.assertTrue(done_payload["artifacts_ready"])
        self.assertTrue(done_payload["final_mp4_ready"])
        self.assertTrue(done_payload["result_json_ready"])
        self.assertEqual(done_payload["current_phase"], "done")
        self.assertEqual(done_payload["result"]["final_phase"], "assembled")
        self.assertIn("all final artifacts are ready", done_payload["status_summary"].lower())
        self.assertTrue(done_payload["refs"]["final_mp4_url"].endswith("/agent-runs/bridge-status/final.mp4"))

    def test_async_status_endpoint_reports_failed_job(self) -> None:
        started_event = threading.Event()
        release_event = threading.Event()
        client = self._client(fail=True, started_event=started_event, release_event=release_event)

        response = client.post(
            "/agent-core/jobs",
            json={
                "job": {
                    "job_id": "bridge-failure",
                    "idea": "A modular bridge returns failed result contracts.",
                    "duration_sec": 4,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                }
            },
        )

        self.assertEqual(response.status_code, 202)
        self.assertTrue(started_event.wait(timeout=2))
        release_event.set()
        payload = self._wait_for_payload(
            client,
            "bridge-failure",
            lambda item: item["status"] == "failed" and item["is_terminal"] and item["result_json_ready"],
            "terminal failed payload with ready failure result",
        )
        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["current_phase"], "failed")
        self.assertTrue(payload["is_terminal"])
        self.assertFalse(payload["should_poll"])
        self.assertIsNone(payload["retry_after_sec"])
        self.assertTrue(payload["artifacts_ready"])
        self.assertFalse(payload["final_mp4_ready"])
        self.assertTrue(payload["result_json_ready"])
        self.assertIn("failure result is ready", payload["status_summary"].lower())
        self.assertEqual(payload["error"]["type"], "agent_core_job_failed")
        self.assertTrue(payload["refs"]["result_json_path"].endswith("/result.json"))

    def test_run_endpoint_rejects_invalid_job_payload(self) -> None:
        client = self._client()

        response = client.post(
            "/agent-core/run",
            json={"job": {"idea": "", "script": "", "resolution": "tiny"}},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
