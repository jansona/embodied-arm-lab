# Conversation browser checks

These tests drive the actual static UI and FastAPI HTTP application in a new browser context, using **FAKE-E2E-NOT-QWEN** and a fake robot. They make no external API requests, use no credentials, and provide no evidence about real Qwen/Jev inference, physics, grasp success, or image recognition accuracy.

Install the project's requirements plus `playwright` in the test environment, and install Chromium through Playwright's official installer if needed. Then, from the project root:

```sh
E2E_BROWSER=1 python -m pytest -q tests/e2e_conversation/test_browser.py
# For an existing system Chromium installation:
E2E_BROWSER=1 E2E_CHROMIUM_PATH=/usr/bin/chromium python -m pytest -q tests/e2e_conversation/test_browser.py
```

The conversation flows cover text replies with explicit unknown observation; Enter/Shift+Enter and duplicate submission; Stop and discarded late replies; truthful provider failure and retry; reload during a pending response without duplicate inference; expired service/reconnect with draft preservation; and honest refusal to claim unimplemented execution.

The final test is a **safety contract**, not product acceptance for conversational robot tasks. End-to-end task acceptance still needs text → real Qwen structured task → image-only observation → Jev control → visible progress → verified completion or explicit bounded failure, plus mid-motion Stop and interruption recovery. Do not count the current preview-only flow as passing that requirement.

## Execution status (2026-10-16)

Python bytecode compilation and pytest collection pass. Fake-provider FastAPI TestClient smoke checks pass for both conversation and task completion. The Python Playwright runner itself has **not run**: the authoring shell cannot launch Chromium. Tests are opt-in so the ordinary offline suite skips them explicitly; skipped tests are not passing browser evidence.

The same flows were exercised through the supported cloud-browser automation against isolated native localhost fixture servers:

- Text → reply with explicitly unknown scene and FAKE provenance: passed
- Duplicate click → one provider request; disabled send during pending: passed
- Stop during pending reply, discard late response, recover send: passed
- Provider failure → truthful no-motion message → successful retry: passed after UI correction
- Reload during inference → restored history without duplicate request: passed
- Service expiry → disabled send, editable preserved draft; reconnect restores send: passed
- Task text → synthetic observation → fake plan → fake executor progress → accepted completion flag: passed
- Reload while executing → same in-flight task without duplicate request: passed
- Stop during fake execution → cancelled state survives late success: passed
- Unverified fake executor result → failure/stopped/unconfirmed completion: passed

These browser checks use fake providers, synthetic scene contracts and a fake executor. They do **not** validate image recognition, robot physics, grasping, Qwen or Jev. Real-provider acceptance remains separately blocked by provider-network policy; no route-around was attempted.

A P1 raw `planning_failed` message was found, fixed by the UI owner, and verified in browser to show a helpful failure/no-motion explanation. Low-priority untranslated execution/event details were also fixed and rechecked in the browser.

## Interactive fixture server

On a supported local runtime:

```sh
python -m tests.e2e_conversation.serve_fixture
# http://127.0.0.1:8767, controls at /__qa
python -m tests.e2e_conversation.serve_fixture --tasks
# http://127.0.0.1:8768, explicitly fake validated executor
```

These servers bind localhost and must never be deployed. They expose deliberate fake-provider controls, no credentials or model calls. Close them after QA.

## Automatic observation regression (2026-10-16, package-layout branch)

Run the independent HTTP contract checks:

```sh
python -m pytest -q tests/e2e_conversation/test_auto_observation.py
python -m tests.e2e_conversation.serve_auto_observation
# MOCK-only interactive browser fixture: localhost:8770, controls /__qa
```

Executed: 10 HTTP checks passed against the new automatic-observation implementation. They cover first relevant chat acquiring a scene without manual recognition, unavailable vision blocking brain inference, stale-scene refresh, Stop discarding late observation, reconnect, valid-scene reuse, missing brain configuration, RGB-only preview boundaries, continuous local capture with zero provider calls, and stale/non-motion plan labeling.

Executed through the supported cloud-browser interface: first red-cube instruction with no diagnostic clicks; visible observation/planning/non-motion structured plan; idle Stop disabled and active Stop enabled; stale-plan warning and on-demand refresh; unavailable vision message explicitly says no upload is needed; cancelled observation never produces a late plan; reload/reconnect; offline brain disables Send while preserving the editable draft. Local frame IDs advanced at approximately one second while provider counters stayed unchanged until a request.

These results use MOCK scene and provider contracts, not real image recognition or Qwen/Jev. The image-only boundary was also reviewed in the production source: local sampling strips the packet to RGB frame fields, carries no provider handle, and simulator camera capture is serialized. Full real-provider/physical acceptance is outside this evidence. The Python Playwright runner remains unexecuted in this environment; do not equate authored/collected tests with browser-runner passes.
