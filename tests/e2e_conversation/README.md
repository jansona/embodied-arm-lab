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

## Bounded dynamic-loop acceptance (2026-10-16)

Executed independently:

```sh
python -m pytest -q tests/e2e_conversation/test_dynamic_loop.py \
  tests/test_jev_motion.py tests/test_task_service.py tests/test_brain.py \
  tests/test_phase_visual_feedback.py tests/test_image_task_integration.py \
  tests/test_so101_ui.py
# 171 passed (31 new independent dynamic-loop checks)
```

The new checks exercise observation-before-decision, changed targets reversing the next Jev action, per-action motion bounds, stale/occluded evidence failing closed, cancellation, invalid covariance/targets, shared call/deadline budgets, exactly one permitted safe high-level repair, fresh repair context, no replan for unsafe outcomes, frame changes invalidating a repair, and explicit accounting. Four local observations do not produce four high-level model calls. Accepted but unverified motion never becomes task completion.

Independent testing found and reproduced a late-result bug: an executor could return `verified:true` after the shared deadline and still complete. The implementation now rejects it with `time_budget`; the regression passed. Missing or invalid bounded-executor accounting also fails closed.

Evidence labels matter: most tests use MOCK observations, a fake high-level brain and FakeJev/executor contracts. One test runs the real SO101 servo/physics with FakeJev and a synthetic moving target; it forbids evaluation-state access, joint/body teleport and constraint attachment during execution. It proves corrective servo motion, not image-derived target accuracy or successful grasping. Existing image-integration tests supply separate rendered-RGB fail-closed evidence. These checks do not establish a full successful RGB→Qwen→Jev pick/place task, and no live API was called. No new browser-runner result is claimed for this dynamic-loop checkpoint.

Final focused rerun: 32 dynamic-loop checks passed after adding a preclose regression. A fresh 12 mm visual-object displacement from an initially aligned encoder pose rejects gripper closure alignment; this uses a synthetic RGB estimate contract, not simulator object ground truth. The builder added the corresponding preclose guard following independent review.

## Production HTTP / IPC / physics path (2026-10-16)

```sh
python -m pytest -q -s tests/e2e_conversation/test_production_http_loop.py
```

Executed final run: passed in 90.60 seconds (including expiration of the bounded isolated IPC worker). The HTTP task traversed the production app factory, QwenVision parser, scaled RGB refinement, QwenBrain parser, manipulation executor, FileSO101Bridge, worker, SO101JevAdapter parser and real SO101 physics. Only external provider HTTP responses used MockTransport; no fake application executor/source/simulation was injected. Placeholder keys are non-secret test strings, and no network/model account was accessed.

Observed: one vision request, one planning request, twelve Jev decisions, ten physical simulation actions. During descent, static RGB association rejected initial-anchor drift; one stationary re-observation failed, and the task ended `execution_unverified` / `perception_rejected`, with `verified:false`. This proves the execution path can move and report a safe failure, not successful grasping or live model quality. Earlier runs included one pre-motion failure with an unavailable truncated diagnostic, one action-producing pass, and one artificial 40-second test timeout; therefore this is not a timing-stability claim. Final test bounds are a 75-second wait and 90-second IPC lease; production safety gates were not relaxed.

Separate supported cloud-browser check on a FAKE application fixture: primary execute enabled when ready; idle Stop disabled; Enter submitted execution without diagnostic buttons; active Stop cancelled and late results did not complete; structured plan and execution logs visible; original model text explicitly labeled a proposal; offline Enter preserved the draft and made no additional request. The Python Playwright runner remains unexecuted. QA tabs and the new isolated port8771 fixture were closed afterward.
