# Product checkpoint — 2026-10-16

## Acceptance status

This is an incomplete SO-ARM101 product checkpoint. Production manipulation remains disabled. The requested live conversation → Qwen → Jev → image-verified pick/place acceptance has **not passed**.

Two independent blockers remain:

1. Live provider access is blocked by the current execution environment's network policy. A supported escalation failed before execution with a sandbox filesystem error. No live Qwen or Jev acceptance run occurred during this refactor.
2. The offline integrated rendered-RGB → explicitly fake Jev → real PyBullet pick stops during descent when the arm partly occludes the blue obstacle in one camera. The strict visual association guard rejects the biased center. The held-object and obstacle visibility requirements have not both passed with any tested replacement camera pair.

## What changed and was checked

- Conversation/task lifecycle is centralized in `embodied_arm/services/task_service.py`: history, observation, planning, execution feedback, cancellation, reset and duplicate-submission protection.
- SO101 UI has text conversation and separate capability-gated execution. A missing Jev worker blocks execution before vision or brain requests; typing/chat remain independent of scene readiness.
- `embodied_arm/runtime/start_so101.py` / `scripts/run-so101.sh` provide one startup entry point. README gives deployment and current limitations.
- Tests live in `tests/`; standalone diagnostics live in `experiments/`; source packaging lives in `tools/`.
- The physical grasp path was corrected using jaw geometry; closure is gradual. Idle stop preserves completed grip force. IK convergence was tightened to avoid accumulated off-axis drift.
- RGB refinement excludes yellow jaw pixels from the red object mask. A separate offline RGB-grounded closure/lift fixture verifies a red block following a lift through rendered images.
- Phase-scoped visual tracking and bounded fixed bench-marker memory have explicit age, frame, calibration and cancellation checks. They do not read simulator object poses or contacts. Placement integration is still experimental and unaccepted.

## Evidence boundaries

Offline semantic and Jev fixtures are labelled as such. Physical tests use real simulated servos and contacts, without attaching or teleporting objects. Ground-truth capture-envelope evaluation is independent scoring, not a controller input. Successful low-level fixtures do not establish the full conversation task's success.

The standard camera and motion defaults remain unchanged. Experimental shadow-free high-resolution rendering and alternative camera surveys are not promoted as a validated product profile. No motion button should be enabled merely because unit tests pass.

## Reproduce

- Conversation/service: `python -m pytest tests/test_task_service.py tests/test_app_so101.py tests/test_so101_ui.py`
- Image/physics acceptance probe: `python -m experiments.image_manipulation_fixture`
- Marker memory contracts: `python -m pytest tests/test_landmark_memory.py`
- Camera-survey reports are retained locally and are not included in this source-only publication
- Browser fixture scenarios: `tests/e2e_conversation/` (explicitly fake providers; separate test server)

The image/physics probe currently returns an unverified result. This failure is expected evidence of the remaining integration issue, not a passing acceptance test.

## Next concrete acceptance work

Resolve provider access using supported environment permissions. For image execution, establish a calibrated camera arrangement or validated multi-view visibility strategy that keeps the manipulated object and nearby obstacle observable through approach, closure, carry and release. Preserve identity/uncertainty guards. Then run the complete single-block pick/place flow with offline fixtures, followed by live Qwen and Jev, before enabling production execution.

## Final targeted regression results

- Service/UI/micro-motion/visual feedback/launcher/source-package selection: 81 passed
- Marker memory: 30 passed
- Actual rendered-RGB phase observer: 2 passed
- Default-resolution image/task precision rejection plus micro-motion suite: 16 passed after updating the fake provider's explicit readiness contract
- SO101 IPC isolated rerun: 14 passed + 4 subtests
- Known intermittent issue: the combined selection produced one IPC subtest failure (`low_confidence` expected, safe `unavailable` returned); 95 tests and 3 subtests passed in that run. The isolated rerun does not establish that the flake is fixed

These selections overlap and must not be added into a single unique-test total. At the original development checkpoint, the entire suite had not yet been rerun after the final changes. Browser lifecycle checks used explicitly fake providers in a separate test app. No final live-model or full image/physics pick/place success is asserted.

## Source publication regression — 2026-10-16

After source freeze and publication cleanup, the complete default offline suite
passed: **800 tests and 121 subtests passed; 16 tests skipped**, in 245.76 seconds.
Ten skips are explicitly opt-in browser tests; six require recorded image
fixtures that are not published. One existing Starlette/httpx warning remains.
JavaScript syntax validation also passed. This run did not reproduce the earlier
IPC flake; it does not establish that its cause has been fixed.

Generated simulation reports, recorded fixtures, credentials and runtime files
are not committed. The source packaging helper now excludes recorded report
fixtures as well. No live API calls were made by publication validation. The
network and image-occlusion acceptance blockers above remain unresolved, and
production manipulation stays disabled.

## Subsequent package refactor

The structural refactor and deterministic diagnosis/fix of the IPC snapshot race are documented in [package verification](PACKAGE-REFACTOR.md). Its full offline regression passed; the live-model and image/physics acceptance limits above remain unchanged.
