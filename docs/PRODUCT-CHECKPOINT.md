# Product checkpoint — 2026-10-16

> 历史 CV 路线记录：此文的预算、跟踪、阶段控制与验收结果不再描述默认主入口。当前入口已切换为 Qwen 逐动作视觉闭环，见 [当前主路线](VISUAL-LOOP-CHECKPOINT.md)。以下保留旧实验事实，不作为新路线成功证据。
## Acceptance status

This is an incomplete SO-ARM101 product checkpoint. Ordinary preview disables manipulation. Explicit `--live` now enables **experimental simulator-only tasks**, while the requested live conversation → Qwen → Jev → image-verified pick/place acceptance has **not passed**. This is not a hardware-control capability or a reliability certification.

Two independent blockers remain:

1. Live provider access is blocked by the current execution environment's network policy. A supported escalation failed before execution with a sandbox filesystem error. No live Qwen or Jev acceptance run occurred during this refactor.
2. The offline integrated rendered-RGB → explicitly fake Jev → real PyBullet pick stops during descent when the arm partly occludes the blue obstacle in one camera. The strict visual association guard rejects the biased center. The held-object and obstacle visibility requirements have not both passed with any tested replacement camera pair.

## What changed and was checked

- Conversation/task lifecycle is centralized in `embodied_arm/services/task_service.py`: history, observation, planning, execution feedback, cancellation, reset and duplicate-submission protection.
- The primary SO101 action is “提交并执行任务” / Enter: observe, plan and attempt execution. Optional conversation/plan preview is collapsed and never queues execution. Runtime state, not model wording such as “waiting to execute,” identifies whether a task was submitted. A missing Jev worker blocks execution before vision or brain requests; typing/chat remain independent of scene readiness.
- `embodied_arm/runtime/start_so101.py` / `scripts/run-so101.sh` provide one startup entry point. README gives deployment and current limitations.
- Tests live in `tests/`; standalone diagnostics live in `experiments/`; source packaging lives in `tools/`.
- The physical grasp path was corrected using jaw geometry; closure is gradual. Idle stop preserves completed grip force. IK convergence was tightened to avoid accumulated off-axis drift.
- RGB refinement excludes yellow jaw pixels from the red object mask. A separate offline RGB-grounded closure/lift fixture verifies a red block following a lift through rendered images.
- Phase-scoped visual tracking and bounded fixed bench-marker memory have explicit age, frame, calibration and cancellation checks. They do not read simulator object poses or contacts. Placement integration is still experimental and unaccepted.

## Evidence boundaries

Offline semantic and Jev fixtures are labelled as such. Physical tests use real simulated servos and contacts, without attaching or teleporting objects. Ground-truth capture-envelope evaluation is independent scoring, not a controller input. Successful low-level fixtures do not establish the full conversation task's success.

The explicit experimental task factory accepts only the exact `SO101Simulation` class and uses the real provider IPC clients; it does not attach hardware or silently substitute fake inference. Its `validated` flag remains false. An experimental warning is nonblocking, while runtime checks still enforce freshness, visibility, confidence, workspace, budgets and cancellation.

Experimental execution uses complete 1920×1440 frames from both cameras and `shadow_free_rgb_v1` rendering with the actual `observer_wrist` profile: a fixed observer and moving wrist camera, calibrated per frame using robot FK; the legacy camera pair is diagnostic-only. The live UI preview is 640×480 to limit preview CPU cost; high-resolution perception still costs additional CPU/processing time. The UI exposes the active resolutions, lighting profile. Ordinary preview retains the previous low-resolution setup. This selected experimental profile is not a claim that occlusion or full-task acceptance is solved.

The Jev session window is 200 calls/420 seconds; task sequences share at most 200 calls/420 seconds, with local cycles of 32 calls by default and at most 8 high-level reviews. Brain and vision windows are 10 calls/600 seconds each; brain launches with `--max-replans 8`. Session clocks include idle time from worker launch, so a newly submitted task is limited by the remaining session time. No automatic renewal or fallback occurs. Worker-only startup cannot enable an already-running preview app; the single-host `--live` entry starts the app in the intended experimental mode.

## Reproduce

- Conversation/service: `python -m pytest tests/test_task_service.py tests/test_app_so101.py tests/test_so101_ui.py`
- Image/physics acceptance probe: `python -m experiments.image_manipulation_fixture`
- Marker memory contracts: `python -m pytest tests/test_landmark_memory.py`
- Camera-survey reports are retained locally and are not included in this source-only publication
- Browser fixture scenarios: `tests/e2e_conversation/` (explicitly fake providers; separate test server)

The image/physics probe currently returns an unverified result. This failure is expected evidence of the remaining integration issue, not a passing acceptance test.

## Next concrete acceptance work

Resolve provider access using supported environment permissions. For image execution, establish a calibrated camera arrangement or validated multi-view visibility strategy that keeps the manipulated object and nearby obstacle observable through approach, closure, carry and release. Preserve identity/uncertainty guards. Then run the complete single-block pick/place flow with offline fixtures, followed by live Qwen and Jev, before claiming production reliability. The experimental simulator entry can currently be tried but may stop on its runtime guards.

## Historical targeted regression results (before experimental unlock)

- Service/UI/micro-motion/visual feedback/launcher/source-package selection: 81 passed
- Marker memory: 30 passed
- Actual rendered-RGB phase observer: 2 passed
- Default-resolution image/task precision rejection plus micro-motion suite: 16 passed after updating the fake provider's explicit readiness contract
- SO101 IPC isolated rerun: 14 passed + 4 subtests
- Known intermittent issue: the combined selection produced one IPC subtest failure (`low_confidence` expected, safe `unavailable` returned); 95 tests and 3 subtests passed in that run. The isolated rerun does not establish that the flake is fixed

These selections overlap and must not be added into a single unique-test total. At the original development checkpoint, the entire suite had not yet been rerun after the final changes. Browser lifecycle checks used explicitly fake providers in a separate test app. No final live-model or full image/physics pick/place success is asserted.

## Historical source publication regression — 2026-10-16

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
manipulation remained disabled at that historical checkpoint. Current explicit simulator-only enablement is described above.

## Subsequent package refactor

The structural refactor and deterministic diagnosis/fix of the IPC snapshot race are documented in [package verification](PACKAGE-REFACTOR.md). Its full offline regression passed; the live-model and image/physics acceptance limits above remain unchanged.

## Dynamic-loop evidence boundary

The cloud rendered-RGB → offline semantic fixture → FakeJev → PyBullet probe issued **11 controller calls and applied 10 physical actions**, then stopped on blue-obstacle occlusion. This remains a failed complete-task probe, not a live-model result or a prediction about a user's local machine. Current cloud provider access remains independently blocked; no live Qwen/Jev acceptance is asserted. See [dynamic-loop checkpoint](DYNAMIC-LOOP-CHECKPOINT.md) for the detailed scope.

## Next-phase validation boundary

The actual observer/wrist path is experimental. Pause → fresh observation → review → continuation must preserve completed primitive progress, without replaying grasp/release actions. A hidden supporting block cannot verify stack success. See [the local validation plan](NEXT-PHASE-VALIDATION.md). The earlier physical HTTP retest encountered stale initial-goal evidence; goal recording now precedes expensive tracker initialization. The completed rerun used the production factory, RGB, mocked Qwen response parsing, Jev IPC and real simulation physics: vision 1, brain 1 and Jev 4 mocked calls produced 3 physical actions, then stopped safely during approach with `perception_rejected`. One retry captured the recovery error `Moving RGB anchors unavailable`; the first tracking error was not retained, so its specific cause is unknown. `verified=false`, with no completed step or cycle review and no full grasp/stack success. All provider responses were mocked; no real API was called. First-perception-error preservation was added after this run and does not retroactively establish the missing initial error. Historical RGB/FakeJev evidence above remains unchanged.

The production active-phase semantic identity lease is now 420 seconds, bounded by the existing task deadline. Every action still requires fresh RGB association; the lease does not automatically renew and Stop invalidates it. Legacy direct fixtures retain their 120-second default. The separate 30-second current-frame age bounds processing freshness and does not authorize stale-image reuse. Request telemetry persists only whitelisted metadata in the existing runtime `last-run-audit`; public IPC results are unchanged.

Local cycle size is configurable with `--cycle-max-calls 32` (1..200); the shared 200-call/420-second task/session limits remain unchanged. The recorded four-call visual rejection happened before the cycle cap and is not explained or fixed by increasing it. While a task is active, input remains locked until Stop is acknowledged; subsequent work uses the current state rather than resetting the scene.

## Historical ROI/continuity patch (ROI feature removed)

The ROI optimization described below has been removed and these results do not describe the current full-frame path. Historical offline ROI replay recovered blue/right `roi_clipped` with one bounded 2× crop expansion on the same fresh pixels, preserving association, shape and covariance gates. Positions x=0.192 and 0.186 m passed; x=0.180 m passed RGB fitting but stopped at `excessive_3d_uncertainty` under the unchanged 3 mm limit. At x=0.176 m, fitting reported `ambiguous_face_patches`. These are local replay results, not full-grasp success.

First/retry fit diagnostics, cumulative call/action counts, cycle progress and runtime notices are now retained/displayed. Input remains locked until Stop acknowledgment; subsequent work freshly observes the current pose without scene reset. Stopped-hold continuity requires fresh RGB plus encoder evidence; helper tests do not establish physical held-object recovery. Configurable 32-call cycles and shared global limits are unchanged. No API calls were made for this patch; the final full offline regression passed 1,266 tests and 121 subtests, with 16 skipped and one existing deprecation warning (389.69 seconds). Independent review cleared the frozen patch. These tests do not establish live-model or complete grasp/stack success.

## Current full-frame path

The observer and wrist cameras now supply complete frames, with no render ROI, predicted crop, 2× crop retry or ROI toggle. Original camera intrinsics/extrinsics are unchanged; the wrist camera still uses its per-frame pose calibration. Measured full-pair capture was 3.4 seconds and the initial full-frame pipeline 12.2 seconds; these measurements are not latency guarantees. One explicit `max_observation_age_s=30` propagates through paused-scene observation paths, preserving original timestamps and robot-state checks. The production global 420-second window is unchanged. The corrected full-frame production HTTP probe used mocked providers: vision 1, brain 1 and Jev 3 calls produced 2 physical actions, then failed during approach because the previous 120-second semantic identity lease expired. The rerun and final aggregate after the production 420-second lease alignment are pending. No full grasp is established; previous ROI tests and historical failed probes remain historical evidence.
