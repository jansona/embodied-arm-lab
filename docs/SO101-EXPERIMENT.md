# SO-ARM101 RGB experiment

This separate entry point preserves the historical Panda demonstration. It is an
experimental migration, not a claim that the old Panda stack result transfers to
SO-ARM101 or to image-only control.

## Implemented and tested offline

- Pinned official SO101 printed-part geometry, five arm joints and one gripper
- Contact-based physics; no object teleport, grasp weld, or object constraint
- Stereo RGB JPEG capture with known camera calibration
- Qwen3.8-max image adapter producing paired top-face landmarks and uncertainty
- Calibrated triangulation, explicit unknowns, stale/uncertain/occluded rejection
- Qwen Chinese symbolic plan preview from estimated scene coordinates
- Jev exclusive-band local direction/hold and gripper decisions, pooled client
- Bounded visual waypoint executor; every motion requires fresh RGB perception
- Temporal RGB lift evidence, distinct from independent contact evaluation

The default SO101 inputs contain joint encoders, FK and gripper position. They do
not include simulator object poses, depth, masks, contact identities or forces.
Optional simulator tactile data is evaluation-only. The independent oracle is
not read by perception, brain, target grounding or the local executor.

## Important limits

The first camera profile freezes an idle simulation while images are interpreted.
It allows at most a 30-second frozen observation window. This is not a dynamic,
real-time camera controller. Every motion invalidates the previous observation.

The 3 mm position-standard-deviation gate accounts for declared pixel uncertainty;
it does not prove calibration accuracy or eliminate systematic VLM landmark bias.
All required landmarks must remain visible. Occlusion stops this initial profile.

Waypoint execution is a small local-target experiment, not an enabled full grasp
or stack execution path. Its default UI feature gate remains disabled until live
localization and runtime QA succeed. A 6-call waypoint budget can terminate before
reaching a distant target. No rule-based controller silently takes over.

The five-DOF arm cannot reach arbitrary six-dimensional poses. The added grasp
center is 12 mm inside the official jaw-surface frame; upstream frames are retained.
Servo collision boxes and decomposed printed collision geometry are documented in
assets/so101. Simulated dynamics are not calibrated real motor torque limits.

## Local entry point

After installing requirements in a Python 3.11 virtual environment:

    python -m uvicorn embodied_arm.web.app_so101:app --host 127.0.0.1 --port 8766

Opening the UI or preview cameras does not call a model. Inference workers are
explicit bounded processes, use the existing server-only external configuration,
and never receive API keys, endpoints or headers from browser/IPC jobs. Do not put
credentials in this directory or Git. Cloud scratch files may disappear when the
computer lifecycle ends; a repository preserves code, not external credentials.

Live SO101 image localization, waypoint accuracy, image-only grasp and stacking
have not yet been verified. The historical Panda evidence remains separate.

When the user has saved the approved external provider configuration, a normal
managed-network terminal can start a single bounded vision session explicitly:

    .venv311/bin/python -m embodied_arm.runtime.vision_worker --config /path/outside/project/config.json --max-calls 6 --max-seconds 600

The cloud browser UI and network workers may run in separate supported execution
contexts on the same cloud computer. Their private, schema-checked queues carry
only task observations/results, never credentials. There is no proxy or arbitrary
HTTP forwarding service. No worker auto-renews after expiry.

The separate SO101 microcontroller window is similarly explicit:

    .venv311/bin/python -m embodied_arm.runtime.so101_worker --config /path/outside/project/config.json --max-calls 12 --max-seconds 180

For a reviewed waypoint trial only, start the UI with SO101_ENABLE_WAYPOINT=1.
The browser can choose red/blue, but cannot submit arbitrary Cartesian targets;
it uses a fixed 35 mm offset above the observed top face. Each trial stops after
6 Jev requests or 90 seconds. A completed waypoint is not a verified grasp.

## Latest live perception result

One fresh hybrid observation has passed: Qwen identifies coarse semantic regions;
a local RGB-only quadrilateral detector refines top-face centers before stereo
triangulation. The UI names this hybrid method explicitly. Model coordinates use
a declared normalized_1000 schema, converted with width−1/height−1 scaling and
consistent uncertainty units. See SO101-VALIDATION.md for actual call counts,
accuracy and important coverage limits. Motion remains disabled by default.

## Public source and historical evaluation limits

The public checkout excludes recorded AI pixel observations and the historical
resolution, ROI-projection and scaled-holdout replay evaluators that depend on
those observations. It does not include the inputs needed to reproduce the
historical image trials described in SO101-VALIDATION.md. No substitute data is
presented as the original model output.

The offline unit tests generate synthetic scenes and use explicit mocked model
responses. They do not require live API credentials or the excluded recorded
fixture. High-resolution mode and the scaled refiner remain opt-in interfaces;
the default browser app remains the 640-pixel experimental preview. No validated
SO101 image-based pick/stack command is available.
