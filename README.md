# Qwen → Jev → Virtual Arm

CPU-only robotics sandbox with a browser view, a Qwen conversation planner, and a Jev micro-action controller. The user-selected models are Alibaba Cloud Bailian `qwen3.8-max` and `decision-model-preview`.

## Published scope

This public checkout contains the Panda v3 source baseline only. The baseline source checkpoint is preserved on `baseline/panda-v3`. The experimental SO101 v4 implementation has not been published here.

Historical experiment records, runtime/session artifacts, credentials and private configuration are not included. The results below summarize prior development trials; this checkout does not contain their original evidence or reproduce those sessions automatically.

## Historical development results

- Real Qwen understood a red-block pick, a pronoun-based placement follow-up, and an unknown-object clarification
- Real Jev completed a physical red-block pick: 37 requests, 36 applied actions, block lifted to about25.8cm; the already accepted Qwen plan was reused for that recovery
- Qwen generated an abstract stacking plan: pick red → place red in the goal area → pick blue → place blue on red
- Earlier stacking attempts stopped visibly on a gripper-confidence rejection and then on measured grasp drift. The former was corrected with clearer exclusive transition criteria. The latter exposed a missing Panda finger mimic linkage
- The corrected robot-only finger gear linkage reduced held-block drift from8.65mm/12 simulated seconds to0.0067mm/30 seconds. Both stacking orders pass deterministic physical regression tests
- Full real Qwen-plan/Jev stacking replay passed:167 Jev requests,165 applied actions, all four symbolic steps physically verified. Blue was released on red with0.375mm relative horizontal error and0.025mm support displacement. The original accepted Qwen plan was reused with zero additional brain calls

Recorded outcomes are bounded demonstrations, not general reliability estimates. The scripted/mock modes are explicitly labelled and never silently replace a failed model call.

## Architecture

1. **Qwen brain:** converts conversation into validated symbolic `pick`, `place`, `stop`, `status`, or `clarify` intents, or an ordered sequence of up to six steps. Known objects are red/blue; placement targets are the goal or the other block
2. **Deterministic geometry and phase decomposition:** grounds the chosen symbols in fresh measured poses, generates approach/release waypoints, schedules X/Y/Z phases, and checks physical outcomes
3. **Jev cerebellum:** chooses positive/negative/hold within the current axis phase, or open/close/hold for the fingers, from measured JSON state and explicit decision criteria. All candidates remain available. The program does not substitute a rule-chosen model answer
4. **Physical servo:** bounded Cartesian steps, inverse kinematics, motor position control and contact simulation execute accepted decisions. This is not model-generated torque control

A separate deterministic controller remains available as a baseline. The Panda is not an SO-ARM101 digital twin; a future hardware backend requires its own geometry, calibration, sensing and emergency-stop design.

## Local setup

Python3.11 is recommended so PyBullet can use a prebuilt wheel.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765 on that same computer. It is a loopback development app, not a public website. The prepared cloud installation uses `.venv311`.

The browser offers chat, explicit brain/controller selection, a live camera, manual joints/end-effector/fingers, pause, stop, reset, worker availability and measured task results. Stop invalidates pending responses and freezes physics; reset creates a fresh episode. Never expose this prototype publicly.

## Cloud desktop / managed-network split

In this environment the visible desktop and managed-network shell are separate execution namespaces. The desktop runs physics/UI, while explicitly started inference workers use a private fixed-schema application queue. This is not an arbitrary HTTP proxy: queue jobs cannot supply URLs, headers or keys.

Visible desktop terminal:

```sh
./run-ipc.sh
```

Approved managed-network execution environment, separate terminals:

```sh
.venv311/bin/python brain_worker.py --config ../jev-arm-config.json --max-calls 6 --max-seconds 600
.venv311/bin/python inference_worker.py --micro-phased --max-calls 200 --max-seconds 420
```

For an ordinary computer where the browser and API networking share one environment, run `./run-session.sh` to launch the UI and both bounded workers with one command. In this cloud split, keep the visible desktop server and run `./run-session.sh --workers-only` in the approved managed-network environment. The launcher sends no request until an explicit UI task and never renews a window automatically.

Then select live Qwen and Jev in the UI, load the server configuration and submit a task. Workers terminate at their explicit call/time budgets; they do not restart automatically. The offline mock brain uses no API and is visibly labelled. Normal dialogue sequences use the explicitly shown200-request/420-second Jev preset. Stop still cancels immediately; limits are maximums, not promised usage.

## Credentials and provider

`config.example.json` is an empty template. The actual key belongs in `../jev-arm-config.json`, outside the project, entered and saved by the owner. No browser credential entry, client-side storage, source-control key, or key-bearing log is needed. This repository excludes the actual configuration and private runtime queues.

- Jev: `POST https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/systemone`, model `decision-model-preview`, typed `state` + `questions` schema
- Qwen: same provider base with `/chat/completions`, model `qwen3.8-max`
- Exact destinations/models are allowlisted; TLS verification stays on; redirects are refused
- State is text/JSON, not camera images. Camera rendering is for human observation
- The configured confidence threshold is0.65. It is an experimental gate, not a calibrated probability of safety
- Errors, malformed responses, low confidence and stale responses stop execution. There is no automatic model or controller fallback
- A provider may charge an in-flight request even after Stop; cancellation prevents application of its result, not necessarily billing

Provider documentation: https://docs.bailian.console.aliyun.com/zh/model-studio/token-plan-decision-model

## Physics and safety boundaries

PyBullet DIRECT, CPU TinyRenderer, bundled Franka Panda assets, fixed240Hz simulation. Motor-driven fingers grasp ordinary dynamic cubes through contacts/friction. No cube is teleported during a task or attached to the gripper. The only explicit constraint is the Panda's finger9↔finger10 gear linkage, implementing its mechanical mimic relation; reset creates initial poses.

Controller movement is at most2cm per accepted micro-action, capped at remaining target distance. Joint/workspace/finite-value checks and nonprogress rejection remain independent of model confidence. A deterministic terminal tolerance check stops finished waypoints before requesting another move; this is not attributed to a Jev hold answer. Pause/reset/stop generations reject stale decisions. These checks are not a full collision planner or real-robot safety certification.

The continuation utilities are narrow, provenance-checked experiment recovery tools. They require exact saved-plan/checkpoint evidence and reject changed scenes; they are not generic remote-control proxies. The historical checkpoints and audit records required to replay the original sessions are not included in this repository.

## Tests and evidence

```sh
.venv311/bin/python -m pytest -q
.venv311/bin/python evaluate.py
```

Tests cover provider schemas/errors, cancelled and stale responses, private IPC validation, bounded worker budgets, physical pick/place/stack, finger stability, conversation plans and recovery guards. They use mocked API transport unless a script is explicitly described as a live diagnostic. Do not run live scripts merely to run the offline test suite.

Historical pick/stack summaries, gripper diagnostics and continuation audits are excluded. The offline tests use their own generated or mocked inputs; running `evaluate.py` creates a new local evaluation report rather than restoring historical trial evidence.

Cold request latency includes connection/proxy/TLS setup. Reused-client warm requests have often been roughly0.5–1second in these trials; this is observed end-to-end latency, not guaranteed model inference time. A provider `latency_ms` field is retained separately and is not assumed to cover the entire request.

Main modules: `brain.py`, `brain_ipc.py`, `brain_worker.py`, `microcontrol_experiment.py`, `inference_worker.py`, `ipc.py`, `sequence_executor.py`, `stack_plans.py`, `task_executor.py`, `control.py`, `simulator.py`, `app.py`, and `static/`.

## Source-only publication

Only the Panda v3 source baseline is published. Historical demonstration evidence, runtime/session records, credentials and private configuration are excluded. SO101 v4 publication remains pending.

Publication excludes these legacy runtime replay/audit packaging utilities: `gripper_context_replay.py`, `microcontrol_context_replay.py`, `microcontrol_hold_audit.py`, `package_project.py`. The offline tests do not depend on them.
