# Qwen / Jev role and output contracts

These are implemented request contracts, not evidence that live-model manipulation succeeds. No live API calls are needed to preview or test them.

## Qwen brain

`providers/brain.py` injects a system instruction on each planning request. Qwen chooses object identities, symbolic goals, and up to six ordered `pick`/`place` steps. It returns the existing strict JSON plan; no extra success, reasoning, coordinate, or joint-command fields are added. The short `message` explains the decision or limitation. Response budget remains 1,024 tokens with thinking disabled.

The planner is lower-frequency than the cerebellum: initial planning and bounded event-driven replanning, not camera-frame inference. RGB detection is a separate Qwen request with its own strict pixel-output contract. A validated structured scene provides image-derived coordinates to planning; the planner does not receive hidden simulator object truth on the RGB route.

Optional `scene.runtime_capabilities` contains exactly:

- `robot_name`: `SO-ARM101` or `Panda`
- `execution_requested`, `experimental`: booleans
- `supported_actions`: unique subset of `pick`, `place`, `sequence`
- `supported_targets`: unique subset of `goal`, `on:red`, `on:blue`
- `controller`: `jev` or `deterministic`
- `completion_verification`: `temporal_rgb` or `physics`

Executable plans are rejected when outside the declared capabilities. SO101 execution requires explicit pick/place steps rather than legacy `pick` plus `goal` shorthand. Capability absence preserves legacy behavior; this is not a declaration that every runtime supports every planner action. Unavailable observations still permit only clarify/status/stop, even when capabilities are supplied. `perception_source=rgb_only` and configured `cube_edge_m` of 0.02 or 0.05 survive the scene whitelist. The latter scales support/occupancy checks for the actual block size.

## Jev cerebellum

`providers/so101_jev.py` injects role and output instructions in `questions.action.instructions`, as required by `/systemone` and `decision-model-preview`. It does not add chat messages, a system-role field, arbitrary task text, or new state fields.

One request produces one categorical microdecision from fresh local state:

- Scheduled Cartesian axis: positive, negative, or hold
- Gripper: full open, full close, or hold

The caller schedules an axis; Jev chooses its sign. Existing signed-error criteria and action bounds remain unchanged. Confidence below 0.65 or inconsistent answers are rejected without a substitute action. Hold means a local tolerance criterion; it is not proof of grasp, placement, or task completion. The runtime owns fresh observation, grounding, IK, safety checks, cancellation, and application of each accepted action. Frequent Jev calls do not authorize queued stale-state actions or bypass observation checks.

## Paused-cycle feedback

Optional `execution_feedback.cycle_context` preserves bounded evidence across a pause:

- Integer `cycle_index`, `completed_micro_actions`, `remaining_global_calls`, each 0–200
- `held_object`: red/blue/null; non-null requires `held_verified=true`. Null with false is unknown; null with true is verified empty
- `phase`: approach/descend/preclose/close/lift/verify_lift/transit/lower/release/retreat/verify_release
- `residual_xyz_m`: finite three-vector or null
- `stop_reason`: cycle_budget/scene_changed/controller_error
- `observation_status`: fresh/unavailable
- `recent_actions`: at most eight exact `{action, applied}` records using the existing nine atomic action names and a boolean

An unfinished primitive may be confirmed using a singleton pick/place with exactly the active action, object, and target. Singleton place accepts a known other block (`on:red`/`on:blue`) as well as `goal`, subject to runtime capabilities; self-placement is rejected. This confirmation resumes an internal execution checkpoint, including release-pending verification, rather than pretending a fresh grasp/place. Unknown holding after release must not be changed to verified-held just to satisfy planning.

Existing completed-step prefix and failed-step checks remain mandatory. A cycle cap means pause, obtain fresh observation, and consider the remaining task. It never resets global budgets or implies return to the initial scene. Qwen's adapter accepts an explicit `max_replans` from 0–8, default 1; cumulative model-call limits remain independent and are not reset by replanning. Runtime/IPC wiring must supply the chosen limits and actual evidence.

Grasp completion requires fresh temporal observation of lifting with the gripper; release completion requires observed placement. Planning acceptance, closing the gripper, local target arrival, and a model's message do not establish success.

## Offline validation

`tests/test_model_role_contracts.py` verifies injected role text and protocol shape, bounded output, capabilities and provenance through IPC normalization, unavailable scenes, unsupported actions/goals, 20 mm supporting-block protection, strict bounded cycle evidence, and cumulative configurable replans. `tests/test_brain_cycle_budgets.py` verifies eight replans through a bounded mock IPC worker, replacement-bridge enforcement, legacy default one, malformed status/cap rejection, and CLI propagation without configuration reads. Existing provider/IPC tests remain applicable. These tests use mock HTTP responses and do not establish live model reliability.

## Request observability and ambiguous timeouts

Brain, vision, and SO101 Jev adapters expose `last_request`, a single in-memory `provider-request-telemetry-v1` record. It contains component/model, Unix start time, elapsed seconds, fixed outcome, HTTP status when received, a format-checked response request ID, and only bounded integer token-usage fields reported by the provider. Missing usage is an empty object, not a claim of zero consumption. Nonstandard request-ID formats are omitted. No prompt, image, authorization header, raw error, or response body is recorded.

`processing_uncertain=true` on pending, timeout, or network-error outcomes. A timeout can happen after a provider processed a paid request; it is not proof of zero processing or zero token usage. The adapters perform one POST per explicit call with no automatic retries (`automatic_retry=false`). Qwen brain and vision discard responses exceeding their elapsed timeout; Jev uses HTTP phase timeouts with additional runtime worker deadlines.

This metadata does not change strict public model result or IPC schemas. A private helper envelope carries separately revalidated metadata to each worker's existing bounded `last-run-audit.json`, one record per processed request. Missing or malformed metadata is explicitly null with `request_metadata_available=false` and conservative processing uncertainty; killed helpers cannot provide a response request ID or token usage. No automatic retry is introduced. The audit can support manual correlation with a provider request ID when both sides expose one, but does not independently prove billing attribution or task completion. The UI does not yet display provider usage/request IDs; runtime round-trip timing is a separate measure that includes IPC overhead. `tests/test_request_telemetry.py` exercises mock success, rejection, HTTP error, ambiguous timeout, and all three worker audit paths, including secret-key stripping and malformed metadata.

## Cycle progress evidence and current robot state

Optional `cycle_context.progress` contains at most eight exact records: `phase`, `goal_start_xyz_m`, `goal_end_xyz_m`, `error_start_xyz_m`, `error_end_xyz_m`, `object_start_xyz_m`, `object_end_xyz_m`, and `observations`. Phase uses the existing cycle phase enum. Every xyz component is finite and between −1 and +1 metre; observation count is an integer from 1 to 1,000. No precomputed progress verdict is accepted. The runtime supplies fresh RGB object positions and FK-derived errors; this validator checks structure and bounds, not provenance by itself.

Qwen judges progress from this evidence. It compares error within the same phase and local goal, accounts for changing goals, and does not infer trends from one observation or missing evidence. Shrinking residuals are not grasp or placement completion. Current observations, object movement, and verification state determine whether continuing, revising, or clarifying is appropriate.

Optional `scene.robot_state` is exactly `{joints, gripper_openness, holding_status}`: five finite joint values (absolute value below 100), normalized openness in [0,1], and unknown/verified_held/verified_empty. It describes the actual current state after Stop, not an assumed reset posture. A supplied `held_object` must agree with verified_held when robot state is present. Encoder/gripper state alone does not establish an object identity or valid vision; an unavailable scene remains conversation-only. Legacy scenes and cycle contexts without these optional fields remain valid.
