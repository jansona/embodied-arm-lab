# Experimental v4 validation checkpoint

Date: 2026-10-13 UTC

- Complete offline suite: **514 tests and 121 subtests passed** in 187.49 seconds
- One dependency warning: Starlette TestClient deprecates its current httpx integration
- Python compilation and embodied_arm/web/static_so101/ui.js syntax checks passed
- Actual SO101 app startup through TestClient: paired RGB cameras available,
  no tactile fields in default proprioception, paused robot, motion feature disabled
- Contact-only SO101 grasp/lift/release verified by independent physics tests;
  no object constraints or teleports
- Oracle-exclusion, uncertainty, stale-frame, cancellation, worker expiry,
  strict provider schema, wrong-direction rejection and pooled IPC tests passed
- Camera-to-Jev-to-actuator integration used explicit offline fixtures and fake
  provider responses. It is not evidence of real model perception or control

At the original offline checkpoint no live request had been made. After the user
confirmed saving the external key, three normal-app vision requests were tested:

1. Rejected response, 16.07 s end-to-end
2. Diagnostic rejection, 15.19 s: model supplied normalized coordinates despite
   the old native-pixel prompt; image bounds rejected them
3. Explicit normalized_1000 contract: six detections accepted, 18.85 s; independent
   triangulation rejected all landmarks at the unchanged 3 mm uncertainty gate

No robot motion or Jev control request occurred. Independent evaluation against a
separate deterministic initial-scene replica found red/blue top-face errors of
7.52/7.80 mm, almost entirely height bias: the model had localized near the body
center. Goal error was 1.19 mm. These oracle measurements are evaluation-only and
never enter model inputs or control targets. Confidence and low reprojection error
did not detect the common landmark bias.

The normalized-coordinate/diagnostics patch passed 75 focused tests and 15
subtests. Image-only face refinement is a pending experiment, not yet validated.

The historical Panda v3 success is separate and remains in its original archive.

## Hybrid RGB milestone, 2026-10-13 06:29 UTC

Qwen semantic regions plus an isolated RGB top-face quadrilateral refiner passed
one fresh normal-app observation. All six landmarks passed geometry checks, with
provenance `hybrid_qwen_region_rgb_geometry`, initial detector `qwen_rgb`. The
model request took 17.96 seconds end-to-end. Four live vision calls were used in
total during debugging and this milestone. Robot motion remained disabled.

On the frozen real-observation scene, independent errors after refinement were
0.84/0.88/0.66 mm for red/blue/goal. Propagated standard deviations were
2.31/2.67/2.77 mm; the 3 mm gate was unchanged. No true positions or fixed height
correction entered refinement. It used only the model-limited RGB regions.

A separate 15-case offline region-robustness test used five scene seeds and three
camera configurations, with the original Qwen regions held fixed as noisy ROI
anchors. This was NOT 15 fresh model runs. It accepted81/90 image landmarks and
36/45 stereo pairs; accepted errors averaged0.763 mm with maximum1.092 mm. All
accepted pairs passed the3 mm uncertainty gate, but only6/15 complete scenes
passed all three pairs. Nine conservative face-shape rejections remain a material
coverage limitation. Six synthetic occlusion/ambiguity/confidence failures
rejected without fabricated coordinates. No detector tuning used oracle scores.

The method assumes visible, plain-colored cuboids with distinguishable top/side
brightness and supported edges. It is not general object perception. Live SO101
hover, grasp and stacking have not yet been validated at this checkpoint.

## Scoped RGB tracking experiment (2026-10-13)

The hover controller now requests one semantic image observation and then verifies
new RGB frames locally before every Jev decision. The semantic lease is 60 seconds,
maximum 30 frames, with a 4-pixel displacement limit from the original refined
anchors. Red hover requires fresh stereo estimates of both red and blue. Only the
non-collision painted goal can be unknown; partial scenes cannot enter high-level
planning or symbolic grounding. Geometry thresholds remain unchanged.

A conservative local TCP path envelope requires 8 cm clearance from the other
image-estimated block center and a 4.5 cm minimum TCP height. This is a narrow,
static-scene hover guard, not general full-arm collision planning.

Offline actual-physics diagnostic with recorded AI pixel fixtures and synthetic
Jev responses completed 9 moves in 17.4 seconds. It made zero model calls. The
scoped tracker, scene boundary, executor and UI suite passed 72 tests. This
fixture result alone does not establish live-model hover success.

### Live trial at 06:52 UTC

One fresh Qwen stereo request was accepted (18.86 s). Real Jev selected x_minus
with confidence .86 (4.28 s), producing approximately 7.9 mm of physical TCP
motion. Fresh red and blue RGB tracking remained valid. The next Jev response
also selected x_minus but confidence was .47 (0.48 s), below the unchanged .65
acceptance threshold. The app stopped after two Jev calls and one applied action;
full hover was not completed. The residual X error was roughly 3.1 mm near the
3 mm stopping band, with image-estimate sigma approximately 2.33 mm. This single
marginal observation does not establish general model capability or calibrated
confidence. Both bounded inference workers were stopped after the trial.

### Offline uncertainty-aware follow-up

A later, offline-only executor revision uses per-axis 2σ plus a 1 mm diagnostic
servo allowance. This envelope is not calibrated coverage and does not include
unknown systematic calibration errors. An axis is eligible only when its entire
interval lies outside the original ±3 mm band; Jev still selects direction/hold.
Motion magnitude cannot cross the nearest plausible target, and responses for a
different scheduled axis are rejected. All intervals must be inside ±3 mm for
completion; otherwise, when no axis is clearly outside, execution stops with
`uncertainty_unresolved`. It does not repeatedly resample until a favorable result.

Actual-physics offline fixtures made seven moves in12.6 seconds, then correctly
stopped with unresolved uncertainty on all axes. This revision has not been run
with live models, and the currently running browser process still uses the
previous live-tested executor. No SO101 grasp/stack success is claimed.

The observed precision floor prevents strict completion: with the policy
2σ + 1 mm ≤ 3 mm, every per-axis σ must be ≤1 mm even at zero measured error.
The final diagnostic scene has σX/Y/Z =1.759/2.207/2.099 mm, so its margins are
4.518/5.414/5.199 mm. No target point can satisfy the strict completion criterion
with those margins. More language-model retries cannot resolve this measurement
floor. Camera resolution/view geometry and uncertainty calibration need further
validation, or the task tolerance would need an explicitly justified change.
The triangulator currently retains a2-pixel measurement floor; lower uncertainty
must be supported by image evidence rather than simply reducing that floor.

Final uncertainty-stage regression:603 tests and121 subtests passed; one existing
Starlette/httpx deprecation warning. The earlier legacy IPC readiness-test race
was fixed by waiting for the worker's ready file before probing its advisory lock.
Subsequent local TCP-segment clearance and UI-label changes were covered by the
focused executor/uncertainty/UI tests. No further paid calls were made.

## Optional high-resolution ROI profile (offline only)

The default640×480 camera/refiner is preserved. An explicit1920×1440 sensor
option, bounded coarse-image downsampling/remapping, a frozen opt-in footprint-
scaled refiner, and image-selected off-center ROI rendering are implemented.
ROI bounds come only from detected pixels. Unobserved pixels in the native-
coordinate canvas are black and never interpreted as an observed empty scene.
Both red and blue must remain inside the observed ROI. Full-scene planning is
not permitted from this partial observation.

The native ROI prototype preserves camera rays and yielded zero integer image
shift in six comparisons; all12 landmark accept/reject decisions matched full
rendering. Rasterized edge pixels are not bit-exact. Typical capture/refinement
is much faster than full1920 rendering, but slow frames still reject at3seconds.

Frozen profile heldout evaluation used five NEW seeds and three views, with the
original Qwen coarse ROI points reused. These are not15 fresh model runs:
-48/60 RGB landmarks and18/30 stereo pairs accepted
- Both blocks available in5/15 scenes
- Only1/15 scenes met both unchanged precision and actual3second freshness gates
- Accepted mean/max error0.385/0.752mm, independently scored after predictions
- Median capture-to-result1.44s, maximum3.79s;2frames stale
- All30 synthetic occlusion/ambiguity negatives rejected without coordinates

This is insufficient coverage for reliable image-based hover. No additional paid
model call was made to force a passing scene. The seed0 integrated offline fixture
stopped before motion because the red face remained ambiguous. The high-resolution
profile remains experimental/off by default, and neither SO101 image-based grasp
nor stacking has been validated.132 focused integration tests passed before the
final regression pass. Covariance is a heuristic measurement model, not calibrated
real-world probability; synthetic error coverage does not remove systematic risks.

Final expanded source regression:664 tests and121 subtests passed in208 seconds,
with one existing Starlette/httpx deprecation warning. No further source changes
followed this run except documentation and packaging. The opt-in high-resolution
profile remains experimental; passing software tests do not override its1/15
held-out precision-and-freshness acceptance rate.

## Public source-only checkout verification (2026-10-16)

After excluding historical runtime evidence, recorded AI pixel/RGB fixtures and
legacy replay/packaging utilities, the complete public-checkout offline suite
passed: **658 tests and 121 subtests passed; 6 tests skipped** in 192.99 seconds.
The six skipped image-regression cases require recorded RGB experiment fixtures
that are intentionally not published. One existing Starlette/httpx deprecation
warning remains. Both browser UI JavaScript syntax checks passed.

This fresh source-only result supersedes the earlier 664-test count for this
public checkout. Historical high-resolution trial inputs and their dependent
evaluators are not included, so those earlier results cannot be reproduced from
this checkout alone. No new live model calls or robot motions were made during
publication verification.
