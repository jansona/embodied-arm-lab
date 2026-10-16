# SO-ARM101 sandbox assets

The yellow printed-part geometry and robot kinematics come from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), pinned to
commit `a758567c3978dfeefe282ede0500085a48fe8f78` (retrieved 2026-10-13).
Original model: `Simulation/SO101/so101_new_calib.urdf`. This is the **follower**,
not the leader/teleoperation handle. The source repository uses Apache-2.0;
its full license is included in `LICENSE`. `source_manifest.json` records exact
source URLs, original Git blob hashes, sizes and SHA-256 digests.

Attribution: Standard Open SO-100 & SO-101 Arms, TheRobotStudio and contributors,
including Rob Knight, Pepijn Kooijmans, Remi Cadene, Simon Alibert, Michel Aractingi,
Dana Aubakirova, Adil Zouitine, Russi Martino, Steven Palma, Caroline Pascal and
Thomas Wolf, as listed in the pinned upstream `CITATION.cff`.

## Deliberate modifications

1. Servo meshes `sts3215_03a_v1.stl` and `sts3215_03a_no_horn_v1.stl` are **not
   redistributed**. Their visual and collision geometry is replaced with authored
   25 × 45 × 35 mm black boxes. These are illustrative envelopes, not precision
   servo CAD. Upstream [issue #179](https://github.com/TheRobotStudio/SO-ARM100/issues/179)
   asks about servo CAD provenance; this avoids depending on that unresolved asset
   question. Vendor STEP assemblies are not included.
2. The two printed gripper parts use convex-decomposed collision geometry rather
   than a single convex hull that would fill their concave openings. Seven fixed
   jaw hulls and five moving-jaw hulls were generated locally from the pinned
   printed meshes using the already-installed PyBullet 3.2.7 `vhacd` function.
   Conversion consumed geometry as data; no upstream executable code was run.
   Parameters and generated-file hashes are in the manifest. Each hull is a
   separate URDF collision element. Visual meshes are unmodified.
3. All official joint frames, joint limits, masses and inertia tensors remain
   unchanged. One new fixed link, `sandbox_grasp_center_link`, is placed 12 mm
   into the gripper aperture from `gripper_frame_link` (local offset −X). The
   official frame lies on the fixed jaw's inside surface, so centering an object
   there would intersect the finger. The new frame is a nominal grasp center
   for 20 mm cubes. It is not a dynamically inferred object pose.

## Robot model contract

- Five arm joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
  `wrist_roll`; one rotating `gripper` joint; no mimic joints
- The sandbox has eight Bullet joints: six actuated and two fixed
- Revolute angles are radians and distances are meters
- Gripper commands are normalized 0–1, linearly mapped to the URDF angular limits
  −0.174533 to 1.74533. This is not a measured finger-aperture calibration
- IK controls XYZ plus the TCP +Z approach direction, with rotation about that
  direction left free. Arbitrary six-dimensional poses are not supported
- Home is approximately [0.20, 0, 0.06] m, with downward approach. A rectangular
  workspace bound is only an initial guard; the IK residual and joint-limit
  checks reject geometrically unreachable targets within that box
- Tabletop grasp points around x=0.15–0.30 m are viable; top-down reach is limited
  at greater heights. Do not reuse Panda's 20–25 cm lift heights
- Motors use bounded simulation torques, with 3.35 N·m arm torque inspired by
  upstream `joints_properties.xml`. Gains, friction and servo boxes are simulation
  approximations, not a validated real-hardware digital twin

Default `proprioception()` reads only robot encoders and FK; it does not even query
simulated contacts. Real SO101 default hardware has no tactile sensors. A separate
`evaluation_tactile()` method provides explicitly labeled simulation-only oracle
aggregates for independent tests, never model input. It exposes per-jaw contact
count and summed normal force, excluding self-contact, without object IDs, names,
positions, normals or classifications. No method provides true cube coordinates.
Cubes are never attached or teleported; gripping uses
contact and friction. PyBullet self-collision is not enabled; collision avoidance
against the arm itself is not claimed.

## Verification

`test_so101_robot.py` verifies provenance, link/joint names, rendered pixels,
gravity hold, pure FK against Bullet, analytical Jacobians, IK residuals/limits,
invalid-command rejection, gripper travel, and a physical cube grasp/lift without
constraints. Tests use object state only for independent evaluation assertions.
