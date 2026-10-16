# Observer + wrist RGB camera profile

The `observer_wrist` profile uses exactly two cameras for both preview and perception:

- `left`: a fixed calibrated observer, 50° vertical field of view
- `right`: a 95° vertical-field-of-view camera on a rigid virtual bracket beside the wrist. Its position in the jaw TCP frame is (−55, −35, −70) mm; it points toward (0, 0, 35) mm in that same frame

The wrist pose is measured joint-encoder forward kinematics multiplied by a fixed mount transform. It does not aim at simulator object locations. Preview uses the same rig at 640×480; native perception can retain 1920×1440 and ROI capture. `legacy_fixed_stereo` is an explicitly named regression/debug profile.

## Per-capture calibration

The simulation lock freezes one physical state while both RGB images are rendered. The same optical transform drives each view matrix and its camera calibration. Each capture includes a frame ID, monotonic capture timestamp, the two calibrations, the camera profile, and per-view role, calibration source, frame ID and timestamp. The source and phase observer validate the bindings and compare the calibration against the paused encoder pose. Cropped rendering uses each camera's actual field of view.

Camera geometry remains local. The model vision packet still contains only the frame ID, timestamp and two RGB images; no object poses, depth, renderer segmentation, contact identities or evaluation metrics are added.

## Moving-view tracking

The original image-derived triangulated points predict fresh search regions after camera movement. A predicted pixel is never returned as a measurement. Each current image must independently pass the existing visible-face fit, confidence and uncertainty checks; current per-frame calibrations then triangulate those actual fitted pixels. Static association residuals are measured against the projected original world anchor, not the old image coordinate. Carried-face area comparisons account for the changed projected footprint without widening the existing identity gates.

Only wrist extrinsics may change in this profile. Observer pose and both cameras' optics remain fixed. The same semantic lease, frame budgets, 3-second freshness limit, uncertainty limits, missing-view rejection and poor-ray-angle rejection remain in force. Reprojected ROIs preserve native sensor coordinates. If a target leaves the image, a required surface is occluded, or geometry becomes degenerate, observation fails explicitly.

The fixed goal memory retains its original RGB timestamp, covariance and calibration evidence. Wrist movement does not restamp this historical measurement. Step transitions capture new RGB before transferring tracked identities; a place step also requires a prior temporal-RGB verified-held certificate. That certificate never substitutes for a fresh image.

## Verification and limits

Offline tests cover actual bounded servo motion, invariant observer geometry, moving FK-mounted wrist geometry, per-frame/stale-calibration rejection, preview/perception equivalence, synthetic-marker rendering/projection agreement, and actual rendered-RGB tracking across a 1 cm motion without oracle reads. The synthetic marker is calibration evaluation only; its known position is never a controller input.

This is still a simulated, known-colour cuboid profile. It does not establish hardware hand–eye calibration, general object perception, reliable continuous video control, or successful grasp/stack execution. Close views may lose other objects. An equal-size cube can hide the support top after stacking; that hidden top cannot be claimed as a fresh visible measurement. High-resolution capture and fitting consume CPU and are not guaranteed to meet a high control frequency.
