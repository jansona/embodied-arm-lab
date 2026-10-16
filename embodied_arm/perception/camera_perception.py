"""RGB-only perception contracts and fail-closed calibrated stereo geometry.

No simulator, depth, segmentation, body IDs, object poses, or networking enter this
module. The detector supplies matching *visible landmarks*, never world XYZ.
Cameras are rectified pinholes: optical +X right, +Y down, +Z forward; extrinsics
map optical coordinates to world metres. Timestamps share the caller's monotonic
clock. Cube landmarks are top-face centres, NOT body/grasp centres. A later,
explicit known-object geometry transform may use those landmarks.

The covariance propagates independent image-pixel uncertainty through stereo
geometry. It does not certify model correctness or include unmodelled systematic
camera-calibration errors; callers must calibrate/validate those separately.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

OBJECT_IDS = ("red", "blue", "goal")
LANDMARKS = {"red": "top_face_center", "blue": "top_face_center", "goal": "goal_center"}
VISIBILITIES = frozenset(("visible", "occluded", "absent", "ambiguous", "unknown"))
MAX_JPEG_BYTES = 8 * 1024 * 1024


def _number(value: Any, name: str, lo: float = -1e12, hi: float = 1e12) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be a finite number in [{lo}, {hi}]")
    return float(value)


def _id(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= 128
            or any(not (c.isalnum() or c in "_-.:") for c in value)):
        raise ValueError(f"Invalid {name}")
    return value


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} requires {size} finite numbers")
    return tuple(_number(v, name) for v in value)


def _fields(value: Any, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Invalid {name} fields")
    return value


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_from_camera: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        _id(self.camera_id, "camera_id")
        for name in ("width", "height"):
            if type(getattr(self, name)) is not int or not 16 <= getattr(self, name) <= 4096:
                raise ValueError("Image dimensions must be integers in [16, 4096]")
        for name in ("fx", "fy"):
            object.__setattr__(self, name, _number(getattr(self, name), name, 1, 1e6))
        for name, upper in (("cx", self.width), ("cy", self.height)):
            object.__setattr__(self, name, _number(getattr(self, name), name, 0, upper - 1))
        matrix = self.world_from_camera
        if not isinstance(matrix, (list, tuple)) or len(matrix) != 4:
            raise ValueError("world_from_camera requires a rigid 4x4 transform")
        matrix = tuple(_vector(row, 4, "world_from_camera") for row in matrix)
        transform = np.asarray(matrix)
        rotation = transform[:3, :3]
        if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                or not math.isclose(float(np.linalg.det(rotation)), 1, abs_tol=1e-6)):
            raise ValueError("world_from_camera must be a proper rigid transform")
        object.__setattr__(self, "world_from_camera", matrix)

    def to_dict(self) -> dict:
        return {"camera_id": self.camera_id, "width": self.width, "height": self.height,
                "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "world_from_camera": [list(row) for row in self.world_from_camera]}

    @classmethod
    def from_dict(cls, value: dict) -> "CameraCalibration":
        return cls(**_fields(value, {"camera_id", "width", "height", "fx", "fy", "cx", "cy",
                                     "world_from_camera"}, "camera calibration"))

    def project(self, xyz: Sequence[float]) -> tuple[float, float]:
        """Project a world point, for geometry and synthetic calibration tests."""
        point = np.asarray(_vector(xyz, 3, "xyz"))
        transform = np.asarray(self.world_from_camera)
        optical = transform[:3, :3].T @ (point - transform[:3, 3])
        if optical[2] <= 1e-9:
            raise ValueError("Point is behind camera")
        return (float(self.fx * optical[0] / optical[2] + self.cx),
                float(self.fy * optical[1] / optical[2] + self.cy))

    def ray(self, uv: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        u, v = _vector(uv, 2, "uv")
        transform = np.asarray(self.world_from_camera)
        direction = transform[:3, :3] @ np.array([(u-self.cx)/self.fx, (v-self.cy)/self.fy, 1.])
        return transform[:3, 3].copy(), direction / np.linalg.norm(direction)


@dataclass(frozen=True)
class RGBFrame:
    frame_id: str
    captured_at: float
    calibration: CameraCalibration
    jpeg_base64: str

    def __post_init__(self):
        _id(self.frame_id, "frame_id")
        object.__setattr__(self, "captured_at", _number(self.captured_at, "captured_at", 0))
        if not isinstance(self.calibration, CameraCalibration):
            raise ValueError("Frame requires validated CameraCalibration")
        if not isinstance(self.jpeg_base64, str) or not 4 <= len(self.jpeg_base64) <= 4*((MAX_JPEG_BYTES+2)//3):
            raise ValueError("Invalid RGB JPEG payload size")
        try:
            raw = base64.b64decode(self.jpeg_base64, validate=True)
            if len(raw) > MAX_JPEG_BYTES:
                raise ValueError("RGB JPEG too large")
            with Image.open(BytesIO(raw)) as image:
                if (image.format != "JPEG" or image.mode != "RGB"
                        or image.size != (self.calibration.width, self.calibration.height)):
                    raise ValueError("Expected RGB JPEG matching calibrated image dimensions")
                image.load()
        except (binascii.Error, OSError, UnidentifiedImageError) as exc:
            raise ValueError("Invalid RGB JPEG payload") from exc

    @property
    def camera_id(self) -> str:
        return self.calibration.camera_id

    @property
    def data_url(self) -> str:
        return "data:image/jpeg;base64," + self.jpeg_base64

    @classmethod
    def from_jpeg(cls, frame_id: str, captured_at: float, calibration: CameraCalibration,
                  jpeg: bytes) -> "RGBFrame":
        if not isinstance(jpeg, bytes):
            raise ValueError("JPEG must be bytes")
        return cls(frame_id, captured_at, calibration, base64.b64encode(jpeg).decode("ascii"))

    def to_dict(self) -> dict:
        return {"frame_id": self.frame_id, "captured_at": self.captured_at,
                "calibration": self.calibration.to_dict(), "jpeg_base64": self.jpeg_base64}

    @classmethod
    def from_dict(cls, value: dict) -> "RGBFrame":
        value = _fields(value, {"frame_id", "captured_at", "calibration", "jpeg_base64"}, "RGB frame")
        return cls(value["frame_id"], value["captured_at"],
                   CameraCalibration.from_dict(value["calibration"]), value["jpeg_base64"])


@dataclass(frozen=True)
class PixelDetection:
    object_id: str
    camera_id: str
    frame_id: str
    uv: tuple[float, float] | None
    confidence: float
    pixel_sigma_px: float | None
    visibility: str = "visible"
    landmark: str = "top_face_center"

    def __post_init__(self):
        if not isinstance(self.object_id, str) or self.object_id not in OBJECT_IDS:
            raise ValueError("Unknown object_id")
        _id(self.camera_id, "camera_id")
        _id(self.frame_id, "frame_id")
        if not isinstance(self.visibility, str) or self.visibility not in VISIBILITIES:
            raise ValueError("Invalid visibility")
        if self.landmark != LANDMARKS[self.object_id]:
            raise ValueError("Object landmark must name the same visible feature across cameras")
        object.__setattr__(self, "confidence", _number(self.confidence, "confidence", 0, 1))
        if self.visibility == "visible":
            object.__setattr__(self, "pixel_sigma_px", _number(self.pixel_sigma_px, "pixel_sigma_px", .1, 4096))
            object.__setattr__(self, "uv", _vector(self.uv, 2, "uv"))
        elif self.uv is not None or self.pixel_sigma_px is not None:
            raise ValueError("Non-visible detections must have uv=None and pixel_sigma_px=None")

    def to_dict(self) -> dict:
        return {"object_id": self.object_id, "camera_id": self.camera_id, "frame_id": self.frame_id,
                "uv": list(self.uv) if self.uv is not None else None, "confidence": self.confidence,
                "pixel_sigma_px": self.pixel_sigma_px, "visibility": self.visibility,
                "landmark": self.landmark}

    @classmethod
    def from_dict(cls, value: dict) -> "PixelDetection":
        return cls(**_fields(value, {"object_id", "camera_id", "frame_id", "uv", "confidence",
                                     "pixel_sigma_px", "visibility", "landmark"}, "pixel detection"))


@dataclass(frozen=True)
class PerceptionConfig:
    min_confidence: float = .75
    min_pixel_sigma_px: float = 2.
    max_frame_age_s: float = 3.
    max_frame_skew_s: float = .05
    future_tolerance_s: float = .02
    min_ray_angle_deg: float = 5.
    max_reprojection_error_px: float = 4.
    max_ray_separation_m: float = .01
    max_position_std_m: float = .015

    def __post_init__(self):
        ranges = {"min_confidence": (.5, 1), "min_pixel_sigma_px": (.5, 100),
                  "max_frame_age_s": (.001, 60), "max_frame_skew_s": (0, 1),
                  "future_tolerance_s": (0, .1), "min_ray_angle_deg": (1, 89),
                  "max_reprojection_error_px": (.1, 20), "max_ray_separation_m": (.00001, .05),
                  "max_position_std_m": (.00001, .05)}
        for name, (lo, hi) in ranges.items():
            object.__setattr__(self, name, _number(getattr(self, name), name, lo, hi))


@dataclass(frozen=True)
class ObjectEstimate:
    object_id: str
    status: str
    landmark: str
    xyz: tuple[float, float, float] | None = None
    covariance_m2: tuple[tuple[float, ...], ...] | None = None
    position_std_m: float | None = None
    confidence: float = 0.
    reprojection_error_px: float | None = None
    ray_separation_m: float | None = None
    ray_angle_deg: float | None = None
    reason: str | None = None

    def __post_init__(self):
        if self.object_id not in OBJECT_IDS or self.landmark != LANDMARKS[self.object_id]:
            raise ValueError("Invalid estimated landmark")
        if self.status not in ("estimated", "unknown"):
            raise ValueError("Invalid estimate status")
        object.__setattr__(self, "confidence", _number(self.confidence, "confidence", 0, 1))
        if self.status == "unknown":
            if any(value is not None for value in (self.xyz, self.covariance_m2, self.position_std_m,
                      self.reprojection_error_px, self.ray_separation_m, self.ray_angle_deg)):
                raise ValueError("Unknown landmark cannot carry geometric coordinates or metrics")
            _id(self.reason, "failure reason")
            return
        if self.reason is not None:
            raise ValueError("Estimated landmark must not have a failure reason")
        object.__setattr__(self, "xyz", _vector(self.xyz, 3, "xyz"))
        if not isinstance(self.covariance_m2, (list, tuple)) or len(self.covariance_m2) != 3:
            raise ValueError("Expected 3x3 covariance")
        covariance = tuple(_vector(row, 3, "covariance") for row in self.covariance_m2)
        matrix = np.asarray(covariance)
        if (not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
                or np.min(np.linalg.eigvalsh(matrix)) < -1e-12):
            raise ValueError("Covariance must be symmetric positive semidefinite")
        object.__setattr__(self, "covariance_m2", covariance)
        for name in ("position_std_m", "reprojection_error_px", "ray_separation_m", "ray_angle_deg"):
            object.__setattr__(self, name, _number(getattr(self, name), name, 0))
        actual_std = float(np.sqrt(max(0, np.max(np.linalg.eigvalsh(matrix)))))
        if not math.isclose(actual_std, self.position_std_m, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError("position_std_m must match largest covariance standard deviation")

    @property
    def valid(self) -> bool:
        return self.status == "estimated"

    def to_dict(self) -> dict:
        return {"object_id": self.object_id, "status": self.status, "landmark": self.landmark,
                "xyz": list(self.xyz) if self.xyz is not None else None,
                "covariance_m2": [list(row) for row in self.covariance_m2] if self.covariance_m2 is not None else None,
                "position_std_m": self.position_std_m, "confidence": self.confidence,
                "reprojection_error_px": self.reprojection_error_px, "ray_separation_m": self.ray_separation_m,
                "ray_angle_deg": self.ray_angle_deg, "reason": self.reason}

    @classmethod
    def from_dict(cls, value: dict) -> "ObjectEstimate":
        return cls(**_fields(value, {"object_id", "status", "landmark", "xyz", "covariance_m2",
                                     "position_std_m", "confidence", "reprojection_error_px",
                                     "ray_separation_m", "ray_angle_deg", "reason"}, "object estimate"))


@dataclass(frozen=True)
class EstimatedScene:
    captured_at: float
    estimated_at: float
    frame_ids: Mapping[str, str]
    objects: Mapping[str, ObjectEstimate]

    def __post_init__(self):
        object.__setattr__(self, "captured_at", _number(self.captured_at, "captured_at", 0))
        object.__setattr__(self, "estimated_at", _number(self.estimated_at, "estimated_at", 0))
        if not isinstance(self.frame_ids, Mapping) or len(self.frame_ids) != 2:
            raise ValueError("Estimated scene requires two camera frame IDs")
        for camera, frame in self.frame_ids.items():
            _id(camera, "camera_id"); _id(frame, "frame_id")
        if (not isinstance(self.objects, Mapping) or set(self.objects) != set(OBJECT_IDS)
                or any(not isinstance(value, ObjectEstimate) or value.object_id != key
                       for key, value in self.objects.items())):
            raise ValueError("Estimated scene requires all explicit object statuses")
        object.__setattr__(self, "frame_ids", MappingProxyType(dict(self.frame_ids)))
        object.__setattr__(self, "objects", MappingProxyType(dict(self.objects)))

    def to_dict(self) -> dict:
        return {"schema": "camera-estimated-scene-v1", "source": "calibrated_rgb_triangulation",
                "captured_at": self.captured_at, "estimated_at": self.estimated_at,
                "frame_ids": dict(self.frame_ids),
                "objects": {key: value.to_dict() for key, value in self.objects.items()}}

    @classmethod
    def from_dict(cls, value: dict) -> "EstimatedScene":
        value = _fields(value, {"schema", "source", "captured_at", "estimated_at", "frame_ids", "objects"}, "estimated scene")
        if value["schema"] != "camera-estimated-scene-v1" or value["source"] != "calibrated_rgb_triangulation":
            raise ValueError("Invalid estimated scene provenance")
        if not isinstance(value["objects"], dict):
            raise ValueError("Invalid objects mapping")
        return cls(value["captured_at"], value["estimated_at"], value["frame_ids"],
                   {key: ObjectEstimate.from_dict(item) for key, item in value["objects"].items()})

    def require(self, object_id: str, *, now: float, max_age_s: float = 3.) -> ObjectEstimate:
        """Recheck freshness at use time; cached scenes never stay valid forever."""
        now = _number(now, "now", 0)
        max_age_s = _number(max_age_s, "max_age_s", .001, 60)
        if not 0 <= now-self.captured_at <= max_age_s:
            raise ValueError("Estimated scene is stale or future dated")
        estimate = self.objects.get(object_id)
        if estimate is None or not estimate.valid:
            raise ValueError("Requested landmark is not safely estimated")
        return estimate


def _closest_point(calibrations: Sequence[CameraCalibration], pixels: np.ndarray):
    origin_a, ray_a = calibrations[0].ray(tuple(float(v) for v in pixels[:2]))
    origin_b, ray_b = calibrations[1].ray(tuple(float(v) for v in pixels[2:]))
    matrix = np.column_stack((ray_a, -ray_b))
    if np.linalg.matrix_rank(matrix, tol=1e-8) < 2:
        raise ValueError("parallel_rays")
    distances = np.linalg.lstsq(matrix, origin_b-origin_a, rcond=None)[0]
    if np.any(distances <= 0):
        raise ValueError("behind_camera")
    point_a = origin_a + distances[0]*ray_a
    point_b = origin_b + distances[1]*ray_b
    # Acute line angle rejects both near-parallel and near-antiparallel views.
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(ray_a, ray_b)), 0, 1))))
    return (point_a+point_b)/2, float(np.linalg.norm(point_a-point_b)), angle


def estimate_scene(frames: Sequence[RGBFrame], detections: Sequence[PixelDetection], *,
                   now: float, workspace: Sequence[Sequence[float]],
                   config: PerceptionConfig | None = None) -> EstimatedScene:
    """Triangulate two matching RGB observations, or return explicit unknowns.

    workspace is an *observation volume*, which must include the goal plane.
    It is distinct from the robot's smaller motion workspace. No single-view
    table-height fallback exists, even for a marker believed to be on a table.
    Missing a view, uncertain correspondence, stale images, and inconsistent
    rays are fail-closed. Malformed contracts/configuration raise ValueError.
    """
    now = _number(now, "now", 0)
    config = config if config is not None else PerceptionConfig()
    if not isinstance(config, PerceptionConfig):
        raise ValueError("Invalid perception config")
    if (not isinstance(frames, (list, tuple)) or len(frames) != 2
            or not all(isinstance(frame, RGBFrame) for frame in frames)
            or len({frame.camera_id for frame in frames}) != 2):
        raise ValueError("Exactly two distinct calibrated RGB cameras are required")
    if (not isinstance(detections, (list, tuple)) or len(detections) > 32
            or not all(isinstance(item, PixelDetection) for item in detections)):
        raise ValueError("Invalid pixel detections")
    if not isinstance(workspace, (list, tuple)) or len(workspace) != 3:
        raise ValueError("Observation workspace must have three bounds")
    bounds = tuple(_vector(pair, 2, "workspace bounds") for pair in workspace)
    if any(lo >= hi for lo, hi in bounds):
        raise ValueError("Observation workspace bounds must be increasing")
    frame_map = {frame.camera_id: frame for frame in frames}
    captured_at = min(frame.captured_at for frame in frames)
    global_reason = None
    if any(frame.captured_at > now + config.future_tolerance_s for frame in frames):
        global_reason = "future_frame"
    elif any(now-frame.captured_at > config.max_frame_age_s for frame in frames):
        global_reason = "stale_frame"
    elif max(frame.captured_at for frame in frames)-captured_at > config.max_frame_skew_s:
        global_reason = "unsynchronized_frames"
    objects = {}
    for object_id in OBJECT_IDS:
        def reject(reason: str):
            return ObjectEstimate(object_id, "unknown", LANDMARKS[object_id], reason=reason)
        if global_reason:
            objects[object_id] = reject(global_reason)
            continue
        selected = [item for item in detections if item.object_id == object_id]
        reason = None
        if any(item.camera_id not in frame_map or item.frame_id != frame_map[item.camera_id].frame_id for item in selected):
            reason = "frame_mismatch"
        elif any(sum(item.camera_id == camera for item in selected) > 1 for camera in frame_map):
            reason = "ambiguous_correspondence"
        elif len(selected) != 2:
            reason = "missing_view"
        elif any(item.visibility != "visible" for item in selected):
            reason = "landmark_not_visible"
        elif any(item.confidence < config.min_confidence for item in selected):
            reason = "low_confidence"
        elif any(not (0 <= item.uv[0] <= frame_map[item.camera_id].calibration.width-1
                      and 0 <= item.uv[1] <= frame_map[item.camera_id].calibration.height-1) for item in selected):
            reason = "pixel_out_of_bounds"
        if reason:
            objects[object_id] = reject(reason)
            continue
        calibrations = [frame_map[item.camera_id].calibration for item in selected]
        pixels = np.array([v for item in selected for v in item.uv], dtype=float)
        try:
            xyz, separation, angle = _closest_point(calibrations, pixels)
            if angle < config.min_ray_angle_deg:
                raise ValueError("poor_stereo_angle")
            if separation > config.max_ray_separation_m:
                raise ValueError("inconsistent_rays")
            reprojection = max(math.dist(calibration.project(tuple(float(v) for v in xyz)), item.uv)
                               for calibration, item in zip(calibrations, selected))
            if reprojection > config.max_reprojection_error_px:
                raise ValueError("reprojection_error")
            if not all(lo <= value <= hi for value, (lo, hi) in zip(xyz, bounds)):
                raise ValueError("outside_observation_workspace")
            # Numerical Jacobian of the SAME estimator used above, in m/pixel.
            # Conservative pixel floor prevents a model from claiming zero error.
            jacobian = np.empty((3, 4))
            epsilon = .01
            for column in range(4):
                delta = np.zeros(4); delta[column] = epsilon
                positive = _closest_point(calibrations, pixels+delta)[0]
                negative = _closest_point(calibrations, pixels-delta)[0]
                jacobian[:, column] = (positive-negative)/(2*epsilon)
            sigmas = np.array([max(config.min_pixel_sigma_px, item.pixel_sigma_px)
                               for item in selected for _ in range(2)])
            covariance = (jacobian*sigmas) @ (jacobian*sigmas).T
            # Residual mismatch is additional uncertainty, never extra confidence.
            covariance += np.eye(3)*(separation/2)**2
            std = float(np.sqrt(max(0, np.max(np.linalg.eigvalsh(covariance)))))
            if not math.isfinite(std) or std > config.max_position_std_m:
                raise ValueError("excessive_3d_uncertainty")
            objects[object_id] = ObjectEstimate(
                object_id, "estimated", LANDMARKS[object_id], tuple(float(v) for v in xyz),
                tuple(tuple(float(v) for v in row) for row in covariance), std,
                min(item.confidence for item in selected), reprojection, separation, angle)
        except (ValueError, np.linalg.LinAlgError) as exc:
            known = {"parallel_rays", "behind_camera", "poor_stereo_angle", "inconsistent_rays",
                     "reprojection_error", "outside_observation_workspace", "excessive_3d_uncertainty"}
            reason = str(exc) if str(exc) in known else "invalid_geometry"
            objects[object_id] = reject(reason)
    return EstimatedScene(captured_at, now, {frame.camera_id: frame.frame_id for frame in frames}, objects)
