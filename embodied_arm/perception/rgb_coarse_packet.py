"""Offline, frame-bound downsampling for coarse stereo RGB model localization.

No capture, upload, network, calibration, depth, simulator state or inference is
performed here. Keep the original native packet local for RGB refinement.

Pillow resize uses pixel-centre coordinates: a coarse centre (u,v) maps back to
((u+.5)*native_width/coarse_width-.5, (v+.5)*native_height/coarse_height-.5).
There is no crop, pad, align-corners transform, rounding or clamping. Different
integer-rounded X/Y scales are tracked separately. An isotropic input sigma is
multiplied by the larger scale, conservatively enclosing affine-transformed
covariance. This does not estimate extra JPEG/resampling or semantic error.
Mapped detections are coarse seeds, not a claim of native-resolution accuracy.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
import json
import math
import time
from typing import Sequence

from PIL import Image

from embodied_arm.perception.camera_perception import PixelDetection
from embodied_arm.providers.vision_adapter import VisionResult, validate_detections

MAX_NATIVE_DIMENSION = 4096
MAX_NATIVE_JPEG_BYTES = 8 * 1024 * 1024
MAX_MODEL_WIDTH = 960
MAX_MODEL_HEIGHT = 720
MAX_MODEL_PACKET_BYTES = 1024 * 1024 - 1024
JPEG_PREFIX = 'data:image/jpeg;base64,'


def _json(value):
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ValueError('Invalid RGB packet data') from None


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _finite(value):
    return type(value) in (int, float) and -1e12 < value < 1e12 and math.isfinite(value)


def _packet_shape(packet, *, now, max_age_s):
    if not _finite(now) or not _finite(max_age_s) or not 0 < max_age_s <= 60:
        raise ValueError('Invalid freshness bounds')
    if not isinstance(packet, dict) or set(packet) != {'frame_id', 'captured_at', 'images'}:
        raise ValueError('Only image-only stereo packets are allowed')
    fid, stamp, images = packet['frame_id'], packet['captured_at'], packet['images']
    if (not isinstance(fid, str) or not 1 <= len(fid) <= 80
            or not all(c.isalnum() or c in '-_.' for c in fid)):
        raise ValueError('Invalid frame identity')
    if not _finite(stamp) or not 0 <= now - stamp <= max_age_s:
        raise ValueError('Stale or future RGB packet')
    if not isinstance(images, list) or len(images) != 2:
        raise ValueError('Exactly two RGB views are required')
    safe, seen = [], set()
    for item in images:
        if not isinstance(item, dict) or set(item) != {'camera_id', 'width', 'height', 'data_url'}:
            raise ValueError('Invalid image-only view fields')
        cid, width, height, url = (item[k] for k in ('camera_id', 'width', 'height', 'data_url'))
        if not isinstance(cid, str) or cid not in ('left', 'right') or cid in seen:
            raise ValueError('Invalid stereo view identity')
        if any(type(v) is not int or not 16 <= v <= MAX_NATIVE_DIMENSION for v in (width, height)):
            raise ValueError('Unsupported native image dimensions')
        if (not isinstance(url, str) or not url.startswith(JPEG_PREFIX)
                or len(url) > len(JPEG_PREFIX) + 4 * ((MAX_NATIVE_JPEG_BYTES + 2) // 3)):
            raise ValueError('Only bounded inline RGB JPEGs are allowed')
        seen.add(cid)
        safe.append(dict(item))
    return {'frame_id': fid, 'captured_at': stamp, 'images': sorted(safe, key=lambda i: i['camera_id'])}


@dataclass(frozen=True)
class PixelScale:
    camera_id: str
    native_width: int
    native_height: int
    model_width: int
    model_height: int

    @property
    def scale_x(self): return self.native_width / self.model_width

    @property
    def scale_y(self): return self.native_height / self.model_height

    def to_native(self, uv):
        if (not isinstance(uv, (tuple, list)) or len(uv) != 2
                or not all(_finite(v) for v in uv)
                or not 0 <= uv[0] <= self.model_width - 1
                or not 0 <= uv[1] <= self.model_height - 1):
            raise ValueError('Coarse point is outside pixel-centre bounds')
        if self.native_width == self.model_width and self.native_height == self.model_height:
            return tuple(uv)
        return ((uv[0] + .5) * self.scale_x - .5,
                (uv[1] + .5) * self.scale_y - .5)


@dataclass(frozen=True)
class CoarseStereoPacket:
    """Immutable binding and transforms. Only .packet goes to the model."""
    frame_id: str
    captured_at: float
    transforms: tuple[PixelScale, ...]
    native_sha256: str
    model_sha256: str
    _packet_json: bytes

    @property
    def packet(self):
        """A fresh detached image-only dict; never includes native images."""
        return json.loads(self._packet_json)


def prepare_coarse_packet(native_packet, *, max_width=MAX_MODEL_WIDTH,
                          max_height=MAX_MODEL_HEIGHT, jpeg_quality=85,
                          now=None, max_age_s=60):
    """Downsample the same frozen pair once; do not recapture or send anything.

    A no-scale image preserves its original JPEG bytes. Resized images use
    LANCZOS and the explicit JPEG quality. Oversized output fails closed rather
    than silently changing quality, cropping, or increasing the model boundary.
    """
    if (type(max_width) is not int or not 16 <= max_width <= MAX_MODEL_WIDTH
            or type(max_height) is not int or not 16 <= max_height <= MAX_MODEL_HEIGHT
            or type(jpeg_quality) is not int or not 50 <= jpeg_quality <= 95):
        raise ValueError('Invalid bounded coarse-image settings')
    now = time.monotonic() if now is None else now
    native = _packet_shape(native_packet, now=now, max_age_s=max_age_s)
    images, transforms = [], []
    for item in native['images']:
        width, height = item['width'], item['height']
        scale = min(1., max_width / width, max_height / height)
        size = (max(1, math.floor(width * scale)), max(1, math.floor(height * scale)))
        if min(size) < 16:
            raise ValueError('Aspect ratio cannot fit the bounded model image')
        try:
            raw = base64.b64decode(item['data_url'][len(JPEG_PREFIX):], validate=True)
            if not 0 < len(raw) <= MAX_NATIVE_JPEG_BYTES or not raw.startswith(b'\xff\xd8') or not raw.endswith(b'\xff\xd9'):
                raise ValueError('Incomplete or oversized native JPEG')
            with Image.open(BytesIO(raw)) as image:
                if image.format != 'JPEG' or image.mode != 'RGB' or image.size != (width, height):
                    raise ValueError('Native JPEG dimensions or RGB mode mismatch')
                image.load()
                url = item['data_url']
                if size != image.size:
                    resized = image.resize(size, Image.Resampling.LANCZOS)
                    output = BytesIO()
                    resized.save(output, 'JPEG', quality=jpeg_quality)
                    url = JPEG_PREFIX + base64.b64encode(output.getvalue()).decode('ascii')
        except (OSError, ValueError, Image.DecompressionBombError):
            raise ValueError('Invalid bounded native RGB JPEG') from None
        images.append({'camera_id': item['camera_id'], 'width': size[0], 'height': size[1], 'data_url': url})
        transforms.append(PixelScale(item['camera_id'], width, height, *size))
    model = {'frame_id': native['frame_id'], 'captured_at': native['captured_at'], 'images': images}
    encoded = _json(model)
    if len(encoded) > MAX_MODEL_PACKET_BYTES:
        raise ValueError('Coarse packet exceeds the existing model IPC byte bound')
    return CoarseStereoPacket(native['frame_id'], native['captured_at'], tuple(transforms),
                              _digest(native), hashlib.sha256(encoded).hexdigest(), encoded)


def remap_detections(detections: Sequence[PixelDetection], prepared: CoarseStereoPacket, *,
                     native_packet, submitted_packet=None, now=None, max_age_s=60):
    """Return native pixel seeds tied to the exact original frozen JPEG pair.

    Non-visible observations retain null coordinates and null sigma; no hidden
    feature is inferred. No-scale remapping preserves coordinates and sigma.
    Pass submitted_packet when the caller retains the actual dict sent to Qwen;
    any mutation of that dict is rejected by its original image digest.
    """
    if not isinstance(prepared, CoarseStereoPacket):
        raise ValueError('A prepared stereo binding is required')
    now = time.monotonic() if now is None else now
    native = _packet_shape(native_packet, now=now, max_age_s=max_age_s)
    if _digest(native) != prepared.native_sha256:
        raise ValueError('Native frame changed after coarse preparation')
    model = prepared.packet
    if (hashlib.sha256(prepared._packet_json).hexdigest() != prepared.model_sha256
            or model['frame_id'] != prepared.frame_id or model['captured_at'] != prepared.captured_at
            or (submitted_packet is not None and _digest(submitted_packet) != prepared.model_sha256)):
        raise ValueError('Submitted coarse packet differs from its frozen binding')
    if not isinstance(detections, (list, tuple)) or not all(isinstance(d, PixelDetection) for d in detections):
        raise ValueError('Canonical PixelDetection observations are required')
    canonical = validate_detections({'frame_id': prepared.frame_id,
        'detections': [d.to_dict() for d in detections]}, model,
        min_confidence=.75, max_uncertainty_px=25)
    transforms = {t.camera_id: t for t in prepared.transforms}
    sizes = {i['camera_id']: (i['width'], i['height']) for i in native['images']}
    model_sizes = {i['camera_id']: (i['width'], i['height']) for i in model['images']}
    if set(transforms) != {'left', 'right'} or len(prepared.transforms) != 2:
        raise ValueError('Invalid stereo transform binding')
    for cid, transform in transforms.items():
        if ((transform.native_width, transform.native_height) != sizes[cid]
                or (transform.model_width, transform.model_height) != model_sizes[cid]):
            raise ValueError('Transform dimensions do not match the bound images')
    out = []
    for detection in canonical:
        if detection.visibility != 'visible':
            out.append(detection)
            continue
        transform = transforms[detection.camera_id]
        uv = transform.to_native(detection.uv)
        sigma = detection.pixel_sigma_px * max(transform.scale_x, transform.scale_y)
        out.append(replace(detection, uv=uv, pixel_sigma_px=sigma))
    return tuple(out)


def remap_result(result: VisionResult, prepared: CoarseStereoPacket, *, native_packet,
                 submitted_packet=None, now=None, max_age_s=60):
    """Convenience wrapper for an accepted result with matching frame AND time."""
    if (not isinstance(prepared, CoarseStereoPacket) or not isinstance(result, VisionResult)
            or result.accepted is not True or result.reason != 'accepted'
            or result.frame_id != prepared.frame_id or result.captured_at != prepared.captured_at):
        raise ValueError('Accepted perception must match the original frame and capture time')
    detections = remap_detections(result.detections, prepared, native_packet=native_packet,
        submitted_packet=submitted_packet, now=now, max_age_s=max_age_s)
    if any(d.visibility != 'visible' or d.confidence < .75 for d in detections):
        raise ValueError('Accepted perception contains an uncertain or invisible feature')
    return replace(result, detections=detections)
