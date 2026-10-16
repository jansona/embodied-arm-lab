"""Bounded request metadata only; never prompts, images, headers or raw errors."""
import math
import re
import time

SCHEMA = 'provider-request-telemetry-v1'
COMPONENT_MODELS = {'brain': 'qwen3.8-max', 'vision': 'qwen3.8-max', 'cerebellum': 'decision-model-preview'}
USAGE_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'input_tokens', 'output_tokens')
OUTCOMES = frozenset(('pending', 'accepted', 'timeout', 'network_error', 'http_error', 'api_error',
    'invalid_response', 'unexpected_model', 'incomplete_response', 'response_too_large', 'stale_frame',
    'occluded_or_missing_feature', 'low_confidence', 'criterion_mismatch'))
_ID = re.compile(r'(?:[A-Fa-f0-9]{8}(?:-[A-Fa-f0-9]{4}){3}-[A-Fa-f0-9]{12}|(?:req[_-]|chatcmpl-)[A-Za-z0-9_-]{1,100})\Z')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def sanitize_request_telemetry(value):
    """Rebuild the fixed public grammar; absence never means zero usage."""
    if not isinstance(value, dict) or not isinstance(value.get('component'), str) or value['component'] not in COMPONENT_MODELS:
        return None
    component = value['component']
    if value.get('model') != COMPONENT_MODELS[component]:
        return None
    start, elapsed = value.get('started_at_unix'), value.get('elapsed_seconds')
    if not _number(start) or start > 1e11 or not _number(elapsed) or elapsed > 86400:
        return None
    outcome = value.get('outcome')
    outcome = outcome if isinstance(outcome, str) and outcome in OUTCOMES else 'invalid_response'
    status = value.get('http_status')
    request_id = value.get('request_id')
    usage = value.get('usage')
    return {'schema': SCHEMA, 'component': component, 'model': COMPONENT_MODELS[component],
        'started_at_unix': round(start, 6), 'elapsed_seconds': round(elapsed, 6), 'outcome': outcome,
        'http_status': status if type(status) is int and 100 <= status <= 599 else None,
        'request_id': request_id if isinstance(request_id, str) and _ID.fullmatch(request_id) else None,
        'usage': {k: usage[k] for k in USAGE_FIELDS if isinstance(usage, dict)
                  and type(usage.get(k)) is int and 0 <= usage[k] <= 1_000_000_000},
        'processing_uncertain': outcome in ('pending', 'timeout', 'network_error'), 'automatic_retry': False}


class RequestTelemetry:
    """One attempted request, recording metadata on its adapter only."""
    def __init__(self, owner, component):
        self.owner = owner
        self.started = time.monotonic()
        self.data = {'component': component, 'model': COMPONENT_MODELS[component],
            'started_at_unix': time.time(), 'elapsed_seconds': 0, 'outcome': 'pending',
            'http_status': None, 'request_id': None, 'usage': {}}
        self.publish()

    def publish(self):
        self.data['elapsed_seconds'] = min(86400, max(0, time.monotonic() - self.started))
        self.owner.last_request = sanitize_request_telemetry(self.data)

    def response(self, response):
        self.data['http_status'] = response.status_code
        self.data['request_id'] = response.headers.get('x-request-id') or response.headers.get('request-id')
        self.publish()
        self.data['request_id'] = self.owner.last_request['request_id']

    def payload(self, payload):
        if isinstance(payload, dict):
            self.data['usage'] = payload.get('usage', {})
            self.data['request_id'] = self.data['request_id'] or payload.get('request_id') or payload.get('id')
        self.publish()
        # Do not retain an untrusted usage object or identifier even privately.
        self.data['usage'] = self.owner.last_request['usage']
        self.data['request_id'] = self.owner.last_request['request_id']

    def finish(self, reason):
        if reason.startswith(('http_error_', 'provider_')):
            reason = 'http_error'
        self.data['outcome'] = reason
        self.publish()


def worker_telemetry(value, component):
    """Validate helper metadata independently of its public model result."""
    clean = sanitize_request_telemetry(value)
    return clean if clean is not None and clean['component'] == component else None


def request_audit_fields(value, component):
    clean = worker_telemetry(value, component)
    return {'request_telemetry': clean, 'request_metadata_available': clean is not None,
            'processing_uncertain': clean['processing_uncertain'] if clean else True,
            'automatic_retry': False}
