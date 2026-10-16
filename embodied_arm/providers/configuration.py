"""Server-only, versioned provider configuration; no writes or network access.

Legacy flat Jev configuration is mapped in memory. Never serialize a loaded
configuration: use public_status() for diagnostics, which excludes credentials.
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from urllib.parse import urlsplit
from embodied_arm.paths import CONFIG_EXCLUSION_ROOT

BASE_URL='https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
PROVIDER='aliyun_bailian'
MODELS={'brain':'qwen3.8-max','cerebellum':'decision-model-preview'}
ROUTES={'brain':'/chat/completions','cerebellum':'/systemone'}

@dataclass(frozen=True)
class ProviderSettings:
    component: str
    base_url: str
    model: str
    api_key: str = field(repr=False,compare=False)
    timeout: float = 30.
    min_confidence: float = .65
    def public_status(self):
        return {'component':self.component,'provider':PROVIDER,'base_url':self.base_url,
                'endpoint':self.endpoint,'model':self.model,'configured':bool(self.api_key),
                'timeout':self.timeout,'min_confidence':self.min_confidence}
    @property
    def endpoint(self):return self.base_url+ROUTES[self.component]

def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate_config_field')
        result[key]=value
    return result

def normalize_base_url(value,component):
    if component not in ROUTES or not isinstance(value,str):raise ValueError('unsupported_provider_destination')
    # Permit either the exact base or this component's complete route, never
    # guess a protocol from a different suffix or append a route twice.
    raw=value.rstrip('/')
    if raw==BASE_URL+ROUTES[component]:raw=BASE_URL
    parts=urlsplit(raw)
    if (raw!=BASE_URL or parts.scheme!='https' or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):raise ValueError('unsupported_provider_destination')
    return raw

def _number(value,upper):
    return type(value) in (int,float) and math.isfinite(value) and 0<value<=upper

def _settings(group,component):
    allowed={'provider','base_url','model','api_key','timeout'}
    if component=='cerebellum':allowed.add('min_confidence')
    if not isinstance(group,dict) or set(group)-allowed:raise ValueError('invalid_provider_fields')
    if group.get('provider')!=PROVIDER or group.get('model')!=MODELS[component]:raise ValueError('unsupported_provider_model')
    base=normalize_base_url(group.get('base_url'),component)
    key=group.get('api_key')
    if not isinstance(key,str) or not key or any(not 33<=ord(c)<=126 for c in key):raise ValueError('missing_or_invalid_api_key')
    timeout=group.get('timeout',30 if component=='brain' else 15)
    if not _number(timeout,60 if component=='brain' else 30):raise ValueError('invalid_provider_timeout')
    confidence=group.get('min_confidence',.65)
    if type(confidence) not in (int,float) or not math.isfinite(confidence) or not .65<=confidence<=1:raise ValueError('invalid_confidence_threshold')
    return ProviderSettings(component,base,MODELS[component],key,float(timeout),float(confidence))

def parse_configuration(config,*,component=None):
    if component is not None and component not in MODELS:raise ValueError('unsupported_component')
    if not isinstance(config,dict):raise ValueError('invalid_provider_configuration')
    if 'schema_version' in config:
        if type(config['schema_version']) is not int or config['schema_version']!=2 or set(config)!={'schema_version','brain','cerebellum'}:
            raise ValueError('invalid_configuration_version')
        return {name:_settings(config[name],name) for name in ([component] if component else MODELS)}
    allowed={'provider','endpoint','model','api_key','min_confidence','timeout'}
    if (set(config)-allowed or config.get('provider')!=PROVIDER
            or config.get('endpoint')!=BASE_URL+'/systemone' or config.get('model')!=MODELS['cerebellum']):
        raise ValueError('invalid_legacy_configuration')
    threshold=config.get('min_confidence',.65)
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('invalid_confidence_threshold')
    # Legacy brain always used its own30s request timeout, not the Jev field.
    return {'brain':_settings({'provider':PROVIDER,'base_url':BASE_URL,'model':MODELS['brain'],'api_key':config.get('api_key'),'timeout':30},'brain'),
            'cerebellum':_settings({'provider':PROVIDER,'base_url':BASE_URL,'model':MODELS['cerebellum'],'api_key':config.get('api_key'),
                'timeout':config.get('timeout',15),'min_confidence':max(.65,threshold)},'cerebellum')}

def load_configuration(path,*,component=None):
    """Trusted CLI/server path only; errors never include paths, values or keys."""
    try:
        if not isinstance(path,(str,Path)) or not str(path).strip():raise ValueError('invalid_path')
        file=Path(path).expanduser().resolve()
        if file==CONFIG_EXCLUSION_ROOT or CONFIG_EXCLUSION_ROOT in file.parents:raise ValueError('config_must_be_external')
        with file.open('rb') as stream:raw=stream.read(16385)
        if len(raw)>16384:raise ValueError('config_too_large')
        value=json.loads(raw,object_pairs_hook=_pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
        return parse_configuration(value,component=component)
    except (OSError,ValueError,TypeError,RecursionError):
        raise ValueError('Cannot load external provider configuration') from None

def request_timeout(settings,cap=None):
    if cap is None:return settings.timeout
    if not _number(cap,60 if settings.component=='brain' else 30):raise ValueError('invalid_provider_timeout')
    return min(settings.timeout,float(cap))
