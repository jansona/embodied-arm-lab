"""SO-ARM101 encoder/FK-only Jev microcontroller; no simulator or actuation.

The caller supplies an RGB-derived local target and a deterministic axis index.
Axis scheduling is disclosed and selects only an axis, never a direction. Jev
selects the direction or hold. Rejected decisions have action=None; there is no
rule-selected fallback. No default tactile/contact sensor is assumed.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
import math
import threading

import httpx
from embodied_arm.providers.request_telemetry import RequestTelemetry
from embodied_arm.providers.adapters import ALIYUN_ENDPOINT, ALIYUN_MODEL, MAX_RESPONSE_BYTES

SCHEMA='so101-jev-observation-v1'
DECISION_SCHEMA='so101-jev-decision-v1'
MIN_CONFIDENCE=.65
MAX_INPUT_BYTES=16384
ROBOT_FIELDS={'name','arm_dof','ee','joints','joint_limits','gripper_openness','gripper_angle',
              'paused','busy','epoch','actions','workspace','cube_edge_m'}
CARTESIAN_CHOICES=('positive','negative','hold')
GRIPPER_CHOICES=('open','close','hold')
ROLE_INSTRUCTIONS = (
    "You are Jev, the SO-ARM101 cerebellum for this simulation. Qwen is the lower-frequency "
    "high-level planner; the runtime grounds its symbolic goal into this local target. "
    "You make one frequent fresh-state microdecision per request, never a full task plan or queued trajectory. "
    "Answer only the action choice question using its supplied candidates and normal choice confidence/probabilities; "
    "do not add chat, explanations, coordinates, or reasoning. Report uncertainty honestly; confidence below .65 "
    "is rejected without a substitute action. Hold means the stated local tolerance criterion, not task success. "
    "The runtime re-observes after motion and owns collision checks, stale-state rejection, cancellation and budgets. "
    "At a cycle cap it pauses for fresh Qwen planning; you cannot reset global limits or assert a grasp succeeded. "
)



def _number(value):
    return type(value) in (int,float) and -1e9<value<1e9 and math.isfinite(value)


def _vector(value,length):
    return isinstance(value,list) and len(value)==length and all(_number(v) for v in value)


def _json_copy(value):
    try:
        raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
        if len(raw)>MAX_INPUT_BYTES:raise ValueError('Input too large')
        return json.loads(raw)
    except (ValueError,TypeError,OverflowError,RecursionError):
        raise ValueError('Invalid SO101 JSON input') from None


def validate_observation(value):
    """Strict wire boundary: only a local target and robot_only whitelist."""
    value=_json_copy(value)
    if not isinstance(value,dict) or set(value)!={'schema','source','target','robot','axis_index'} or value['schema']!=SCHEMA or value['source']!='rgb_only':
        raise ValueError('RGB-derived SO101 observation required')
    robot=value['robot']
    if not isinstance(robot,dict) or set(robot)!=ROBOT_FIELDS or robot['name']!='SO-ARM101' or type(robot['arm_dof']) is not int or robot['arm_dof']!=5:
        raise ValueError('Only robot_only SO101 proprioception is permitted')
    if not _vector(robot['ee'],3) or not _vector(robot['joints'],5):raise ValueError('Invalid encoders or FK')
    for key,count in (('joint_limits',5),('workspace',3)):
        ranges=robot[key]
        if not isinstance(ranges,list) or len(ranges)!=count or not all(_vector(pair,2) and pair[0]<pair[1] for pair in ranges):raise ValueError('Invalid robot bounds')
    if not all(lo-1e-5<=q<=hi+1e-5 for q,(lo,hi) in zip(robot['joints'],robot['joint_limits'])):raise ValueError('Encoder outside joint limits')
    if not _number(robot['gripper_openness']) or not 0<=robot['gripper_openness']<=1 or not _number(robot['gripper_angle']) or robot['cube_edge_m']!=.02:raise ValueError('Invalid gripper proprioception')
    if any(type(robot[key]) is not bool for key in ('paused','busy')) or any(type(robot[key]) is not int or not 0<=robot[key]<1e9 for key in ('epoch','actions')):raise ValueError('Invalid runtime metadata')
    target=value['target']
    if not isinstance(target,dict) or target.get('source')!='rgb_only':raise ValueError('Explicit RGB-derived target required')
    if target.get('kind')=='cartesian':
        if set(target)!={'source','kind','target_xyz','tolerance_m','step_m'} or not _vector(target['target_xyz'],3):raise ValueError('Invalid Cartesian target')
        if type(value['axis_index']) is not int or value['axis_index'] not in (0,1,2):raise ValueError('Explicit deterministic axis index required')
        if not _number(target['tolerance_m']) or not .001<=target['tolerance_m']<=.02 or not _number(target['step_m']) or not 0<target['step_m']<=.02:raise ValueError('Invalid bounded motion parameters')
        if not all(lo<=q<=hi for q,(lo,hi) in zip(target['target_xyz'],robot['workspace'])):raise ValueError('Target outside workspace')
    elif target.get('kind')=='gripper':
        if set(target)!={'source','kind','target_openness','tolerance'} or type(target['target_openness']) not in (int,float) or target['target_openness'] not in (0,1):raise ValueError('Gripper supports full open or full close only')
        if not _number(target['tolerance']) or not .001<=target['tolerance']<=.1 or value['axis_index'] is not None:raise ValueError('Invalid gripper tolerance or axis')
    else:raise ValueError('Unknown local target kind')
    return value


def build_observation(local_target,robot_only_proprioception,axis_index=None):
    """Caller must derive target from RGB intent and supply robot_only output."""
    return validate_observation({'schema':SCHEMA,'source':'rgb_only','target':local_target,
                                 'robot':robot_only_proprioception,'axis_index':axis_index})


def request_preview(observation):
    """Unauthenticated exact body preview. No scheduled or chosen-action field."""
    obs=validate_observation(observation);robot=obs['robot'];target=obs['target']
    proprioception={key:robot[key] for key in ('name','arm_dof','ee','joints','joint_limits','gripper_openness','gripper_angle')}
    if target['kind']=='cartesian':
        axis=obs['axis_index'];signed=round(target['target_xyz'][axis]-robot['ee'][axis],9)
        state={'schema':'so101-local-axis-v1','source':'rgb_only_local_target_and_encoder_fk',
               'proprioception':proprioception,'target_xyz_m':target['target_xyz'],
               'axis_index':axis,'axis_name':'xyz'[axis],
               'axis_schedule':'Caller-supplied fixed XYZ among clearly unfinished axes; uncertain axes are deferred. Axis selection does not select direction.',
               'signed_error_m':signed,'tolerance_m':target['tolerance_m'],'step_m':target['step_m'],
               'workspace_m':robot['workspace']}
        instructions='Choose the one criterion satisfied by signed_error_m = target coordinate minus measured FK coordinate on the scheduled axis. The positive, negative, and inclusive hold bands are mutually exclusive. Keep other coordinates and gripper unchanged. This is SO-ARM101, with five arm joints. The caller enforces workspace and IK safety independently.'
        criteria={'positive':'signed_error_m > tolerance_m. Increase only the scheduled coordinate by up to step_m, capped at the remaining distance.',
                  'negative':'signed_error_m < -tolerance_m. Decrease only the scheduled coordinate by up to step_m, capped at the remaining distance.',
                  'hold':'-tolerance_m <= signed_error_m <= tolerance_m. Do not move; the scheduled axis is within tolerance.'}
    else:
        signed=round(target['target_openness']-robot['gripper_openness'],9)
        state={'schema':'so101-local-gripper-v1','source':'rgb_only_local_target_and_encoder',
               'proprioception':proprioception,'target_openness':target['target_openness'],
               'signed_openness_error':signed,'tolerance':target['tolerance'],
               'capability':'Full open (1), full close (0), or hold; no partial-servo step is claimed.',
               'grasp_verification':'Gripper position does not prove a grasp; verification requires fresh temporal RGB observations.'}
        instructions='Choose the one criterion satisfied by signed_openness_error = target normalized openness minus measured normalized openness. Openness 0 is fully closed and 1 is fully open. Use mutually exclusive signed-error bands. Keep all arm joints unchanged. No tactile or object-contact information is available.'
        criteria={'open':'signed_openness_error > tolerance. Command full open (normalized openness 1).',
                  'close':'signed_openness_error < -tolerance. Command full close (normalized openness 0).',
                  'hold':'-tolerance <= signed_openness_error <= tolerance. Keep the gripper unchanged; this does not verify a grasp.'}
    return {'endpoint':ALIYUN_ENDPOINT,'method':'POST','body':{'model':ALIYUN_MODEL,'state':state,
            'questions':{'action':{'type':'choice','instructions':ROLE_INSTRUCTIONS+instructions,'criteria':criteria}}}}


@dataclass(frozen=True)
class SO101Decision:
    action:str|None=None
    accepted:bool=False
    proposed_choice:str|None=None
    confidence:float=0.
    probabilities:dict[str,float]=field(default_factory=dict)
    reason:str='unavailable'
    axis_index:int|None=None
    schema:str=DECISION_SCHEMA
    def to_dict(self):return asdict(self)


def parse_response(payload,observation,min_confidence=MIN_CONFIDENCE):
    """Strict categorical parsing. Inconsistent choices are rejected, not fixed."""
    obs=validate_observation(observation)
    if not _number(min_confidence) or not MIN_CONFIDENCE<=min_confidence<=1:raise ValueError('Confidence cannot be reduced below .65')
    axis=obs['axis_index'];target=obs['target'];robot=obs['robot']
    candidates=CARTESIAN_CHOICES if target['kind']=='cartesian' else GRIPPER_CHOICES
    rejected=SO101Decision(reason='invalid_response',axis_index=axis)
    if not isinstance(payload,dict) or payload.get('model')!=ALIYUN_MODEL or not isinstance(payload.get('answers'),dict) or set(payload['answers'])!={'action'}:return rejected
    answer=payload['answers']['action']
    if not isinstance(answer,dict) or set(answer)!={'type','choice','confidence','probabilities'} or answer['type']!='choice':return rejected
    choice,confidence,probabilities=answer['choice'],answer['confidence'],answer['probabilities']
    if choice not in candidates or not _number(confidence) or not 0<=confidence<=1 or not isinstance(probabilities,dict) or set(probabilities)!=set(candidates) or not all(_number(v) and 0<=v<=1 for v in probabilities.values()):return rejected
    total=sum(probabilities.values());rounded=all(abs(v*100-round(v*100))<1e-8 for v in probabilities.values())
    if abs(total-1)>(.015000001 if rounded else .001) or probabilities[choice]+1e-9<max(probabilities.values()):return rejected
    divisor=1.0 if math.isclose(total,1.0,rel_tol=0,abs_tol=1e-12) else total
    values={k:float(v)/divisor for k,v in probabilities.items()}
    base=dict(proposed_choice=choice,confidence=float(confidence),probabilities=values,axis_index=axis)
    if confidence<min_confidence:return SO101Decision(reason='low_confidence',**base)
    if target['kind']=='cartesian':
        error=round(target['target_xyz'][axis]-robot['ee'][axis],9);tolerance=target['tolerance_m']
        consistent={'positive':error>tolerance,'negative':error < -tolerance,'hold':abs(error)<=tolerance}[choice]
        action={'positive':'xyz'[axis]+'_plus','negative':'xyz'[axis]+'_minus','hold':'hold'}[choice]
    else:
        error=round(target['target_openness']-robot['gripper_openness'],9);tolerance=target['tolerance']
        consistent={'open':error>tolerance,'close':error < -tolerance,'hold':abs(error)<=tolerance}[choice]
        action=choice
    if not consistent:return SO101Decision(reason='criterion_mismatch',**base)
    return SO101Decision(action=action,accepted=True,reason='accepted',**base)


def validate_decision(value,observation,min_confidence=MIN_CONFIDENCE):
    """Strict IPC response validation; never turns rejection into model hold."""
    fields=set(SO101Decision().to_dict())
    if not isinstance(value,dict) or set(value)!=fields or value['schema']!=DECISION_SCHEMA or type(value['accepted']) is not bool:raise ValueError('Invalid SO101 decision schema')
    obs=validate_observation(observation)
    if value['axis_index']!=obs['axis_index'] or type(value['axis_index']) is not type(obs['axis_index']):raise ValueError('Mismatched decision axis')
    if not _number(value['confidence']) or not 0<=value['confidence']<=1 or not isinstance(value['probabilities'],dict):raise ValueError('Invalid confidence or probabilities')
    allowed={'accepted','unavailable','invalid_input','invalid_response','low_confidence','criterion_mismatch','busy','timeout','network_error','api_error','response_too_large','call_budget','closed','cancelled','expired','adapter_error','worker_error'}
    if not isinstance(value['reason'],str) or value['reason'] not in allowed:raise ValueError('Invalid decision reason')
    if value['proposed_choice'] is None:
        if value['reason'] in ('accepted','low_confidence','criterion_mismatch') or value['accepted'] or value['action'] is not None or value['probabilities'] or value['confidence']!=0:raise ValueError('Invalid rejected decision')
        return SO101Decision(**value)
    reparsed=parse_response({'model':ALIYUN_MODEL,'answers':{'action':{'type':'choice','choice':value['proposed_choice'],'confidence':value['confidence'],'probabilities':value['probabilities']}}},obs,min_confidence)
    if reparsed.to_dict()!=value:raise ValueError('Inconsistent decision result')
    return reparsed


class SO101JevAdapter:
    """Explicit-key pooled HTTP boundary. No config, env-secret, or file access."""
    def __init__(self,*,api_key,endpoint=ALIYUN_ENDPOINT,model=ALIYUN_MODEL,min_confidence=MIN_CONFIDENCE,timeout=8.,max_calls=30,transport=None,use_environment_network=False):
        if endpoint!=ALIYUN_ENDPOINT or model!=ALIYUN_MODEL:raise ValueError('Only the approved Aliyun endpoint/model pair is allowed')
        if not isinstance(api_key,str) or not api_key or any(not 33<=ord(c)<=126 for c in api_key):raise ValueError('Explicit server-side key required')
        if not _number(min_confidence) or not MIN_CONFIDENCE<=min_confidence<=1:raise ValueError('Confidence cannot be reduced below .65')
        if not _number(timeout) or not 0<timeout<=30 or type(max_calls) is not int or not 1<=max_calls<=200:raise ValueError('Invalid request bounds')
        if type(use_environment_network) is not bool:raise ValueError('Explicit network mode must be boolean')
        self.last_request=None
        self._key=api_key;self.min_confidence=min_confidence;self.calls=0;self.max_calls=max_calls
        self._lock=threading.Lock();self._closed=False
        self._client=httpx.Client(timeout=httpx.Timeout(timeout),follow_redirects=False,trust_env=use_environment_network,transport=transport,
                                  limits=httpx.Limits(max_connections=1,max_keepalive_connections=1))
    @classmethod
    def from_config(cls,path,*,max_calls=12,timeout=None,use_environment_network=False):
        """Load cerebellum settings independently; no legacy robot dependency."""
        from embodied_arm.providers.configuration import load_configuration,request_timeout
        try:
            settings=load_configuration(path,component='cerebellum')['cerebellum']
            bounded_timeout=request_timeout(settings,timeout)
            return cls(api_key=settings.api_key,endpoint=settings.endpoint,model=settings.model,
                min_confidence=settings.min_confidence,timeout=bounded_timeout,max_calls=max_calls,
                use_environment_network=use_environment_network)
        except (OSError,ValueError,TypeError,AttributeError):
            raise ValueError('Cannot load the approved external SO101 Aliyun configuration') from None

    def status(self):
        return {'configured':not self._closed,'model':ALIYUN_MODEL,'endpoint':ALIYUN_ENDPOINT,
                'calls':self.calls,'max_calls':self.max_calls,'min_confidence':self.min_confidence,
                'fallback':False,'source':'rgb_only_local_target_and_encoder_fk'}
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        with self._lock:self._closed=True;self._client.close()
    def decide(self,observation):
        try:obs=validate_observation(observation);body=request_preview(obs)['body']
        except (ValueError,TypeError,KeyError,RecursionError):return SO101Decision(reason='invalid_input')
        axis=obs['axis_index']
        if obs['robot']['busy']:return SO101Decision(reason='busy',axis_index=axis)
        with self._lock:
            if self._closed:return SO101Decision(reason='closed',axis_index=axis)
            if self.calls>=self.max_calls:return SO101Decision(reason='call_budget',axis_index=axis)
            self.calls+=1
            telemetry=RequestTelemetry(self, 'cerebellum')
            def finish(result):
                telemetry.finish(result.reason)
                return result
            try:
                with self._client.stream('POST',ALIYUN_ENDPOINT,headers={'Authorization':'Bearer '+self._key,'Accept':'application/json'},json=body) as response:
                    telemetry.response(response)
                    if response.status_code!=200:return finish(SO101Decision(reason='api_error',axis_index=axis))
                    raw=bytearray()
                    for chunk in response.iter_bytes():
                        raw.extend(chunk)
                        if len(raw)>MAX_RESPONSE_BYTES:return finish(SO101Decision(reason='response_too_large',axis_index=axis))
                payload=json.loads(raw)
                telemetry.payload(payload)
                return finish(parse_response(payload,obs,self.min_confidence))
            except httpx.TimeoutException:return finish(SO101Decision(reason='timeout',axis_index=axis))
            except httpx.HTTPError:return finish(SO101Decision(reason='network_error',axis_index=axis))
            except (ValueError,TypeError,UnicodeError,RecursionError):return finish(SO101Decision(reason='invalid_response',axis_index=axis))
