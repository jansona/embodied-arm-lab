"""Bounded Jev local-control experiment. No brain planning or rule-selected moves."""
import json, math, time
from pathlib import Path
from embodied_arm.paths import WORKSPACE_ROOT
import httpx
from embodied_arm.providers.adapters import ACTIONS, Decision, MAX_RESPONSE_BYTES, validate_provider
from embodied_arm.legacy.panda.control import apply_action

VERSION='geometry-criteria-v4'
CANDIDATES=('x_plus','x_minus','hold')
CRITERIA={
 'x_plus':'Choose this when the target is to the positive-X side (right) of the gripper beyond tolerance and a positive-X step is safe. This action increases world X by up to 0.02 metres, capped at the remaining target distance.',
 'x_minus':'Choose this when the target is to the negative-X side (left) of the gripper beyond tolerance and a negative-X step is safe. This action decreases world X by up to 0.02 metres, capped at the remaining target distance.',
 'hold':'Choose this when the gripper is within target tolerance, or no available movement is safe. This action leaves the gripper still.'}


def make_body(model,current,target,encoding='numeric',reverse=False,distance_band=False):
    if encoding not in ('numeric','relational'):raise ValueError('Unknown encoding')
    for vec in (current,target):
        if not isinstance(vec,list) or len(vec)!=3 or not all(type(x) in (int,float) and math.isfinite(x) for x in vec):raise ValueError('Invalid coordinates')
    if max(abs(target[i]-current[i]) for i in (1,2))>.01:raise ValueError('This experiment is X-only')
    error=target[0]-current[0]
    state={'frame':'robot-base Cartesian world frame; positive X is right, negative X is left',
           'task':'Move the gripper to this local target while keeping Y, Z and fingers unchanged.',
           'step_m':.02,'tolerance_m':.01,'path_clear':True,
           'positive_step_in_workspace':.25<=current[0]+.02<=.7,
           'negative_step_in_workspace':.25<=current[0]-.02<=.7}
    if encoding=='numeric':
        state.update(current_m=dict(zip('xyz',current)),target_m=dict(zip('xyz',target)))
    else:
        state.update(target_relative_to_gripper='right' if error>.01 else 'left' if error<-.01 else 'aligned',
                     x_distance_m=round(abs(error),5),y_aligned=True,z_aligned=True)
    if distance_band:
        state['measured_x_distance_m']=round(abs(error),6)
        state['measured_distance_band']='inside_tolerance' if abs(error)<=.01 else 'outside_tolerance'
    criteria=dict(CRITERIA)
    if distance_band:
        criteria['x_plus']='The measured_distance_band is outside_tolerance AND target X is greater than current X (right side). Both conditions are required. Increase world X by up to 0.02 metres, capped at the remaining target distance.'
        criteria['x_minus']='The measured_distance_band is outside_tolerance AND target X is less than current X (left side). Both conditions are required. Decrease world X by up to 0.02 metres, capped at the remaining target distance.'
        criteria['hold']='The measured_distance_band is inside_tolerance. The gripper is already close enough even when its coordinates are not exactly equal to target. Stay still; do not chase sub-tolerance error.'
    keys=list(reversed(CANDIDATES)) if reverse else CANDIDATES
    return {'model':model,'state':state,'questions':{'action':{'type':'choice',
        'instructions':'Select the single local action whose criterion is satisfied by the observed state. The criteria define the decision boundaries. Positive/negative X refer to the fixed world frame.',
        'criteria':{k:criteria[k] for k in keys}}}}


def make_axis_body(model,current,target,axis='x',reverse=False):
    if axis not in 'xyz' or len(axis)!=1:raise ValueError('Invalid axis')
    idx='xyz'.index(axis)
    if any(abs(target[i]-current[i])>.01 for i in range(3) if i!=idx):raise ValueError('Local phase must be single-axis')
    for vec in (current,target):
        if len(vec)!=3 or not all(type(v) in (int,float) and math.isfinite(v) for v in vec):raise ValueError('Invalid coordinates')
    distance=abs(target[idx]-current[idx]);positive=axis+'_plus';negative=axis+'_minus'
    criteria={positive:f'The measured_distance_band is outside_tolerance AND measured_target_side is positive. Both conditions are required. Increase world {axis.upper()} by up to 0.02 metres, capped at the remaining target distance.',
              negative:f'The measured_distance_band is outside_tolerance AND measured_target_side is negative. Both conditions are required. Decrease world {axis.upper()} by up to 0.02 metres, capped at the remaining target distance.',
              'hold':'The measured_distance_band is inside_tolerance. The gripper is already close enough even if coordinates are not exactly equal. Stay still; do not chase sub-tolerance error.'}
    keys=list(reversed(criteria)) if reverse else list(criteria)
    return {'model':model,'state':{'frame':'robot-base Cartesian world frame','axis_phase':axis,'phase_source':'fixed axis-order task decomposition, not a selected movement direction',
        'current_m':dict(zip('xyz',current)),'target_m':dict(zip('xyz',target)),'measured_axis_distance_m':round(distance,6),'measured_signed_axis_error_m':round(target[idx]-current[idx],6),'measured_target_side':'positive' if target[idx]>current[idx] else 'negative' if target[idx]<current[idx] else 'aligned',
        'measured_distance_band':'inside_tolerance' if distance<=.01 else 'outside_tolerance','step_m':.02,'tolerance_m':.01},
        'questions':{'action':{'type':'choice','instructions':'Select the local action whose criterion is satisfied. Keep the other coordinates and fingers unchanged. Criteria define mutually exclusive decision boundaries.',
        'criteria':{k:criteria[k] for k in keys}}}}


class MicroController:
    def __init__(self,provider,transport=None,max_calls=10):
        if type(max_calls) is not int or not 1<=max_calls<=200:raise ValueError("Invalid call cap")
        self.max_calls=max_calls
        validate_provider(provider.endpoint,provider.model)
        self.provider=provider;self.calls=0
        self.client=httpx.Client(timeout=httpx.Timeout(provider.timeout),follow_redirects=False,trust_env=provider.use_environment_network,transport=transport)
    def close(self):self.client.close()
    def decide(self,current,target,encoding='relational',reverse=False,distance_band=False,axis=None):
        body=make_axis_body(self.provider.model,current,target,axis,reverse) if axis else make_body(self.provider.model,current,target,encoding,reverse,distance_band)
        return self._send(body)
    def _send(self,body):
        if self.calls>=self.max_calls:return Decision(reason='call_budget'),{}
        candidates=tuple(body['questions']['action']['criteria'])
        self.calls+=1;trace,mark=self.provider._begin_transport();raw_choice=None
        try:
            with self.client.stream('POST',self.provider.endpoint,headers={'Authorization':'Bearer '+self.provider._api_key,'Accept':'application/json'},json=body,extensions={'trace':trace}) as response:
                mark('response_headers_received')
                if response.status_code!=200:return Decision(reason='api_error'),{'http_status':response.status_code}
                raw=bytearray()
                for part in response.iter_bytes():
                    raw.extend(part)
                    if len(raw)>MAX_RESPONSE_BYTES:return Decision(reason='response_too_large'),{}
            mark('response_complete');payload=json.loads(raw);self.provider._server_latency(payload)
            answer=payload.get('answers',{}).get('action') if isinstance(payload,dict) else None
            if not isinstance(answer,dict):return Decision(reason='invalid_response'),{}
            probabilities=answer.get('probabilities')
            if not isinstance(probabilities,dict) or set(probabilities)!=set(candidates) or answer.get('choice') not in candidates:return Decision(reason='invalid_response'),{}
            safe=dict(answer);safe['probabilities']={a:probabilities.get(a,0.0) for a in ACTIONS}
            # Normalize only the 3 supplied rounded categorical probabilities.
            values=list(probabilities.values())
            if not all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in values):return Decision(reason='invalid_response'),{}
            total=sum(values);rounded=all(abs(v*100-round(v*100))<1e-8 for v in values)
            if abs(total-1)>(.015000001 if rounded else .001):return Decision(reason='invalid_response'),{}
            safe['probabilities']={a:v/total for a,v in safe['probabilities'].items()}
            result=self.provider._validate({'answers':{'action':safe}})
            return result,{'request':body,'prompt_version':VERSION+'-exclusive-v3','evaluated_candidates':list(candidates),'raw_probabilities':dict(probabilities)}
        except httpx.HTTPError as error:
            self.provider._transport_error(error)
            return Decision(reason='timeout' if isinstance(error,httpx.TimeoutException) else 'network_error'),{}
        except (ValueError,TypeError,UnicodeError):return Decision(reason='invalid_response'),{}
        finally:mark('elapsed_seconds')


def run(provider):
    controller=MicroController(provider);rows=[];physical=[]
    try:
        for encoding,sign,reverse in [('numeric',1,False),('relational',1,True),('numeric',-1,True),('relational',-1,False)]:
            current=[.45,0,.30];target=[.45+.06*sign,0,.30]
            decision,metadata=controller.decide(current,target,encoding,reverse)
            row={'encoding':encoding,'sign':sign,'candidate_order_reversed':reverse,'decision':decision.to_dict(),'correct':decision.proposed_action==('x_plus' if sign>0 else 'x_minus'),'transport':dict(provider.last_transport),**metadata}
            rows.append(row)
            print(json.dumps({'phase':'calibration','trial':len(rows),'correct':row['correct'],'decision':decision.to_dict(),'elapsed_seconds':provider.last_transport.get('elapsed_seconds')}),flush=True)
        if all(row['correct'] and row['decision']['accepted'] for row in rows if row['encoding']=='relational'):
            from embodied_arm.simulation.simulator import Simulation
            sim=Simulation();initial=sim.state()['ee'];target=[initial[0]+.04,initial[1],initial[2]]
            try:
                for _ in range(3):
                    before=sim.state()['ee'];decision,metadata=controller.decide(before,target,'relational',bool(len(physical)%2))
                    entry={'before':before,'target':target,'decision':decision.to_dict(),'transport':dict(provider.last_transport),**metadata,'executed':False}
                    physical.append(entry)
                    if not decision.accepted:break
                    if decision.action=='hold':
                        entry['at_target']=abs(target[0]-before[0])<=.01
                        break
                    apply_action(sim,decision.action,origin='jev microcontrol')
                    while sim.state()['busy']:sim.tick(24)
                    entry.update(executed=True,after=sim.state()['ee'],distance_before=math.dist(before,target),distance_after=math.dist(sim.state()['ee'],target))
                sim.stop()
            finally:sim.close()
        return {'prompt_version':VERSION,'calls':controller.calls,'calibration':rows,'physical':physical,'note':'Same three candidates and unchanged confidence threshold. Geometry encoding does not select an action. Separate simulator, no user scene changes.'}
    finally:controller.close()


def main():
    from embodied_arm.legacy.panda.runner import JevRunner
    class NoRobot:
        def stop(self):pass
    runner=JevRunner(NoRobot(),use_environment_network=True)
    runner.configure(WORKSPACE_ROOT.parent/'jev-arm-config.json')
    result=run(runner._adapter)
    WORKSPACE_ROOT.joinpath('artifacts/microcontrol-experiment.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'phase':'complete','calls':result['calls'],'calibration_correct':sum(r['correct'] for r in result['calibration']),'physical_steps':len(result['physical'])}),flush=True)
if __name__=='__main__':main()


class XMicroAdapter:
    """Opt-in local-X adapter. Rejects multi-axis tasks instead of faking support."""
    def __init__(self,provider,transport=None):
        self.provider=provider;self.micro=MicroController(provider,transport)
    @property
    def last_transport(self):return self.provider.last_transport
    def close(self):self.micro.close()
    def decide(self,state):
        objective=state.get('objective') if isinstance(state,dict) else None
        if not isinstance(objective,dict) or objective.get('id') not in ('local_target','hover_reach'):
            return Decision(reason='invalid_state')
        try:
            decision,_=self.micro.decide(state['end_effector'],objective['target_xyz'],'numeric',False,True)
            return decision
        except (ValueError,KeyError,TypeError):return Decision(reason='invalid_state')


class PhasedMicroAdapter:
    max_worker_calls=200
    max_worker_seconds=420
    """Fixed XYZ phase decomposition; all movement directions selected by Jev."""
    def __init__(self,provider,transport=None):
        self.provider=provider;self.micro=MicroController(provider,transport,max_calls=200)
    @property
    def last_transport(self):return self.provider.last_transport
    def close(self):self.micro.close()
    def decide(self,state):
        try:
            objective=state['objective']
            if objective['id']=='gripper':
                return self._gripper(state,objective['target_state'])
            if objective['id'] not in ('local_target','hover_reach'):return Decision(reason='invalid_state')
            current,target=state['end_effector'],objective['target_xyz']
            if len(current)!=3 or len(target)!=3:return Decision(reason='invalid_state')
            axis=next((i for i in range(3) if abs(target[i]-current[i])>.01),0)
            phase_target=list(current);phase_target[axis]=target[axis]
            result,_=self.micro.decide(current,phase_target,'numeric',False,True,axis='xyz'[axis])
            return result
        except (KeyError,ValueError,TypeError,StopIteration):return Decision(reason='invalid_state')

    def _gripper(self,state,target):
        if target not in ('open','closed'):return Decision(reason='invalid_state')
        width=state['gripper_width']
        if type(width) not in (int,float) or not math.isfinite(width):return Decision(reason='invalid_state')
        body={'model':self.provider.model,'state':{'desired_gripper_state':target,'measured_width_m':width,
            'measured_gripper_band':'open' if width>=.075 else 'closed_or_contact',
            'current_finger_gap':'wide' if width>=.075 else 'narrow',
            'target_finger_gap':'wide' if target=='open' else 'narrow'},'questions':{'action':{'type':'choice',
            'instructions':'Select the gripper action whose mutually exclusive criterion matches current and desired state. Do not move the arm.',
            'criteria':{'open':'Transition narrow -> wide: current_finger_gap is narrow AND target_finger_gap is wide. Increase finger separation.',
                        'close':'Transition wide -> narrow: current_finger_gap is wide AND target_finger_gap is narrow. Decrease finger separation until contact or mechanical limit.',
                        'hold':'No transition: current_finger_gap and target_finger_gap are BOTH wide, OR BOTH narrow. Only identical current and target gaps satisfy this criterion. Keep fingers unchanged.'}}}}
        decision,_=self.micro._send(body)
        return decision
