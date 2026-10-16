"""Resume one provenance-checked interrupted stack through the existing local API.

No new physics world, model planning, reset, debug injection, or arbitrary HTTP
proxy. TaskExecutor/SequenceExecutor remain the only motion engines. The original
page keeps its interruption record; this runner writes a separate measured audit.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import httpx
from embodied_arm.providers.brain import validate_plan
from embodied_arm.legacy.panda.control import observation
from embodied_arm.runtime.ipc import FileBridgeAdapter, validate_decision
from embodied_arm.legacy.panda.sequence_executor import SequenceExecutor
from embodied_arm.legacy.panda.stack_plans import symbolic_step_result
from embodied_arm.legacy.panda.task_executor import TaskExecutor

from embodied_arm.paths import WORKSPACE_ROOT, RUNTIME_ROOT
ROOT=WORKSPACE_ROOT
API='http://127.0.0.1:8765'
MAX_CALLS=130
MAX_SECONDS=300


def _read(path):
    raw=Path(path).read_bytes()
    if len(raw)>262144:raise ValueError('Checkpoint too large')
    return raw,json.loads(raw)


def load_checkpoint(snapshot_path,plan_path):
    raw,snapshot=_read(snapshot_path);_,record=_read(plan_path)
    required={'plan','source_call_id','model','provenance','accepted','snapshot_sha256',
              'failed_objective','rejected_decision'}
    if not required<=set(record) or set(record)-required-{'controller_call_id'}:
        raise ValueError('Invalid checkpoint provenance fields')
    if (record['accepted'] is not True or record['model']!='qwen3.8-max' or record['provenance']!='live'
            or not isinstance(record['source_call_id'],str) or not record['source_call_id']
            or record['snapshot_sha256']!=hashlib.sha256(raw).hexdigest()):
        raise ValueError('Checkpoint provenance mismatch')
    plan=validate_plan(record['plan'],snapshot,check_state=False).to_dict()
    if plan!=snapshot['chat']['plan'] or plan['action']!='sequence':raise ValueError('Plan is not the saved accepted sequence')
    records=snapshot['chat']['task']['steps']
    failed=[i for i,s in enumerate(records) if s.get('status')=='failed']
    if len(failed)!=1:raise ValueError('Exactly one interrupted step is required')
    index=failed[0]
    if (index<1 or index>=len(plan['steps']) or any(s.get('status')!='completed' or s.get('outcome',{}).get('verified') is not True for s in records[:index])
            or any(s.get('status')!='pending' for s in records[index+1:])):
        raise ValueError('Completed steps must be an untouched verified prefix')
    for i,entry in enumerate(records):
        if any(entry.get(k)!=v for k,v in plan['steps'][i].items()):raise ValueError('Step provenance mismatch')
    step=plan['steps'][index]
    if (step['action'],step['object_id'],step['target_id'])!=('place','red','goal'):
        raise ValueError('Only the approved interrupted red release is supported')
    if record['failed_objective']!={'id':'gripper','target_state':'open'}:raise ValueError('Not the failed open primitive')
    decision=validate_decision(record['rejected_decision'])
    if decision.accepted or decision.proposed_action!='open' or decision.reason!='low_confidence':
        raise ValueError('Rejected gripper decision evidence required')
    if snapshot['chat']['controller']!='jev' or snapshot['chat']['provenance']!='live':
        raise ValueError('Original plan must be live Qwen with Jev execution')
    if snapshot['busy'] or not snapshot['paused']:raise ValueError('Checkpoint must be stopped')
    # The existing place primitive was interrupted immediately before release.
    # Resume exactly its final open + vertical retract, never its completed travel.
    red=snapshot['blocks']['red'];ee=snapshot['ee'];goal=snapshot['goal']
    if (math.dist(red[:2],goal[:2])>.03 or abs(red[2]-goal[2])>.03
            or math.dist(red,ee)>.08 or not 0<=snapshot['gripper_width']<=.065):
        raise ValueError('Interrupted release geometry is not supported')
    return snapshot,record,index


class HTTPArmBackend:
    """Fixed simulation API adapter with monotonic action/step guards."""
    task_mode='pick_place'
    def __init__(self,snapshot,*,transport=None):
        self.client=httpx.Client(base_url=API,transport=transport,trust_env=False,
                                 follow_redirects=False,timeout=3)
        self.lock=threading.RLock();self.queue=[];self.mode='Jev continuation via local actuator API'
        self.expected_actions=snapshot['metrics']['actions'];self.last_physics=snapshot['metrics']['physics_steps']
        self.allow_paused=True;self.cancelled=False;self.failure_reason=None;self.expected_seed=snapshot['seed']
        actual=self._get()
        self._guard(actual)
        if (not actual['paused'] or actual['busy'] or actual['metrics']['physics_steps']!=snapshot['metrics']['physics_steps']
            or any(math.dist(actual['blocks'][k],snapshot['blocks'][k])>.001 for k in ('red','blue'))
            or math.dist(actual['ee'],snapshot['ee'])>.001
            or max(abs(a-b) for a,b in zip(actual['joints'],snapshot['joints']))>.003
            or abs(actual['gripper_width']-snapshot['gripper_width'])>.002):
            raise ValueError('Live scene changed; continuation refused')
    def _get(self):
        response=self.client.get('/api/state');response.raise_for_status()
        if len(response.content)>262144:raise ValueError('State response too large')
        return response.json()
    def _guard(self,state):
        if (self.cancelled or state.get('engine')!='PyBullet DIRECT / TinyRenderer'
            or state.get('seed')!=self.expected_seed
            or state['metrics']['actions']!=self.expected_actions
            or state['metrics']['physics_steps']<self.last_physics
            or (state['paused'] and not self.allow_paused)):
            self.cancelled=True;self.failure_reason='external_stop_reset_or_control';raise RuntimeError('User stop/reset or external control detected')
        self.last_physics=state['metrics']['physics_steps']
    def state(self):
        with self.lock:
            state=self._get();self._guard(state);return state
    @property
    def paused(self):return self.state()['paused']
    @paused.setter
    def paused(self,value):
        if value:raise ValueError('Use stop for a safe pause')
        with self.lock:
            if self.cancelled:raise RuntimeError('Continuation cancelled')
            self.state();self.allow_paused=False
            response=self.client.post('/api/control',json={'action':'resume'});response.raise_for_status()
            self._guard(response.json())
    def manual(self,action,values=None):
        if action not in ('xyz','open','close'):raise ValueError('Unsupported continuation actuator action')
        with self.lock:
            before=self.state()
            if before['paused'] or before['busy']:raise RuntimeError('Actuator is not idle')
            if action=='xyz' and (not isinstance(values,list) or len(values)!=3
                or not all(type(v) in (float,int) and math.isfinite(v) for v in values)
                or math.dist(before['ee'],values)>.020001):
                raise ValueError('Continuation movement exceeds two centimetres')
            body={'action':action}
            if values is not None:body['values']=values
            # Never retry a POST with uncertain delivery.
            try:
                response=self.client.post('/api/control',json=body);response.raise_for_status()
                after=response.json();self.expected_actions+=1;self._guard(after)
            except Exception:
                self.cancelled=True;self.stop();raise
    def stop(self):
        with self.lock:
            self.allow_paused=True
            try:self.client.post('/api/control',json={'action':'stop'}).raise_for_status()
            except Exception:self.cancelled=True
    def check_decision_fresh(self,obs):
        now=self.state()
        if (now['paused'] or now['busy'] or math.dist(now['ee'],obs['end_effector'])>.003
            or abs(now['gripper_width']-obs['gripper_width'])>.003
            or any(math.dist(now['blocks'][k],obs['objects'][k])>.005 for k in ('red','blue'))):
            self.cancelled=True;self.failure_reason='scene_changed_during_decision';raise RuntimeError('Observation changed while decision was pending')
    def close(self):self.client.close()


class GuardedController:
    def __init__(self,adapter,backend,deadline,max_calls=MAX_CALLS):
        self.adapter=adapter;self.backend=backend;self.deadline=deadline;self.calls=0;self.decisions=[];self.max_calls=max_calls
    def decide(self,obs):
        if self.calls>=self.max_calls or time.monotonic()>=self.deadline:raise RuntimeError('Continuation budget exhausted')
        self.calls+=1
        decision=self.adapter.decide(obs)
        entry={'call':self.calls,'objective':copy.deepcopy(obs.get('objective')),
               'decision':decision.to_dict(),'applied_eligible':False}
        self.decisions.append(entry)
        self.backend.check_decision_fresh(obs)
        if time.monotonic()>=self.deadline:raise RuntimeError('Continuation response arrived too late')
        entry['applied_eligible']=True
        return decision
    def cancel(self):
        if hasattr(self.adapter,'cancel'):self.adapter.cancel()


def execute(snapshot,provenance,index,backend,controller,*,deadline,audit_path,
            leaf_factory=TaskExecutor,sequence_factory=SequenceExecutor):
    plan=provenance['plan'];failed=plan['steps'][index]
    report={'schema':'stack-continuation-audit-v1','provenance':'reused_live',
            'source_call_id':provenance['source_call_id'],'snapshot_sha256':provenance['snapshot_sha256'],
            'original_plan':plan,'completed_prefix':[s['step_id'] for s in plan['steps'][:index]],
            'max_calls':MAX_CALLS,'max_seconds':MAX_SECONDS,'new_qwen_calls':0,
            'status':'starting','remaining_step_results':[],
            'controller':'jev','actuator_transport':'existing local simulation API',
            'ui_note':'Original page retains interrupted task; manual actuator origin is legacy metadata. These decisions are Jev continuation.'}
    active=None
    def persist():
        report['jev_calls']=controller.calls
        report['decisions']=controller.decisions
        path=Path(audit_path);path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2));temp.replace(path)
    def await_done(executor):
        last=-1
        while executor.status()['status'] in ('running','replanning'):
            if time.monotonic()>=deadline or backend.cancelled:
                executor.stop();raise RuntimeError('Continuation cancelled or expired')
            # Independent observation catches user Stop during a pending API call.
            backend.state()
            if controller.calls!=last:
                last=controller.calls;report['active_task']=executor.status();persist()
            time.sleep(.05)
        result=executor.status()
        if result['status']!='completed':raise RuntimeError('Continuation executor stopped: '+str(result.get('reason')))
        return result
    try:
        persist();report['status']='resuming_failed_open_and_retract'
        active=leaf_factory(backend,controller=controller)
        # These are the unexecuted final primitives of the original grounded
        # place recipe; neither original travel nor a completed step is replayed.
        active.start({'steps':[{'action':'open'},{'action':'lift','height_m':.27}]},
                     max_seconds=min(120,max(.01,deadline-time.monotonic())),max_actions=30)
        await_done(active)
        outcome=symbolic_step_result(failed,snapshot,backend.state())
        report['remaining_step_results'].append({'source_step_id':failed['step_id'],'outcome':outcome})
        if not outcome.get('verified'):raise RuntimeError('Resumed placement was not physically verified')
        suffix=copy.deepcopy(plan['steps'][index+1:])
        if suffix:
            mapping={f's{i+1}':step['step_id'] for i,step in enumerate(suffix)}
            for i,step in enumerate(suffix):step['step_id']=f's{i+1}'
            remaining={'action':'sequence','object_id':None,'target_id':None,
                       'message':'复用原始 Qwen 计划的未完成后缀，不重新规划。','steps':suffix}
            validate_plan(remaining,backend.state())
            report['source_step_mapping']=mapping;report['status']='executing_original_suffix';persist()
            active=sequence_factory(backend,controller=controller,allow_replan=False)
            active.start(remaining);result=await_done(active)
            for step in result['steps']:
                report['remaining_step_results'].append({'source_step_id':mapping[step['step_id']],
                                                         'outcome':step.get('outcome')})
        report['status']='completed';report['verified']=True;report['final_scene']=backend.state()
    except Exception as exc:
        failure_status=active.status() if active else None
        if active:active.stop()
        backend.stop();report.update(status='stopped',verified=False,reason='continuation_safety_or_executor_stop')
        if failure_status:report['active_task']=failure_status
        if getattr(backend,'failure_reason',None):report['reason']=backend.failure_reason
        # No raw exceptions, HTTP contents or environment values enter the audit.
    finally:
        controller.cancel();backend.stop();report['jev_calls']=controller.calls;persist()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--audit',type=Path,default=ROOT/'artifacts'/'stack-continuation-audit.json')
    args=parser.parse_args()
    lock_path=RUNTIME_ROOT/'stack-continuation.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    process_lock=lock_path.open('a')
    try:fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another continuation is already running')
    snapshot,provenance,index=load_checkpoint(args.snapshot,args.plan)
    backend=HTTPArmBackend(snapshot);deadline=time.monotonic()+MAX_SECONDS
    adapter=FileBridgeAdapter()
    if not adapter.status().get('available'):raise RuntimeError('Jev worker is not ready; no motion started')
    controller=GuardedController(adapter,backend,deadline)
    try:
        result=execute(snapshot,provenance,index,backend,controller,deadline=deadline,audit_path=args.audit)
        print(json.dumps({'status':result['status'],'verified':result.get('verified',False),
                          'jev_calls':result['jev_calls'],'new_qwen_calls':0,'audit':str(args.audit)}))
    finally:backend.close()

if __name__=='__main__':main()
