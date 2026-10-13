"""Fixed, owner-only simulator decision IPC. No URLs, headers, or credentials."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid
from adapters import ACTIONS, Decision

IPC_ROOT = Path(__file__).resolve().parent / '.runtime' / 'jev-ipc'
MAX_BYTES = 32767
_NAME = re.compile(r'(?:[0-9a-f]{32}\.(?:job|result|cancel|tmp)|ready\.json|last-run-audit\.json|run-summary\.json|worker\.lock)\Z')
ROBOT = {'name':'Franka Panda','arm_dof':7,'gripper':'parallel fingers','units':{'position':'metres','joints':'radians'}}
TASK = 'Place the red block at goal using physical grasp contact.'
GRIP_TASK = 'Set the gripper to the supplied target state.'
LOCAL_TASK = 'Move to the supplied local Cartesian target, then hold.'
HOVER_TASK = 'Move the gripper to the high hover waypoint above the red block. Do not descend or grasp yet.'
CAPABILITIES = {'cartesian_step_m':.02,'actions':list(ACTIONS)}
METRIC_NUMBERS = {'physics_steps','simulation_seconds','actions','red_max_height_m','goal_distance_m','reach_distance_m'}
METRIC_BOOLS = {'lifted','placed','completed'}


def number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and -1e12 < value < 1e12 and math.isfinite(value)


def vector(value, size):
    return isinstance(value,list) and len(value)==size and all(number(x) for x in value)


def validate_state(state):
    keys={'schema','simulation_only','robot','task','end_effector','joints','gripper_width','objects','goal','workspace','busy','paused','metrics','capabilities'}
    if not isinstance(state,dict) or set(state) not in (keys, keys|{'objective'}): raise ValueError('Invalid simulation fields')
    if state['schema']!='arm-observation-v1' or state['simulation_only'] is not True: raise ValueError('Simulation only')
    if state['robot']!=ROBOT or state['task'] not in (TASK,HOVER_TASK,LOCAL_TASK,GRIP_TASK) or state['capabilities']!=CAPABILITIES: raise ValueError('Invalid fixed simulation metadata')
    if not vector(state['end_effector'],3) or not vector(state['joints'],7) or not vector(state['goal'],3): raise ValueError('Invalid simulation coordinates')
    if not number(state['gripper_width']): raise ValueError('Invalid gripper')
    if not isinstance(state['objects'],dict) or set(state['objects'])!={'red','blue'} or not all(vector(x,3) for x in state['objects'].values()): raise ValueError('Invalid objects')
    objective=state.get('objective')
    if objective is not None:
        if not isinstance(objective,dict): raise ValueError('Invalid objective')
        if objective.get('id')=='hover_reach':
            if (set(objective)!={'id','target_xyz','tolerance_m'} or state['task']!=HOVER_TASK
                    or not vector(objective['target_xyz'],3)
                    or objective['target_xyz']!=[state['objects']['red'][0],state['objects']['red'][1],.44]
                    or objective['tolerance_m']!=.01): raise ValueError('Invalid hover objective')
        elif objective.get('id')=='gripper':
            if set(objective)!={'id','target_state'} or objective['target_state'] not in ('open','closed') or state['task']!=GRIP_TASK:raise ValueError('Invalid gripper objective')
        elif objective.get('id')=='local_target':
            if (set(objective)!={'id','target_xyz','tolerance_m'} or state['task']!=LOCAL_TASK
                    or not vector(objective['target_xyz'],3) or objective['tolerance_m']!=.01
                    or not all(lo<=v<=hi for v,(lo,hi) in zip(objective['target_xyz'],((.25,.7),(-.35,.35),(.022,.6))))): raise ValueError('Invalid local target')
        elif (objective!={'id':'pick_place','target_xyz':state['goal'],'tolerance_m':.055}
              or state['task']!=TASK): raise ValueError('Invalid objective')
    elif state['task']!=TASK or 'objective' in state: raise ValueError('Missing objective')
    if not isinstance(state['workspace'],list) or len(state['workspace'])!=3 or not all(vector(x,2) and x[0]<x[1] for x in state['workspace']): raise ValueError('Invalid workspace')
    if type(state['busy']) is not bool or type(state['paused']) is not bool: raise ValueError('Invalid motion state')
    metrics=state['metrics']
    if not isinstance(metrics,dict) or set(metrics)!=METRIC_NUMBERS|METRIC_BOOLS|{'finger_contacts'}: raise ValueError('Invalid metrics')
    if not all(number(metrics[k]) for k in METRIC_NUMBERS) or not all(type(metrics[k]) is bool for k in METRIC_BOOLS): raise ValueError('Invalid metrics values')
    contacts=metrics['finger_contacts']
    if not isinstance(contacts,dict) or set(contacts)!={'9','10'} or not all(type(v) is int and 0<=v<=1000 for v in contacts.values()): raise ValueError('Invalid contacts')
    return state


def normalize_state(state):
    raw=json.dumps(state,allow_nan=False,separators=(',',':')).encode()
    if len(raw)>MAX_BYTES: raise ValueError('Simulation state too large')
    return validate_state(json.loads(raw))


def validate_decision(data):
    if not isinstance(data,dict) or set(data)!={'action','confidence','probabilities','accepted','reason','proposed_action'}: raise ValueError('Invalid result')
    if data['action'] not in ACTIONS or not number(data['confidence']) or not 0<=data['confidence']<=1 or type(data['accepted']) is not bool: raise ValueError('Invalid result values')
    proposed=data['proposed_action']
    if proposed is not None and proposed not in ACTIONS: raise ValueError('Invalid proposed action')
    if data['accepted'] and proposed is not None and proposed!=data['action']: raise ValueError('Inconsistent proposed action')
    probs=data['probabilities']
    if not isinstance(probs,dict): raise ValueError('Invalid probabilities')
    if probs and (set(probs)!=set(ACTIONS) or not all(number(v) and 0<=v<=1 for v in probs.values()) or not math.isclose(sum(probs.values()),1,abs_tol=.001)): raise ValueError('Invalid probabilities')
    if data['accepted'] and (not probs or probs[data['action']]+1e-9<max(probs.values())): raise ValueError('Invalid accepted decision')
    if proposed is not None and (not probs or probs[proposed]+1e-9<max(probs.values())): raise ValueError('Invalid proposed distribution')
    allowed={'accepted','low_confidence','timeout','network_error','invalid_response','invalid_state','response_too_large','adapter_error','api_error','cancelled','expired'}
    if data['reason'] not in allowed: raise ValueError('Invalid result reason')
    if not data['accepted'] and data['action']!='hold': raise ValueError('Rejected decisions must hold')
    return Decision(**data)


class Store:
    """Directory-FD operations and O_NOFOLLOW avoid file symlink traversal."""
    def __init__(self, root=None):
        root=Path(IPC_ROOT if root is None else root)
        for parent in reversed(root.parents):
            if parent.is_symlink(): raise ValueError('Symlink directory rejected')
        for path in (root.parent,root):
            try: path.mkdir(mode=0o700)
            except FileExistsError: pass
            info=path.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid(): raise ValueError('Unsafe IPC directory')
            os.chmod(path,0o700,follow_symlinks=False)
        self.fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    def close(self): os.close(self.fd)
    def _name(self,name):
        if not isinstance(name,str) or not _NAME.fullmatch(name): raise ValueError('Invalid IPC name')
    def exists(self,name):
        self._name(name)
        try:
            info=os.stat(name,dir_fd=self.fd,follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode): raise ValueError('Unsafe IPC file')
            return True
        except FileNotFoundError: return False
    def write(self,name,data):
        self._name(name);self.exists(name)
        raw=json.dumps(data,allow_nan=False,separators=(',',':')).encode()
        if len(raw)>MAX_BYTES: raise ValueError('IPC payload too large')
        temporary=uuid.uuid4().hex+'.tmp'
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.fd)
        try:
            with os.fdopen(fd,'wb') as file: file.write(raw);file.flush();os.fsync(file.fileno())
            os.replace(temporary,name,src_dir_fd=self.fd,dst_dir_fd=self.fd)
        finally:
            try: os.unlink(temporary,dir_fd=self.fd)
            except FileNotFoundError: pass
    def read(self,name):
        self._name(name)
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=self.fd)
        with os.fdopen(fd,'rb') as file:
            info=os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_size>MAX_BYTES: raise ValueError('Unsafe IPC file')
            raw=file.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES: raise ValueError('IPC payload too large')
        return json.loads(raw)
    def delete(self,name):
        self._name(name)
        try: os.unlink(name,dir_fd=self.fd)
        except FileNotFoundError: pass
    def jobs(self): return sorted(n for n in os.listdir(self.fd) if re.fullmatch(r'[0-9a-f]{32}\.job',n))


class FileBridgeAdapter:
    def __init__(self):
        self._lock=threading.RLock();self._pending=set();self._epoch=0
    def status(self):
        return worker_status()
    def cancel(self):
        with self._lock:
            self._epoch+=1
            store=Store()
            try:
                for job in self._pending: store.write(job+'.cancel',{'cancelled':True})
            finally: store.close()
    def decide(self,state):
        try: state=normalize_state(state)
        except (ValueError,TypeError,OverflowError,RecursionError): return Decision(reason='invalid_state')
        store=None;job=uuid.uuid4().hex;deadline=time.monotonic()+20
        try:
            store=Store()
            with self._lock:
                epoch=self._epoch;self._pending.add(job)
                store.write(job+'.job',{'id':job,'expires_at':time.time()+20,'state':state})
            while time.monotonic()<deadline:
                with self._lock:
                    if epoch!=self._epoch: return Decision(reason='cancelled')
                if store.exists(job+'.result'):
                    result=validate_decision(store.read(job+'.result'))
                    with self._lock:
                        return result if epoch==self._epoch else Decision(reason='cancelled')
                time.sleep(.02)
            store.write(job+'.cancel',{'cancelled':True})
            return Decision(reason='timeout')
        except (OSError,ValueError,TypeError,RecursionError): return Decision(reason='invalid_response')
        finally:
            with self._lock: self._pending.discard(job)
            if store:
                for suffix in ('.job','.result'):
                    try: store.delete(job+suffix)
                    except OSError: pass
                store.close()


def worker_status():
    """Report availability only while the worker lock and deadline are live."""
    import fcntl
    safe={'available':False,'status':'offline','remaining_calls':0,'remaining_seconds':0.0}
    store=None;fd=None
    try:
        store=Store()
        fd=os.open('worker.lock',os.O_RDWR|os.O_NOFOLLOW,dir_fd=store.fd)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(fd,fcntl.LOCK_UN)
            return safe
        except BlockingIOError: pass
        data=store.read('ready.json')
        if not isinstance(data,dict) or data.get('status') not in ('ready','calling'): return safe
        calls,limit,deadline=data.get('calls'),data.get('max_calls'),data.get('deadline_unix')
        if type(calls) is not int or type(limit) is not int or not 0<=calls<=limit<=200 or not number(deadline): return safe
        remaining=max(0,deadline-time.time())
        if not 0<remaining<=420.5 or calls>=limit: return safe
        return {'available':True,'status':data['status'],'remaining_calls':limit-calls,'remaining_seconds':round(remaining,2)}
    except (OSError,ValueError,TypeError): return safe
    finally:
        if fd is not None: os.close(fd)
        if store is not None: store.close()
