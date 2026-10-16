"""Robot-independent capability boundary; no hardware transport exists here."""
from typing import Protocol

class ArmBackend(Protocol):
    def state(self) -> dict: ...
    def manual(self, action: str, values=None): ...
    def stop(self): ...

def observation(backend:ArmBackend)->dict:
    s=backend.state()
    hover=getattr(backend,'task_mode','pick_place')=='hover_reach'
    objective={'id':'hover_reach','target_xyz':[s['blocks']['red'][0],s['blocks']['red'][1],.44],'tolerance_m':.01} if hover else {'id':'pick_place','target_xyz':s['goal'],'tolerance_m':.055}
    return {'schema':'arm-observation-v1','simulation_only':True,
        'robot':{'name':'Franka Panda','arm_dof':7,'gripper':'parallel fingers','units':{'position':'metres','joints':'radians'}},
        'task':'Move the gripper to the high hover waypoint above the red block. Do not descend or grasp yet.' if hover else 'Place the red block at goal using physical grasp contact.',
        'objective':objective,
        'end_effector':s['ee'],'joints':s['joints'],'gripper_width':s['gripper_width'],
        'objects':s['blocks'],'goal':s['goal'],'workspace':s['workspace'],
        'busy':s['busy'],'paused':s['paused'],'metrics':s['metrics'],
        'capabilities':{'cartesian_step_m':.02,'actions':['x_plus','x_minus','y_plus','y_minus','z_plus','z_minus','open','close','hold']}}

def apply_action(backend:ArmBackend, action:str, origin='jev',step_m=.02):
    """Apply one bounded high-level action through the same manual safety gate.

    Caller must await completion and get a fresh observation before deciding again.
    This boundary intentionally knows nothing about API keys or model transport.
    """
    import math
    if type(step_m) not in (int,float) or not math.isfinite(step_m) or not 0<step_m<=.02:raise ValueError('Invalid bounded step')
    s=backend.state()
    if s['busy'] or s['paused']:raise ValueError('Robot must be idle and resumed')
    if action=='hold':return
    if action in ('open','close'):backend.manual(action)
    elif action in ('x_plus','x_minus','y_plus','y_minus','z_plus','z_minus'):
        xyz=list(s['ee']);xyz['xyz'.index(action[0])]+=step_m if action.endswith('plus') else -step_m
        backend.manual('xyz',xyz)
    else:raise ValueError('Unknown bounded action')
    # Queue origin is metadata, never a safety override.
    with backend.lock:
        backend.mode=origin
        if backend.queue:backend.queue[-1]['origin']=origin
