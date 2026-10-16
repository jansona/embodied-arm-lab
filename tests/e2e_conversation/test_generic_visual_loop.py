"""Independent generic loop acceptance. MOCK semantics do not validate vision."""
from copy import deepcopy
import threading
import time
import pytest
from embodied_arm.services.visual_action_loop import VisualActionLoop
from embodied_arm.providers import visual_brain as qb, visual_cerebellum as jc
from tests.test_visual_providers import packet, observation, choice
from tests.test_jev_motion import FakeSim


class Sim(FakeSim):
    def __init__(self):
        super().__init__(); self.frames=[]
    def capture_frames(self):
        value=packet();value['frame_id']=f'mock-{len(self.frames)}';self.frames.append(value)
        return deepcopy(value)
    def manual(self,action,values=None,*,origin):
        assert origin in ('jev','qwen')
        super().manual(action,values,origin='offline_fixture')


class Brain:
    def __init__(self,complete_after=1):self.requests=[];self.complete_after=complete_after;self.hook=None;self.action='x_minus_0.5mm'
    def status(self):return {'available':True}
    def observe(self,task,packet,robot,history=None,review=False):
        frame=packet
        self.requests.append(deepcopy((frame,robot,history,review)))
        if self.hook:self.hook()
        return dict(accepted=True,reason='accepted',calls=len(self.requests),observation=observation(
            frame_id=frame['frame_id'],decision='completed' if len(self.requests)>self.complete_after else 'continue',
            action=None if len(self.requests)>self.complete_after else self.action))


class Jev:
    def __init__(self,action='y_minus_2mm'):self.action=action;self.requests=[];self.hook=None
    def status(self):return {'available':True}
    def decide(self,observation,robot):
        obs=observation
        self.requests.append(jc.request_preview(obs,robot))
        if self.hook:self.hook()
        return jc.parse_response(choice(self.action))


def settle(loop,timeout=5):
    limit=time.monotonic()+timeout
    while time.monotonic()<limit:
        state=loop.status()
        if not state['pending']:return state
        time.sleep(.01)
    raise AssertionError('MOCK loop did not settle')


def make(**options):
    sim=Sim();brain=Brain();jev=Jev()
    return sim,brain,jev,VisualActionLoop(sim,brain,jev,sleep=sim.external_tick,mode=options.pop('mode','two_model'),**options)


def test_fresh_images_and_full_33_choice_action_then_model_assessment():
    sim,brain,jev,loop=make();loop.submit('MOCK move arbitrary mug')
    state=settle(loop)
    assert state['state']=='completed' and len(sim.frames)==2 and len(jev.requests)==1
    assert brain.requests[0][0]['frame_id']!=brain.requests[1][0]['frame_id']
    assert set(jev.requests[0]['body']['questions']['action']['criteria'])==set(qb.ACTIONS)
    assert 'cube_edge_m' not in brain.requests[0][1]
    assert brain.requests[1][2]['applied_actions']==[{'action':'y_minus_2mm','applied':True}]
    assert state['outcome']['completion_basis']=='qwen_visual_assessment'
    assert state['outcome']['model_assessed_complete'] is True and 'verified' not in state['outcome']
    assert sim.commands[0][1][1] == pytest.approx(-.002)


@pytest.mark.parametrize('stage',['qwen','jev'])
def test_stop_rejects_late_provider_reply_and_new_request_until_settled(stage):
    sim,brain,jev,loop=make();entered=threading.Event();release=threading.Event()
    def block():entered.set();assert release.wait(3)
    (brain if stage=='qwen' else jev).hook=block
    try:
        loop.submit('MOCK wait');assert entered.wait(2)
        loop.stop();assert loop.status()['pending']
        with pytest.raises(ValueError,match='task_busy'):loop.submit('MOCK premature')
    finally:release.set()
    assert settle(loop)['state']=='cancelled' and not sim.commands
    brain.hook=jev.hook=None;brain.complete_after=0
    loop.submit('MOCK new command');assert settle(loop)['state']=='completed'


def test_review_after_32_calls_keeps_global_budget_and_hold_is_not_completion():
    sim,brain,jev,loop=make(max_calls=33);brain.complete_after=100;jev.action='hold'
    loop.submit('MOCK indefinite task');state=settle(loop)
    assert state['state']=='failed' and state['outcome']['reason']=='call_budget'
    assert state['controller_calls']==33 and state['controller_actions']==0
    assert len(jev.requests)==33 and not sim.commands
    assert [i for i,r in enumerate(brain.requests) if r[3]]==[32]
    assert len({r[0]['frame_id'] for r in brain.requests})==34
    assert state['outcome']['model_assessed_complete'] is False


def test_stale_capture_never_reaches_providers_or_motion():
    sim,brain,jev,loop=make();capture=sim.capture_frames
    def stale():
        value=capture();value['captured_at']-=121;return value
    sim.capture_frames=stale;loop.submit('MOCK inspect');state=settle(loop)
    assert state['outcome']['reason']=='stale_frame'
    assert not brain.requests and not jev.requests and not sim.commands


def test_real_physics_one_bounded_move_with_mock_visual_completion(monkeypatch):
    from embodied_arm.simulation.so101_simulation import SO101Simulation
    sim=SO101Simulation(camera_profile='observer_wrist');brain=Brain();jev=Jev('x_minus_0.5mm')
    monkeypatch.setattr(sim,'evaluation_state',lambda:(_ for _ in ()).throw(AssertionError('No ground truth')))
    stop=threading.Event()
    def tick():
        while not stop.is_set():sim.tick(4);time.sleep(.004)
    ticker=threading.Thread(target=tick,daemon=True);ticker.start()
    loop=VisualActionLoop(sim,brain,jev,mode='two_model',max_calls=2,max_seconds=30)
    try:
        initial=sim.proprioception()['ee'];loop.submit('MOCK move once then assess')
        state=settle(loop,30)
        assert state['state']=='completed',state
        assert state['controller_actions']==1 and sim.proprioception()['actions']==1
        assert sim.proprioception()['ee']!=initial and len(brain.requests)==2
        assert state['outcome']['completion_basis']=='qwen_visual_assessment'
    finally:
        loop.stop();stop.set();ticker.join(2);sim.close()


@pytest.mark.parametrize('action',[a for a in qb.ACTIONS if a[0] in 'xyz'])
def test_all_thirty_selected_signed_magnitudes_map_exactly(action):
    sim,brain,jev,loop=make();jev.action=action;before=list(sim.state['ee'])
    loop.submit('MOCK jog');state=settle(loop)
    assert state['state']=='completed'
    axis,sign,size=action.split('_');expected=list(before)
    expected['xyz'.index(axis)]+=float(size[:-2])/1000*(1 if sign=='plus' else -1)
    assert sim.commands==[('xyz',expected)]


@pytest.mark.parametrize('stage',['qwen','jev'])
@pytest.mark.parametrize('field,value',[('paused',False),('gripper_openness',.5)])
def test_delayed_decision_rejects_mechanical_change(stage,field,value):
    sim,brain,jev,loop=make()
    (brain if stage=='qwen' else jev).hook=lambda:sim.state.update({field:value})
    loop.submit('MOCK changed robot');state=settle(loop)
    assert state['outcome']['reason']=='robot_changed_during_decision' and not sim.commands


@pytest.mark.parametrize('field',['epoch','actions'])
def test_external_midmotion_change_cannot_emit_motion_completed(field):
    sim,brain,jev,loop=make()
    def tick(dt):sim.external_tick(dt);sim.state[field]+=1
    loop.sleep=tick;loop.submit('MOCK external action');state=settle(loop)
    assert state['outcome']['reason']=='robot_changed_during_decision'
    assert not any(e['detail']['phase']=='motion_completed' for e in state['events'])


def test_last_qwen_assessment_allowed_after_jev_budget_exhausts():
    sim,brain,jev,loop=make(max_calls=1)
    jev.status=lambda:{'available':not jev.requests}
    loop.submit('MOCK final assessment');state=settle(loop)
    assert state['state']=='completed' and len(brain.requests)==2 and len(jev.requests)==1


@pytest.mark.parametrize('action,opening',[('gripper_open_10pct',1.),('gripper_close_10pct',0.)])
def test_gripper_outside_hardware_bounds_never_applies(action,opening):
    sim,brain,jev,loop=make();sim.state['gripper_openness']=opening;jev.action=action
    loop.submit('MOCK limit');state=settle(loop)
    assert state['outcome']['reason']=='mechanical_limit' and not sim.commands


@pytest.mark.parametrize('mode',['two_model','qwen_direct'])
def test_real_file_bridge_worker_and_half_mm_physics_with_mock_assessment(tmp_path,monkeypatch,mode):
    import embodied_arm.runtime.visual_loop_ipc as ipc
    from embodied_arm.runtime.visual_loop_worker import worker_run
    from embodied_arm.simulation.so101_simulation import SO101Simulation
    monkeypatch.setattr(ipc,'VISUAL_IPC_ROOT',tmp_path/'visual-ipc')
    brain,jev=Brain(),Jev('x_minus_0.5mm');errors=[]
    if mode=='qwen_direct':
        def forbidden(*args,**kwargs):raise AssertionError('Jev must never be invoked in direct mode')
        jev.decide=forbidden
    def worker_run_safe():
        try:worker_run(brain,jev,max_calls=4,max_seconds=12,mode=mode)
        except BaseException as exc:errors.append(repr(exc))
    worker=threading.Thread(target=worker_run_safe);worker.start()
    until=time.monotonic()+2
    while not ipc.worker_status()['available'] and time.monotonic()<until:time.sleep(.01)
    assert ipc.worker_status()['available']
    sim=SO101Simulation(camera_profile='observer_wrist')
    monkeypatch.setattr(sim,'evaluation_state',lambda:(_ for _ in ()).throw(AssertionError('No ground truth')))
    stop=threading.Event()
    def tick():
        while not stop.is_set():sim.tick(4);time.sleep(.004)
    ticker=threading.Thread(target=tick,daemon=True);ticker.start()
    loop=VisualActionLoop(sim,ipc.FileVisualBridge('qwen'),ipc.FileVisualBridge('jev'),mode=mode,max_calls=2,max_seconds=15)
    try:
        initial=sim.proprioception()['ee'];loop.submit('MOCK half millimetre jog')
        state=settle(loop,17);final=sim.proprioception()['ee']
        assert state['state']=='completed',state
        assert state['controller_actions']==1 and state['qwen_calls']==2
        assert state['decision_calls']==1 and state['controller_calls']==(0 if mode=='qwen_direct' else 1)
        assert state['outcome']['completion_basis']=='qwen_visual_assessment'
        displacement=initial[0]-final[0]
        print('MOCK_VISUAL_IPC_PHYSICS',{'command_m':.0005,'measured_fk_x_m':displacement,'state':state['state']},flush=True)
        assert .0002 < displacement < .0008  # Servo response tolerance, not visual precision.
    finally:
        loop.stop();stop.set();ticker.join(2);sim.close();worker.join(14)
    assert not worker.is_alive() and not errors


class ForbiddenJev:
    def status(self):raise AssertionError('Direct readiness must not inspect Jev')
    def decide(self,*args,**kwargs):raise AssertionError('Direct mode must not invoke Jev')


def test_default_direct_mode_needs_only_qwen_and_records_no_jev_calls():
    sim=Sim();brain=Brain();loop=VisualActionLoop(sim,brain,sleep=sim.external_tick)
    assert loop.status()['execution_available'] and loop.status()['control_mode']=='qwen_direct'
    loop.jev=ForbiddenJev();loop.submit('MOCK direct command');state=settle(loop)
    assert state['state']=='completed' and state['controller_calls']==0
    assert state['decision_calls']==1 and state['controller_actions']==1
    assert sim.commands[0][1][0]==pytest.approx(.1995)


def test_direct_budget_hold_does_not_complete_and_never_calls_jev():
    sim=Sim();brain=Brain(100);brain.action='hold'
    loop=VisualActionLoop(sim,brain,ForbiddenJev(),max_calls=2,sleep=sim.external_tick)
    loop.submit('MOCK hold');state=settle(loop)
    assert state['outcome']['reason']=='call_budget' and state['decision_calls']==2
    assert state['controller_calls']==0 and state['controller_actions']==0


@pytest.mark.parametrize('action',['x_plus_100mm','invented',None])
def test_direct_invalid_model_action_cannot_move(action):
    sim=Sim();brain=Brain();brain.action=action
    loop=VisualActionLoop(sim,brain,sleep=sim.external_tick)
    loop.submit('MOCK invalid');state=settle(loop)
    assert state['state']=='failed' and not sim.commands and state['controller_calls']==0


def test_direct_stop_during_qwen_suppresses_late_action():
    sim=Sim();brain=Brain();entered=threading.Event();release=threading.Event()
    def blocked():entered.set();assert release.wait(3)
    brain.hook=blocked;loop=VisualActionLoop(sim,brain,sleep=sim.external_tick)
    try:
        loop.submit('MOCK stop');assert entered.wait(2);loop.stop()
        with pytest.raises(ValueError,match='task_busy'):loop.submit('MOCK too early')
    finally:release.set()
    assert settle(loop)['state']=='cancelled' and not sim.commands
