from types import SimpleNamespace
from tests.test_so101_control import SafeFake,Vision
from embodied_arm.control.so101_control import CameraSceneSource
from embodied_arm.control.so101_executor import VisualWaypointExecutor

class Sim(SafeFake):
    def __init__(self):super().__init__();self.ee=[.198,-.05,.06];self.commands=[]
    def proprioception(self):return {**super().proprioception(),'ee':list(self.ee)}
    def resume(self):self.paused=False
    def pause_idle(self):self.paused=True
    def stop(self):self.paused=True;self.epoch+=1
    def manual(self,kind,xyz,*,origin):assert origin=='jev';self.commands.append((kind,xyz));self.ee=list(xyz);self.actions+=1
class Controller:
    def __init__(self,sim,action='x_minus',accepted=True,mutate=False):self.sim=sim;self.action=action;self.accepted=accepted;self.mutate=mutate;self.inputs=[]
    def decide(self,request):
        self.inputs.append(request)
        if self.mutate:self.sim.epoch+=1
        return SimpleNamespace(action=self.action,accepted=self.accepted)

def run(action='x_minus',accepted=True,mutate=False,max_calls=3):
    sim=Sim();inner=CameraSceneSource(sim,Vision(sim))
    class Source:
        def __init__(self):self.semantic_calls=0;self.tracking_frames=0
        def refresh(self,**kwargs):
            self.semantic_calls+=1
            assert self.semantic_calls==1
            return inner.refresh()
        def track(self):self.tracking_frames+=1;return inner.refresh() # Explicit image fixture, no API.
        def snapshot(self):return inner.snapshot()
        def cancel(self):return inner.cancel()
    source=Source();controller=Controller(sim,action,accepted,mutate)
    runner=VisualWaypointExecutor(sim,source,controller)
    result=runner.run('red',[0,0,.04],max_calls=max_calls)
    return sim,controller,result

def test_mock_camera_to_jev_to_bounded_actuator():
    sim,controller,result=run()
    assert result['status']=='stopped' and result['reason']=='uncertainty_unresolved' and result['calls']==1 and result['actions']==1
    assert len(sim.commands)==1 and .19<sim.commands[0][1][0]<.198
    assert sim.commands[0][1][1:]==[-.05,.06]
    assert 'ground_truth' not in repr(controller.inputs) and 'SECRET_ORACLE' not in repr(controller.inputs)
    assert 'tactile' not in repr(controller.inputs)
    assert sim.paused

def test_rejected_model_does_not_move():
    sim,_,result=run(accepted=False)
    assert not sim.commands and result['reason']=='controller_rejected'

def test_changed_scene_rejects_pending_action():
    sim,_,result=run(mutate=True)
    assert not sim.commands and result['reason']=='robot_changed_during_decision'

def test_wrong_direction_is_rejected_without_replacement():
    sim,_,result=run(action='x_plus')
    assert not sim.commands and result['reason']=='controller_rejected'

def test_hold_budget_stops_without_fallback():
    sim,_,result=run(action='hold',max_calls=2)
    assert not sim.commands and result['calls']==2 and result['reason']=='call_budget_exhausted'

def test_obstacle_clearance_and_table_guards():
    import pytest
    with pytest.raises(ValueError,match='unsafe_hover_path'):
        VisualWaypointExecutor._clearance([.2,0,.06],[.19,-.05,.055],[.19,-.03,.01])
    with pytest.raises(ValueError,match='unsafe_hover_path'):
        VisualWaypointExecutor._clearance([.2,0,.02],[.19,-.05,.055],[.3,.15,.01])
    VisualWaypointExecutor._clearance([.2,0,.06],[.19,-.05,.055],[.25,.055,.01])
