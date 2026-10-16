import time
import threading
import pytest
from embodied_arm.simulation.simulator import Simulation
from embodied_arm.legacy.panda.task_executor import TaskExecutor, validate_plan, plan_from_intent
from embodied_arm.providers.adapters import Decision

@pytest.mark.parametrize('bad',[{}, {'steps':[]},{'steps':[{'action':'exec','code':'bad'}]}, {'steps':[{'action':'move','target_xyz':[.4,0,float('nan')]}]}, {'steps':[{'action':'lift','height_m':1}]}])
def test_invalid_plans(bad):
    with pytest.raises(ValueError):validate_plan(bad)

@pytest.mark.parametrize('seed',[0,1])
def test_physical_pick(seed):
    sim=Simulation();sim.reset(seed)
    try:
        ex=TaskExecutor(sim)
        ex.start(plan_from_intent({'action':'pick','object_id':'red'},sim.state()))
        limit=time.monotonic()+20
        while ex.status()['status']=='running' and time.monotonic()<limit:
            sim.tick(96);time.sleep(.0005)
        assert ex.status()['status']=='completed',ex.status()
        assert sim.state()['blocks']['red'][2]>.15,sim.state()['blocks']
        assert ex.status()['controller']=='deterministic servo'
        assert ex.status()['actions']<100
    finally:sim.close()

def test_pending_controller_cancel_never_actuates():
    entered=threading.Event();release=threading.Event()
    class Pending:
        def decide(self,state):
            entered.set();release.wait(2)
            return Decision('x_plus',.99,{},True,'accepted','x_plus')
    sim=Simulation()
    try:
        ex=TaskExecutor(sim,Pending());now=sim.state()['ee'];target=[now[0]+.04,*now[1:]]
        ex.start({'steps':[{'action':'move','target_xyz':target}]})
        assert entered.wait(1);ex.stop();release.set();ex.thread.join(2)
        assert ex.status()['actions']==0
        assert sim.state()['paused']
    finally:sim.close()

def test_physical_pick_and_place():
    sim=Simulation()
    try:
        ex=TaskExecutor(sim)
        ex.start(plan_from_intent({'action':'pick','object_id':'red','target_id':'goal'},sim.state()))
        limit=time.monotonic()+30
        while ex.status()['status']=='running' and time.monotonic()<limit:
            sim.tick(96);time.sleep(.0005)
        assert ex.status()['status']=='completed',ex.status()
        assert sim.state()['metrics']['placed'],sim.state()['blocks']
    finally:sim.close()

def test_controller_result_after_deadline_cannot_move():
    class Slow:
        def decide(self,state):
            time.sleep(.05)
            return Decision('x_plus',.99,{},True,'accepted','x_plus')
    sim=Simulation()
    try:
        ex=TaskExecutor(sim,Slow());now=sim.state()['ee']
        ex.start({'steps':[{'action':'move','target_xyz':[now[0]+.04,*now[1:]]}]},max_seconds=.01)
        ex.thread.join(1)
        assert ex.status()['actions']==0
        assert ex.status()['status']=='failed'
    finally:sim.close()
