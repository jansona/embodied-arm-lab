import math
import pytest
import pybullet as p
from simulator import Simulation
from control import observation,apply_action

@pytest.fixture
def sim():
    s=Simulation();yield s;s.close()

def finish(s):
    for _ in range(2000):
        s.tick(4)
        if not s.current and not s.queue:return
    pytest.fail('Motion exceeded fixed step budget')

def test_physical_baseline_repeated(sim):
    results=[]
    for _ in range(3):
        sim.reset();sim.baseline();finish(sim);s=sim.state()
        assert s['error'] is None
        assert s['metrics']['lifted'] and s['metrics']['placed']
        assert s['metrics']['physics_steps']==2928
        assert sim.call(p.getNumConstraints)==1
        linkage=sim.call(p.getConstraintInfo,sim.finger_constraint)
        assert linkage[:5]==(sim.robot,9,sim.robot,10,p.JOINT_GEAR)
        results.append(s['blocks']['red'])
    assert results[0]==results[1]==results[2]

@pytest.mark.parametrize('xyz',[[9,0,.3],[.5,0,float('nan')],[.5,0,-.1],[.5],None])
def test_reject_invalid_motion(sim,xyz):
    with pytest.raises((ValueError,TypeError)):sim.manual('xyz',xyz)
    assert not sim.queue

def test_manual_limits(sim):
    with pytest.raises(ValueError):sim.manual('xyz',[.65,0,.3])
    with pytest.raises(ValueError):sim.manual('joints',[99]*7)
    q=sim.state()['joints'];q[0]+=.1;sim.manual('joints',q);finish(sim)
    assert abs(sim.state()['joints'][0]-q[0])<.02

def test_pause_stop_reset(sim):
    sim.baseline();sim.tick(50);sim.paused=True
    before=sim.state();sim.tick(100)
    assert sim.state()['metrics']['physics_steps']==before['metrics']['physics_steps']
    sim.paused=False;sim.tick(50);assert sim.steps>before['metrics']['physics_steps']
    sim.stop();assert not sim.queue and sim.current is None and sim.paused
    with pytest.raises(ValueError):sim.manual('open')
    sim.reset();assert sim.steps==0 and not sim.paused

def test_busy_and_bounded_adapter(sim):
    o=observation(sim);assert o['simulation_only'] and o['robot']['arm_dof']==7
    apply_action(sim,'x_plus');assert sim.queue[-1]['origin']=='jev'
    with pytest.raises(ValueError):apply_action(sim,'x_plus')
    finish(sim)
    with pytest.raises(ValueError):apply_action(sim,'teleport')

def test_render(sim):
    img=sim.render();assert img[:2]==b'\xff\xd8' and len(img)>10000

@pytest.mark.parametrize('seed',[1,2,3,42])
def test_seeded_physical_baseline(sim,seed):
    sim.reset(seed);sim.baseline();finish(sim)
    assert sim.state()['metrics']['placed']
    assert sim.state()['metrics']['lifted']

@pytest.mark.parametrize('action,axis,sign',[(a+'_'+s,i,n) for i,a in enumerate('xyz') for s,n in [('plus',1),('minus',-1)]])
def test_all_axis_directions_in_world_coordinates(sim,action,axis,sign):
    before=sim.state()['ee'];apply_action(sim,action);finish(sim);after=sim.state()['ee']
    delta=[b-a for a,b in zip(before,after)]
    assert abs(delta[axis]-.02*sign)<.001
    assert all(abs(v)<.001 for i,v in enumerate(delta) if i!=axis)
