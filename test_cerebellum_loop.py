"""Offline closed-loop plumbing tests. Mock answers are not model capability evidence."""
import json
import math
import pytest
import httpx
from adapters import ACTIONS, JevAdapter
from runner import JevRunner
from control import observation
from test_runner import FakeSim, wait_done

@pytest.mark.parametrize('axis,sign',[(0,1),(0,-1),(1,1),(1,-1),(2,1),(2,-1)])
def test_mock_local_target_loop(axis,sign,monkeypatch):
    sim=FakeSim()
    target=sim.position[:];target[axis]+=.04*sign
    seen=[]
    def local_observation(backend):
        state=observation(backend)
        state["objective"]={"id":"hover_reach","target_xyz":target,"tolerance_m":.01}
        state["workspace"]=[[.25,.7],[-.35,.35],[.022,.6]]
        return state
    monkeypatch.setattr("runner.observation",local_observation)
    def respond(request):
        body=json.loads(request.content)
        current=body['state']['current_xyz_m']
        assert body['state']['target_xyz_m']==[round(v,4) for v in target]
        assert 'high hover' not in json.dumps(body)
        assert 'Do not descend' not in json.dumps(body)
        seen.append(current[:])
        # This deterministic oracle exists ONLY in this offline mock test.
        action='hold' if abs(target[axis]-current[axis])<.01 else 'xyz'[axis]+('_plus' if sign>0 else '_minus')
        return httpx.Response(200,json={'answers':{'action':{'type':'choice','choice':action,'confidence':.99,'probabilities':{a:float(a==action) for a in ACTIONS}}}})
    adapter=JevAdapter(api_key='offline-test-only',transport=httpx.MockTransport(respond))
    runner=JevRunner(sim,adapter);runner.start(max_calls=3,max_seconds=2);wait_done(runner)
    assert len(seen)==3
    assert math.dist(sim.position,target)<1e-9
    assert len(sim.moves)==2
    assert runner.status()['reason']=='model_hold'
    assert runner.status()['actions_applied']==2
    assert sim.paused
    assert seen[0]!=seen[1]!=seen[2] # Every decision saw a fresh post-action observation.
