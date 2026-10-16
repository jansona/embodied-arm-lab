import copy
import pytest
from tests.test_so101_control import SafeFake,Vision
from embodied_arm.control.so101_control import CameraSceneSource
from embodied_arm.perception.so101_visual_feedback import verify_visual_lift

def observations():
    sim=SafeFake();source=CameraSceneSource(sim,Vision(sim),clock=lambda:sim.now+.1)
    first=source.refresh();last=copy.deepcopy(first)
    last['scene']['captured_at']+=1
    last['scene']['estimated_at']+=1
    last['scene']['frame_ids']={'left':'later','right':'later'}
    last['robot']['actions']+=1
    first['robot']['ee']=[.19,-.05,.013]
    last['robot']['ee']=[.19,-.05,.063]
    last['scene']['objects']['red']['xyz'][2]+=.05
    return first,last,last['scene']['captured_at']+.1

def test_rgb_lift_evidence_no_contact_claim():
    a,b,now=observations();result=verify_visual_lift(a,b,'red',now=now)
    assert result['verified'] and not result['physical_contact_verified']
    assert result['source']=='temporal_rgb'

def test_robot_rise_without_cube_rise_is_not_grasp():
    a,b,now=observations();b['scene']['objects']['red']['xyz']=a['scene']['objects']['red']['xyz']
    assert not verify_visual_lift(a,b,'red',now=now)['verified']

def test_independent_cube_movement_is_not_grasp():
    a,b,now=observations();b['scene']['objects']['red']['xyz'][0]+=.04
    assert not verify_visual_lift(a,b,'red',now=now)['verified']

@pytest.mark.parametrize('mutation',[lambda a,b:b['scene'].update(captured_at=a['scene']['captured_at']),lambda a,b:b['robot'].update(epoch=2),lambda a,b:b.update(source='ground_truth')])
def test_stale_reset_or_oracle_rejected(mutation):
    a,b,now=observations();mutation(a,b)
    with pytest.raises(ValueError):verify_visual_lift(a,b,'red',now=now)

def test_latest_frame_must_be_fresh():
    a,b,now=observations()
    with pytest.raises(ValueError):verify_visual_lift(a,b,'red',now=now+31)
