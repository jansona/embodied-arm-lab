"""Robot-only bounds; no semantic image or grasp claims."""
import pytest
from embodied_arm.simulation.so101_simulation import SO101Simulation


def test_incremental_gripper_target_is_applied_without_full_close():
    sim=SO101Simulation()
    try:
        before=sim.proprioception()['gripper_openness']
        target=before-.1
        sim.resume();sim.manual('gripper',target,origin='jev')
        assert sim.queue[-1]['gripper']==pytest.approx(target)
        assert sim.actions==1
        sim.tick(400)
        after=sim.proprioception()['gripper_openness']
        assert after<before and abs(after-target)<.03
    finally:sim.close()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,-.1,1.1,0.])
def test_invalid_gripper_increment_never_queues(value):
    sim=SO101Simulation()
    try:
        sim.resume()
        with pytest.raises(ValueError):sim.manual('gripper',value,origin='jev')
        assert sim.actions==0 and not sim.queue
    finally:sim.close()
