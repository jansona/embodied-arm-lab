"""Real servo setpoints ramp over 360 physics ticks, without changing friction."""
import pytest
from embodied_arm.simulation.so101_simulation import SO101Simulation


def test_gripper_open_close_smoothstep_duration_and_limits(monkeypatch):
    sim=SO101Simulation()
    try:
        original=sim.robot.command_gripper;commands=[]
        def record(value):commands.append(value);return original(value)
        monkeypatch.setattr(sim.robot,'command_gripper',record)
        sim.resume();sim.manual('close',origin='offline_fixture')
        assert sim.queue[0]['ticks']==360
        sim.tick(1)
        assert sim.proprioception()['busy'] and commands[-1]>.99
        sim.tick(178)
        assert sim.proprioception()['busy'] and .49<commands[-1]<.51
        sim.tick(180)
        assert sim.proprioception()['busy'] and commands[-1]>0
        sim.tick(1)
        assert not sim.proprioception()['busy'] and commands[-1]==pytest.approx(0)
        assert all(0<=v<=1 for v in commands)
        assert all(b<=a for a,b in zip(commands,commands[1:]))
        commands.clear();sim.manual('open',origin='offline_fixture');sim.tick(360)
        assert commands[-1]==pytest.approx(1) and all(0<=v<=1 for v in commands)
        assert all(b>=a for a,b in zip(commands,commands[1:]))
    finally:sim.close()
