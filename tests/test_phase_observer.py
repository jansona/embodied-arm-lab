"""Rendered-RGB integration; detector is an explicit offline color fixture."""
import pytest
from so101_simulation import SO101Simulation
from so101_control import CameraSceneSource
from embodied_arm.phase_observer import PhaseSceneObserver
from rgb_coarse_fixture import RGBColourFixture

def test_actual_rgb_phase_initialization_never_reads_oracle(monkeypatch):
    sim=SO101Simulation(seed=0)
    try:
        def denied(*args,**kwargs):raise AssertionError('Oracle access forbidden')
        monkeypatch.setattr(sim,'evaluation_state',denied)
        source=CameraSceneSource(sim,vision=RGBColourFixture(),refine_rgb=True)
        source.refresh();observer=PhaseSceneObserver(source)
        first=observer.begin_phase('approach','red');second=observer.observe()
        assert first['scene']['frame_ids']!=second['scene']['frame_ids']
        assert first['visual_evidence']['identity_token']==second['visual_evidence']['identity_token']
        assert second['robot']['actions']==0
        assert len(second['visual_evidence']['detections'])==4
        assert not observer.validated # physical manipulation evidence still pending
    finally:sim.close()

def test_shadow_free_rgb_v1_high_resolution_roi_meets_precision_floor(monkeypatch):
    from so101_uncertainty import axis_bands
    sim=SO101Simulation(seed=0,camera_size=(1920,1440),illumination_profile='shadow_free_rgb_v1')
    try:
        monkeypatch.setattr(sim,'evaluation_state',lambda:(_ for _ in ()).throw(AssertionError('Oracle forbidden')))
        source=CameraSceneSource(sim,vision=RGBColourFixture(),refine_rgb=True,refinement_profile='scaled',roi_tracking=True)
        source.refresh();observer=PhaseSceneObserver(source)
        current=observer.begin_phase('approach','red')
        for name in ('red','blue'):
            covariance=current['scene']['objects'][name]['covariance_m2']
            assert all(x['status']=='inside' for x in axis_bands([0,0,0],covariance))
        assert current['robot']['actions']==0
    finally:sim.close()
