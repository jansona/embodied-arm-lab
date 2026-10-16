"""Genuine encoder-mounted RGB cameras; synthetic geometry is evaluation only."""
import base64
from copy import deepcopy
from io import BytesIO
import time
import numpy as np
from PIL import Image
import pybullet as p
import pytest
from embodied_arm.simulation.so101_simulation import SO101Simulation
from embodied_arm.simulation.camera_rig import TCP_FROM_WRIST_CAMERA
from embodied_arm.perception.camera_perception import CameraCalibration
from embodied_arm.perception.moving_camera_geometry import validate_capture_geometry


def test_rigid_mount_tracks_encoder_fk_and_observer_is_fixed():
    sim=SO101Simulation(camera_profile='observer_wrist')
    try:
        first=sim.capture_frames();validate_capture_geometry(first)
        sim.resume();sim.manual('xyz',[.20,0,.07]);sim.tick(264);sim.pause_idle()
        second=sim.capture_frames();validate_capture_geometry(second)
        assert first['calibrations'][0]==second['calibrations'][0]
        assert first['calibrations'][1]!=second['calibrations'][1]
        expected=sim.robot.forward_kinematics(sim.robot.joint_positions())@TCP_FROM_WRIST_CAMERA
        assert np.allclose(second['calibrations'][1]['world_from_camera'],expected,atol=1e-12)
        assert first['frame_id']!=second['frame_id']
        assert first['captured_at']<second['captured_at']
        assert {m['role'] for m in second['view_metadata']}=={'observer','wrist'}
        for mutate in ('old_calibration','old_frame','old_time'):
            bad=deepcopy(second)
            if mutate=='old_calibration':bad['calibrations'][1]=first['calibrations'][1]
            elif mutate=='old_frame':bad['view_metadata'][1]['frame_id']=first['frame_id']
            else:bad['view_metadata'][1]['captured_at']=first['captured_at']
            with pytest.raises(ValueError,match='Stale or mismatched'):validate_capture_geometry(bad)
    finally:sim.close()


def test_render_projection_consistent_for_synthetic_calibration_marker():
    sim=SO101Simulation(camera_profile='observer_wrist',illumination_profile='shadow_free_rgb_v1')
    try:
        # Known test target only. Never passed into a controller/detector.
        xyz=(.22,-.06,.04)
        shape=sim.call(p.createVisualShape,p.GEOM_SPHERE,radius=.003,rgbaColor=[1,0,1,1])
        sim.call(p.createMultiBody,baseMass=0,baseVisualShapeIndex=shape,basePosition=xyz)
        packet=sim.capture_frames()
        for image,raw in zip(packet['images'],packet['calibrations']):
            rgb=np.asarray(Image.open(BytesIO(base64.b64decode(image['data_url'].split(',')[1])))).astype(float)
            mask=(rgb[:,:,0]>110)&(rgb[:,:,2]>100)&(rgb[:,:,1]<.6*np.minimum(rgb[:,:,0],rgb[:,:,2]))
            ys,xs=np.nonzero(mask)
            assert len(xs)>5
            projected=CameraCalibration.from_dict(raw).project(xyz)
            assert np.linalg.norm(np.array([xs.mean(),ys.mean()])-projected)<1.8
    finally:sim.close()


def test_real_rgb_static_world_tracking_after_wrist_motion_without_oracle(monkeypatch):
    from embodied_arm.control.so101_control import CameraSceneSource
    from embodied_arm.perception.phase_observer import PhaseSceneObserver
    from tests.rgb_coarse_fixture import RGBColourFixture
    sim=SO101Simulation(camera_profile='observer_wrist',camera_size=(1920,1440),illumination_profile='shadow_free_rgb_v1')
    try:
        monkeypatch.setattr(sim,'evaluation_state',lambda:(_ for _ in ()).throw(AssertionError('Oracle forbidden')))
        # This is a geometry/association test, not a wall-clock throughput test.
        # Simulated sensor time advances only on capture; aggregate CPU contention
        # must not masquerade as optical failure. Production clocks/gates stay real.
        sensor_time=[100.]
        capture_rgb=sim.capture_frames
        def deterministic_capture(**kwargs):
            packet=capture_rgb(**kwargs)
            sensor_time[0]+=.1
            packet['captured_at']=sensor_time[0]
            for view in packet['view_metadata']:view['captured_at']=sensor_time[0]
            return packet
        monkeypatch.setattr(sim,'capture_frames',deterministic_capture)
        clock=lambda:sensor_time[0]+.01
        # Coarse image preparation has its own injectable freshness clock.
        import embodied_arm.perception.rgb_coarse_packet as coarse
        def with_sensor_clock(fn):
            def checked(*args,**kwargs):
                kwargs.setdefault('now',clock())
                return fn(*args,**kwargs)
            return checked
        monkeypatch.setattr(coarse,'prepare_coarse_packet',with_sensor_clock(coarse.prepare_coarse_packet))
        monkeypatch.setattr(coarse,'remap_result',with_sensor_clock(coarse.remap_result))
        source=CameraSceneSource(sim,vision=RGBColourFixture(profile='scaled'),refine_rgb=True,refinement_profile='scaled',roi_tracking=True,clock=clock)
        source.refresh();observer=PhaseSceneObserver(source,clock=clock)
        initial=observer.begin_step('pick','red')
        before={f.camera_id:f.calibration for f in observer.frames}
        sim.resume();sim.manual('xyz',[.20,0,.07]);sim.tick(264);sim.pause_idle()
        current=observer.observe()
        assert observer.frames[0].calibration==before['left']
        assert observer.frames[1].calibration!=before['right']
        for obj in ('red','blue'):
            a,b=initial['scene']['objects'][obj],current['scene']['objects'][obj]
            assert b['status']=='estimated' and b['position_std_m']<=.003
            assert np.linalg.norm(np.array(a['xyz'])-b['xyz'])<.003
        assert current['scene']['frame_ids']!=initial['scene']['frame_ids']
        assert observer.fixed_goal()['source']=='prior_rgb_measurement'
        assert observer.fixed_goal()['captured_at']<current['scene']['captured_at']
        with pytest.raises(ValueError,match='Verified held'):observer.begin_step('place','red','goal',{})
        certificate={'object_id':'red','epoch':current['robot']['epoch'],'actions':current['robot']['actions'],
            'frame_ids':current['scene']['frame_ids'],'identity_token':current['visual_evidence']['identity_token'],
            'verified_lift':True,'source':'phase_scoped_temporal_rgb'}
        # Synthetic certificate tests handoff plumbing only, not physical lift success.
        placed=observer.begin_step('place','red','goal',certificate)
        assert placed['scene']['frame_ids']!=current['scene']['frame_ids']
        assert observer.goal_memory is not None and observer.dynamic is not None
        assert observer.export_step_evidence()[0]['source']=='rgb_only'
        from embodied_arm.perception.camera_perception import EstimatedScene
        with pytest.raises(ValueError,match='stale or future'):
            EstimatedScene.from_dict(placed['scene']).require('red',now=clock()+4.)
        original_capture=sim.capture_frames
        def stale_calibration(**kwargs):
            packet=original_capture(**kwargs)
            packet['calibrations'][1]=before['right'].to_dict()
            packet['view_metadata'][1]['calibration']=deepcopy(packet['calibrations'][1])
            return packet
        monkeypatch.setattr(sim,'capture_frames',stale_calibration)
        with pytest.raises(ValueError,match='paused encoders'):source.track()
    finally:sim.close()


def test_preview_and_perception_use_same_rig_not_hidden_stereo():
    sim=SO101Simulation(camera_profile='observer_wrist',camera_size=(1920,1440))
    try:
        main=sim.capture_frames();preview=sim.capture_preview_frames()
        for a,b in zip(main['calibrations'],preview['calibrations']):
            assert a['world_from_camera']==b['world_from_camera']
            assert a['fx']==pytest.approx(3*b['fx'])
        assert len(main['images'])==len(preview['images'])==2
    finally:sim.close()
