"""Rendered-image integration: no fixture object poses or projected landmarks."""
from rgb_coarse_fixture import RGBColourFixture
from so101_control import CameraSceneSource
from so101_simulation import SO101Simulation


def test_actual_rendered_640_rgb_fixture_full_scene_passes_unchanged_gates():
    sim = SO101Simulation(camera_size=(640, 480))
    try:
        source = CameraSceneSource(sim, RGBColourFixture(), refine_rgb=True)
        snapshot = source.refresh()
        assert snapshot['detector_provenance'] == 'offline_fixture_rgb_geometry'
        for obj in snapshot['scene']['objects'].values():
            assert obj['status'] == 'estimated'
            assert 0 < obj['position_std_m'] <= .003
        # A valid RGB observation is not a claim of manipulation feasibility.
        assert not hasattr(source, 'verified_grasp')
    finally:
        sim.close()


def test_coarse_fixture_rejects_two_compact_same_colour_candidates():
    import base64
    from test_rgb_face_refinement import image_bytes
    raw = image_bytes(duplicate=True)
    packet = {'frame_id':'duplicate1','captured_at':1.,'images':[
        {'camera_id':cam,'width':640,'height':480,
         'data_url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()}
        for cam in ('left','right')]}
    result = RGBColourFixture().perceive(packet)
    assert all(d.visibility != 'visible' for d in result.detections if d.object_id=='red')
