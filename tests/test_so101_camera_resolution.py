import base64
from io import BytesIO
from PIL import Image
import pytest
from embodied_arm.simulation.so101_simulation import SO101Simulation

@pytest.mark.parametrize('size',[(640,480),(1280,960),(1920,1440)])
def test_explicit_native_resolution_matches_rgb_and_calibration(size):
    sim=SO101Simulation(camera_size=size)
    try:
        packet=sim.capture_frames()
        for image,calibration in zip(packet['images'],packet['calibrations']):
            assert (image['width'],image['height'])==size
            assert (calibration['width'],calibration['height'])==size
            assert calibration['cx']==(size[0]-1)/2
            assert Image.open(BytesIO(base64.b64decode(image['data_url'].split(',')[1]))).size==size
        assert 'objects' not in packet and 'depth' not in repr(packet.keys())
    finally:sim.close()

def test_unbounded_resolution_rejected():
    with pytest.raises(ValueError):SO101Simulation(camera_size=(9999,9999))

def test_full_frame_capture_has_no_crop_option_or_black_canvas():
    sim=SO101Simulation(camera_size=(640,480))
    try:
        packet=sim.capture_frames()
        assert 'observed_roi_bounds' not in packet and 'capture_profile' not in packet
        assert set(sim._renders)=={'left','right'}
        with pytest.raises(TypeError):sim.capture_frames(roi_bounds={})
    finally:sim.close()
