import base64
from io import BytesIO
from PIL import Image
import pytest
from so101_simulation import SO101Simulation

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

def test_roi_capture_is_explicit_and_preserves_native_coordinates():
    sim=SO101Simulation(camera_size=(1920,1440))
    try:
        bounds={'left':[824,528,1184,992],'right':[752,544,1088,1072]}
        packet=sim.capture_frames(roi_bounds=bounds)
        assert packet['capture_profile']=='native_coordinate_roi_canvas'
        assert packet['observed_roi_bounds']==bounds
        for image,cal in zip(packet['images'],packet['calibrations']):
            assert image['width']==1920 and cal['cx']==959.5
            rgb=Image.open(BytesIO(base64.b64decode(image['data_url'].split(',')[1])))
            assert rgb.getpixel((0,0))==(0,0,0)
        assert not sim._renders # Cropped observations never replace full preview cache.
        with pytest.raises(ValueError):sim.capture_frames(roi_bounds={'left':bounds['left']})
    finally:sim.close()
