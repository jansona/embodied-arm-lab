from types import SimpleNamespace
import pytest
from rgb_roi_geometry import bounds_from_detections,frustum

def test_crop_uses_only_named_image_landmarks():
    images=[{'camera_id':c,'width':1920,'height':1440} for c in ('left','right')]
    points=[SimpleNamespace(camera_id=c,object_id=o,uv=(900,700),visibility='visible') for c in ('left','right') for o in ('red','blue')]
    bounds=bounds_from_detections(images,points,('red','blue'))
    assert bounds['left']==[760,560,1040,840]
    assert bounds['left']==bounds['right']
    with pytest.raises(ValueError):bounds_from_detections(images,points[:-1],('red','blue'))

def test_native_frustum_and_invalid_crop():
    l,r,b,t,n,f=frustum([0,0,1920,1440],1920,1440)
    assert l==-r and b==-t and r/t==pytest.approx(4/3)
    with pytest.raises(ValueError):frustum([-1,0,20,20],1920,1440)
