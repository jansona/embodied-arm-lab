"""Offline held-object RGB diagnostics using saved rendered images only."""
import base64
import json
from pathlib import Path
import time
from rgb_coarse_fixture import RGBColourFixture
from rgb_coarse_packet import prepare_coarse_packet, remap_detections
from rgb_scaled_refinement import refine_rgb_landmark


def inspect(name, stage):
    packet={'frame_id':f'{name}-{stage}','captured_at':time.monotonic(),'images':[]}
    for cam in ('left','right'):
        raw=Path(f'/tmp/so101-shifted-{name}-{stage}-{cam}.jpg').read_bytes()
        packet['images'].append({'camera_id':cam,'width':1920,'height':1440,
            'data_url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()})
    coarse=prepare_coarse_packet(packet)
    result=RGBColourFixture().perceive(coarse.packet)
    detections=remap_detections(result.detections,coarse,native_packet=packet)
    rows=[]
    for d in detections:
        if d.object_id!=name:continue
        if d.uv is None:
            rows.append({'camera':d.camera_id,'accepted':False,'reason':'coarse_'+d.visibility});continue
        raw=Path(f'/tmp/so101-shifted-{name}-{stage}-{d.camera_id}.jpg').read_bytes()
        r=refine_rgb_landmark(raw,object_id=name,uv_1000=[d.uv[0]/1919*1000,d.uv[1]/1439*1000],
            confidence=d.confidence,width=1920,height=1440)
        rows.append({'camera':d.camera_id,**r.to_dict()})
    return {'object_id':name,'stage':stage,'views':rows}

if __name__=='__main__':
    for name in ('red','blue'):
        for stage in ('initial','approach','preclose','closed','lifted'):
            print(json.dumps(inspect(name,stage)),flush=True)
