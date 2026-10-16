"""Read-only RGB failure diagnostics for frozen resolution experiment outputs.

Passively observe original helper returns/exceptions. No thresholds, outputs,
source files, ground-truth inputs, or control paths are modified.
"""
from pathlib import Path
import base64,json,sys,hashlib,collections
import numpy as np
import rgb_face_refinement as refinement

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/tmp/so101-resolution-experiment')
REPORT=ROOT/'artifacts/so101-resolution-precision-evaluation.json'


def main():
    original_hash=hashlib.sha256(Path(refinement.__file__).read_bytes()).hexdigest()
    predictions={};inputs={}
    for width,height in ((640,480),(1280,960),(1920,1440)):
        tag=f'{width}x{height}'
        predictions[(width,height)]=json.loads((WORK/f'predictions-{tag}.json').read_text())
        inputs[(width,height)]={(r['seed'],r['view']):r for r in json.loads((WORK/f'rgb-inputs-{tag}.json').read_text())}
    anchors=json.loads((ROOT/'artifacts/so101-vision-third-diagnostic.json').read_text())['detections']
    anchors={(d['camera_id'],d['object_id']):d for d in anchors}
    failed=[(r['seed'],r['view'],e['camera_id'],e['object_id']) for r in predictions[(1920,1440)] for e in r['evidence'] if not e['accepted']]
    diagnostics=[]
    for seed,view,camera,obj in failed:
        matched=[]
        for width,height in ((640,480),(1280,960),(1920,1440)):
            entry=inputs[(width,height)][(seed,view)]
            image=next(im for im in entry['rgb_packet']['images'] if im['camera_id']==camera)
            raw=base64.b64decode(image['data_url'].split(',',1)[1]);anchor=anchors[(camera,obj)];observed={'components':[],'quad_exceptions':[]}
            def local_trace(frame,event,arg):
                name=frame.f_code.co_name
                if name=='_components' and event=='return' and isinstance(arg,list):
                    observed['components'].append([{'pixels':len(a),'bbox_min':a.min(axis=0).tolist(),'bbox_max':a.max(axis=0).tolist()} for a in arg])
                elif name=='_quadrilateral' and event=='exception' and isinstance(arg[1],ValueError):
                    local=frame.f_locals;detail={'reason':str(arg[1])}
                    for key in ('residual','original_area','support_counts','i'):
                        if key in local:detail[key]=local[key]
                    if all(key in local for key in ('contour','a')) and isinstance(local['a'],np.ndarray) and local['a'].shape==(2,):
                        detail['nearest_observed_boundary_distance_px']=float(np.min(np.linalg.norm(local['contour']-local['a'],axis=1)))
                    observed['quad_exceptions'].append(detail)
                return local_trace
            def global_trace(frame,event,arg):
                if event=='call' and frame.f_code.co_filename==refinement.__file__ and frame.f_code.co_name in ('_components','_quadrilateral'):
                    frame.f_trace_lines=False;frame.f_trace_opcodes=False
                    return local_trace
                return None
            sys.settrace(global_trace)
            try:
                result=refinement.refine_rgb_landmark(raw,object_id=obj,uv_1000=[(anchor['uv_1000'][0]*(width-1)/1000)/(width-1)*1000, (anchor['uv_1000'][1]*(height-1)/1000)/(height-1)*1000],confidence=anchor['confidence'],width=width,height=height)
            finally:sys.settrace(None)
            frozen=next(e for row in predictions[(width,height)] if row['seed']==seed and row['view']==view for e in row['evidence'] if e['camera_id']==camera and e['object_id']==obj)
            assert result.accepted==frozen['accepted'] and result.reason==frozen['reason'], (seed,view,camera,obj,width,result.accepted,result.reason,frozen['accepted'],frozen['reason'])
            matched.append({'resolution':[width,height],'accepted':result.accepted,'reason':result.reason,**observed})
        diagnostics.append({'seed':seed,'view':view,'camera_id':camera,'object_id':obj,'matched_resolution_observations':matched})
    assert hashlib.sha256(Path(refinement.__file__).read_bytes()).hexdigest()==original_hash
    report={'schema':'so101-resolution-rgb-failure-diagnosis-v1','not_for_model_input':True,'ground_truth_read':False,
            'method':'Passive Python tracing of original _components returns and _quadrilateral exceptions; outputs checked against frozen predictions, no threshold or source changes.',
            'refiner_sha256':original_hash,'cases':diagnostics}
    (ROOT/'artifacts/so101-resolution-rgb-failure-diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for reason in ['ambiguous_face_patches','unsupported_corner','irregular_face_boundary']:
        selected=[r for r in diagnostics if r['matched_resolution_observations'][-1]['reason']==reason]
        print('\nREASON',reason,'count',len(selected))
        for case in selected[:3]:print(json.dumps(case))
if __name__=='__main__':main()
