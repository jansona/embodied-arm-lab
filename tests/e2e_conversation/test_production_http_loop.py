"""Offline transport-only integration of the production HTTP/IPC/physics path.

No live provider calls. Qwen/Jev HTTP replies are MOCK; application services,
parsers, file bridge, worker, image refinement and SO101 physics are real.
"""
import json
import threading
import time

import httpx
from fastapi.testclient import TestClient
from embodied_arm.providers.brain import QwenBrain, MODEL
from embodied_arm.providers.vision_adapter import QwenVision
from embodied_arm.providers.so101_jev import SO101JevAdapter, ALIYUN_MODEL
from embodied_arm.runtime.so101_ipc import FileSO101Bridge, worker_status
from embodied_arm.runtime.so101_worker import worker_run
from embodied_arm.web.app_so101 import create_app
from tests.rgb_coarse_fixture import RGBColourFixture


def completion(payload):
    return {'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':json.dumps(payload)}}]}


class MockProviderTransport:
    """Only the external model's response is substituted, never app behavior."""
    def __init__(self):self.vision_calls=0;self.brain_calls=0
    def vision(self,request):
        self.vision_calls+=1
        content=json.loads(request.content)['messages'][1]['content']
        meta=json.loads(content[0]['text'])
        urls=[entry['image_url']['url'] for entry in content if entry['type']=='image_url']
        packet={'frame_id':meta['frame_id'],'captured_at':time.monotonic(),
                'images':[dict(item,data_url=url) for item,url in zip(meta['images'],urls)]}
        assert len(urls)==2 and all(url.startswith('data:image/jpeg;base64,') for url in urls)
        coarse=RGBColourFixture(profile='scaled').perceive(packet)
        dimensions={item['camera_id']:(item['width'],item['height']) for item in packet['images']}
        detections=[]
        for detection in coarse.detections:
            value=detection.to_dict();width,height=dimensions[value['camera_id']]
            uv=value.pop('uv');sigma=value.pop('pixel_sigma_px')
            value['uv_1000']=None if uv is None else [uv[0]*1000/(width-1),uv[1]*1000/(height-1)]
            value['sigma_1000']=None if sigma is None else sigma*1000/max(width-1,height-1)
            detections.append(value)
        return httpx.Response(200,json=completion({'frame_id':meta['frame_id'],
            'coordinate_space':'normalized_1000','detections':detections}))
    def brain(self,request):
        self.brain_calls+=1
        return httpx.Response(200,json=completion({'action':'pick','object_id':'red',
            'target_id':None,'message':'MOCK transport: attempt a bounded red-cube pick'}))
    @staticmethod
    def jev(request):
        body=json.loads(request.content);state=body['state']
        choices=list(body['questions']['action']['criteria'])
        if 'signed_error_m' in state:
            error=state['signed_error_m'];tolerance=state['tolerance_m']
            choice='positive' if error>tolerance else 'negative' if error<-tolerance else 'hold'
        else:
            error=state['signed_openness_error'];tolerance=state['tolerance']
            choice='open' if error>tolerance else 'close' if error<-tolerance else 'hold'
        return httpx.Response(200,json={'model':ALIYUN_MODEL,'answers':{'action':{
            'type':'choice','choice':choice,'confidence':.9,
            'probabilities':{key:.9 if key==choice else .05 for key in choices}}}})


def test_http_task_reaches_production_executor_jev_bridge_and_physics(tmp_path,monkeypatch):
    import embodied_arm.runtime.so101_ipc as ipc
    monkeypatch.setattr(ipc,'SO101_IPC_ROOT',tmp_path/'so101-jev-ipc')
    mock=MockProviderTransport()
    adapter=SO101JevAdapter(api_key='offline-placeholder',transport=httpx.MockTransport(mock.jev),max_calls=100)
    done=[]
    worker=threading.Thread(target=lambda:done.append(worker_run(adapter,100,90)),daemon=True)
    worker.start()
    deadline=time.monotonic()+5
    while not worker_status()['available'] and time.monotonic()<deadline:time.sleep(.02)
    assert worker_status()['available']
    brain=QwenBrain(api_key='offline-placeholder',transport=httpx.MockTransport(mock.brain))
    vision=QwenVision(api_key='offline-placeholder',transport=httpx.MockTransport(mock.vision))
    app=create_app(brain=brain,vision=vision,controller=FileSO101Bridge(),enable_simulation_tasks=True)
    try:
        with TestClient(app) as client:
            session=app.state.session
            monkeypatch.setattr(session.sim,'evaluation_state',lambda:(_ for _ in ()).throw(AssertionError('Object oracle forbidden')))
            initial=client.get('/api/state').json()
            assert initial['workflow']['execution_available'],initial['workflow']
            response=client.post('/api/tasks',json={'text':'夹起红色方块','execute':True})
            assert response.status_code==200,response.text
            deadline=time.monotonic()+75;seen=[]
            while time.monotonic()<deadline:
                state=client.get('/api/state').json();workflow=state['workflow'];seen.append(workflow['state'])
                if not workflow['pending']:break
                time.sleep(.05)
            if workflow['pending']:
                client.post('/api/control',json={'action':'stop'})
                raise AssertionError('Production task exceeded bounded integration wait: '+json.dumps(workflow,ensure_ascii=False))
            print('MOCK_TRANSPORT_PRODUCTION_PATH',json.dumps({'state':workflow['state'],'error':workflow.get('error'),'outcome':workflow.get('outcome'),'physical_actions':session.sim.proprioception()['actions'],'vision_calls':mock.vision_calls,'brain_calls':mock.brain_calls},ensure_ascii=False))
            assert mock.vision_calls>=1 and mock.brain_calls==1,(mock.vision_calls,mock.brain_calls,workflow)
            assert session.sim.proprioception()['actions']>0,json.dumps(workflow,ensure_ascii=False)
            assert workflow['state'] in ('failed','completed'),workflow
            assert workflow['outcome']['calls']>0,workflow
            assert workflow['outcome']['actions']>0,workflow
            if workflow['state']=='failed':
                assert workflow['outcome']['verified'] is False and workflow['outcome']['reason']
            else:assert workflow['outcome']['verified'] is True
            assert any(event['state']=='executing' for event in workflow['events']),workflow
    finally:
        worker.join(92)
        assert not worker.is_alive(),'Bounded test worker failed to expire'
        adapter.close()
