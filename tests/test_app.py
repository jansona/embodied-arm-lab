from fastapi.testclient import TestClient
from app import app

def test_web_api():
    with TestClient(app) as c:
        assert c.get('/').status_code==200
        state=c.get('/api/state').json();assert not state['jev']['configured']
        assert c.get('/api/frame').headers['content-type']=='image/jpeg'
        preview=c.get('/api/jev/preview').json();assert preview['body']['questions']['action']['type']=='choice'
        assert 'Authorization' not in str(preview)
        assert c.post('/api/control',json={'action':'jev'}).status_code==409
        assert c.post('/api/jev/start',json={'max_calls':1,'max_seconds':3}).status_code==409
        assert c.post('/api/control',json={'action':'xyz','values':[100,0,0]}).status_code==409
        assert c.post('/api/control',json={'action':'stop'},headers={'Origin':'https://evil.example'}).status_code==403
        assert c.post('/api/control',json={'action':'reset'}).status_code==200
        assert c.post('/api/control',json={'action':'pause'}).json()['paused']
        assert not c.post('/api/control',json={'action':'resume'}).json()['paused']
