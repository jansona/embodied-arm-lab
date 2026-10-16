"""Local-only MOCK providers/primitive with real task and sequence services."""
import threading
import uvicorn
from fastapi.responses import HTMLResponse,RedirectResponse
from embodied_arm.web.app_so101 import create_app
from tests.e2e_conversation.test_browser import FakeRobot
from tests.e2e_conversation.test_resumable_camera_loop import sequence_task_fixture
source,brain,primitive,service=sequence_task_fixture()
source.status=lambda:{'scene':None,'reason':'MOCK-SYNTHETIC-SCENE'}
gate=threading.Event()
brain.on_repair=lambda:gate.wait(60)
app=create_app(FakeRobot,brain=brain,source_factory=lambda sim,vision=None:source,
               task_executor=service.executor)
@app.get('/__qa',response_class=HTMLResponse)
def qa():
    return '<h1>MOCK cycle-review fixture</h1><p>No live providers or physical actions</p><form method="post" action="/__qa/release"><button>Release review</button></form>'
@app.post('/__qa/release')
def release():
    gate.set();return RedirectResponse('/__qa',status_code=303)
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=8772,log_level='warning')
