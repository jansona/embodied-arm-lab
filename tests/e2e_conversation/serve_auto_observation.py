"""MOCK-only local browser fixture. No external APIs, real vision, physics or credentials."""
from fastapi.responses import HTMLResponse,RedirectResponse
from embodied_arm.web.app_so101 import create_app
from tests.e2e_conversation.test_browser import FakeRobot
from tests.e2e_conversation.test_auto_observation import MockVision,MockBrain,MockSource
import uvicorn
brain=MockBrain();vision=MockVision()
app=create_app(FakeRobot,vision=vision,brain=brain,source_factory=MockSource)
@app.get('/__qa',response_class=HTMLResponse)
def qa():
    source=app.state.session.source
    return '<h1>MOCK AUTO OBSERVATION CONTROLS</h1><p>No real provider or robot</p>'+''.join(
        f'<form method="post" action="/__qa/{x}"><button>{x}</button></form>'
        for x in ['hold-observation','release-observation','vision-offline','vision-online','brain-offline','brain-online','stale','reset'])+f'<p>MOCK vision calls: {vision.calls}; brain calls: {len(brain.calls)}; refreshes: {source.refreshes}</p>'
@app.post('/__qa/{action}')
def control(action:str):
    source=app.state.session.source
    if action=='hold-observation':source.scene=None;source.gate.clear()
    elif action=='release-observation':source.gate.set()
    elif action=='vision-offline':vision.available=False;source.scene=None
    elif action=='vision-online':vision.available=True
    elif action=='brain-offline':brain.available=False
    elif action=='brain-online':brain.available=True
    elif action=='stale':source.stale=True
    elif action=='reset':
        from embodied_arm.web.app_so101 import ControlRequest
        app.state.session.control(ControlRequest(action='reset'))
    return RedirectResponse('/__qa',status_code=303)
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=8770,log_level='warning')
