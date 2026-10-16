"""Local-only fake provider server for supported interactive browser QA; no real models."""
import sys
from pathlib import Path
from tests.e2e_conversation.test_browser import FakeRobot,FakeBrain,FakeTaskSource,FakeTaskExecutor,create_app
from fastapi.responses import HTMLResponse,RedirectResponse
import uvicorn
brain=FakeBrain()
executor=FakeTaskExecutor()
app=create_app(FakeRobot,brain=brain,**({'source_factory':FakeTaskSource,'task_executor':executor} if '--tasks' in sys.argv else {}))
@app.get('/__qa',response_class=HTMLResponse)
def qa():
    return '<h1>FAKE PROVIDER TEST CONTROLS</h1><p>No real model or robot calls</p>'+''.join(
        f'<form method="post" action="/__qa/{x}"><button>{x}</button></form>'
        for x in ['hold','release','fail','succeed','expire','reconnect','hold-task','release-task','unverified','verified'])+f'<p>Calls: {len(brain.calls)}</p>'
@app.post('/__qa/{action}')
def control(action:str):
    if action=='hold-task':executor.gate.clear()
    elif action=='release-task':executor.gate.set()
    elif action=='unverified':executor.verified=False
    elif action=='verified':executor.verified=True
    elif action=='hold':brain.gate.clear()
    elif action=='release':brain.gate.set()
    elif action=='fail':brain.fail=True
    elif action=='succeed':brain.fail=False
    elif action=='expire':brain.available=False
    elif action=='reconnect':brain.available=True
    return RedirectResponse('/__qa',status_code=303)
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=int(sys.argv[sys.argv.index('--port')+1]) if '--port' in sys.argv else (8768 if '--tasks' in sys.argv else 8767),log_level='warning')
