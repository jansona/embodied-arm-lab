from embodied_arm.runtime.start_session import commands

def test_session_is_fixed_local_and_explicitly_bounded():
    all_jobs=commands();workers=commands(True)
    assert len(all_jobs)==3 and len(workers)==2
    assert all_jobs[1:]==workers
    assert all_jobs[0][-3:]==['127.0.0.1','--port','8765']
    assert workers[0][-4:]==['--max-calls','6','--max-seconds','600']
    assert workers[1][-5:]==['--micro-phased','--max-calls','200','--max-seconds','420']
    assert not any('api_key' in arg or 'Bearer' in arg for job in all_jobs for arg in job)
