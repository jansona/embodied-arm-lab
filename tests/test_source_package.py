"""Source packaging must preserve reorganized modules, not local runtime files."""
import json
import zipfile
import pytest
from tools import package_so101


def test_package_includes_layout_and_excludes_local_runtime(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    for name in ['README.md', 'start_so101.py', 'requirements.txt']:
        (root / name).write_text('source')
    (root / 'config.example.json').write_text(json.dumps({'api_key': ''}))
    for name in ['embodied_arm', 'experiments', 'tools', 'tests', 'docs', 'assets']:
        (root / name).mkdir()
        (root / name / 'example.txt').write_text('source')
    (root / 'tests' / 'screenshots').mkdir()
    (root / 'tests' / 'screenshots' / 'private.png').write_bytes(b'dummy image')
    (root / 'tests' / 'fixtures').mkdir()
    (root / 'tests' / 'fixtures' / 'private.json').write_text('{}')
    (root / 'tests' / 'fixtures' / 'stereo_camera_checkpoint.json').write_text('{}')
    for name in ['.runtime', '.venv']:
        (root / name).mkdir()
        (root / name / 'private.txt').write_text('excluded dummy content')
    (root / 'config.local.json').write_text('excluded dummy configuration')
    monkeypatch.setattr(package_so101, 'ROOT', root)
    destination = tmp_path / 'source.zip'
    package_so101.package(destination)
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
    assert all(f'jev-arm-sandbox/{name}/example.txt' in names
               for name in ['embodied_arm', 'experiments', 'tools', 'tests', 'docs', 'assets'])
    assert 'jev-arm-sandbox/tests/fixtures/stereo_camera_checkpoint.json' not in names
    assert not any('private.png' in name or 'private.json' in name for name in names)
    assert not any('.runtime' in name or '.venv' in name or 'config.local' in name for name in names)


def test_nonempty_example_key_is_rejected(tmp_path, monkeypatch):
    (tmp_path / 'config.example.json').write_text(json.dumps({'api_key': 'dummy-value'}))
    monkeypatch.setattr(package_so101, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='nonempty credential'):
        package_so101.package(tmp_path / 'not-created.zip')
    assert not (tmp_path / 'not-created.zip').exists()
