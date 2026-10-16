"""Source packaging must preserve reorganized modules, not local runtime files."""
import json
import zipfile
import pytest
from tools import package_so101


def test_package_includes_layout_and_excludes_local_runtime(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    for name in ['README.md', 'pyproject.toml', 'requirements.txt']:
        (root / name).write_text('source')
    (root / 'config').mkdir()
    (root / 'config/provider.example.json').write_text(json.dumps({'api_key': ''}))
    for name in ['embodied_arm', 'experiments', 'tools', 'tests', 'docs', 'scripts']:
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
    (root / 'private.txt').write_text('excluded dummy text')
    (root / 'embodied_arm/assets/so101/assets').mkdir(parents=True)
    (root / 'embodied_arm/assets/so101/so101.urdf').write_text('robot fixture')
    (root / 'embodied_arm/assets/so101/assets/part.stl').write_text('mesh fixture')
    (root / 'embodied_arm/assets/so101/source_manifest.json').write_text('{}')
    monkeypatch.setattr(package_so101, 'ROOT', root)
    destination = tmp_path / 'source.zip'
    package_so101.package(destination)
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
    assert all(f'jev-arm-sandbox/{name}/example.txt' in names
               for name in ['embodied_arm', 'experiments', 'tools', 'tests', 'docs', 'scripts'])
    assert 'jev-arm-sandbox/pyproject.toml' in names
    assert 'jev-arm-sandbox/config/provider.example.json' in names
    assert 'jev-arm-sandbox/embodied_arm/assets/so101/assets/part.stl' in names
    assert 'jev-arm-sandbox/embodied_arm/assets/so101/source_manifest.json' in names
    assert 'jev-arm-sandbox/tests/fixtures/stereo_camera_checkpoint.json' not in names
    assert not any('private.png' in name or 'private.json' in name for name in names)
    assert not any('.runtime' in name or '.venv' in name or 'config.local' in name or name.endswith('/private.txt') for name in names)


def test_nonempty_example_key_is_rejected(tmp_path, monkeypatch):
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/provider.example.json').write_text(json.dumps({'api_key': 'dummy-value'}))
    monkeypatch.setattr(package_so101, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='nonempty credential'):
        package_so101.package(tmp_path / 'not-created.zip')
    assert not (tmp_path / 'not-created.zip').exists()


def test_symlink_template_is_rejected_before_read(tmp_path, monkeypatch):
    (tmp_path / 'config').mkdir()
    target=tmp_path / 'unrelated.json'
    target.write_text('{}')
    (tmp_path / 'config/provider.example.json').symlink_to(target)
    monkeypatch.setattr(package_so101, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Unsafe configuration template path'):
        package_so101.package(tmp_path / 'not-created.zip')
    assert not (tmp_path / 'not-created.zip').exists()
