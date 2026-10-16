"""Keep the public application boundary organized and relocation-safe."""
import ast
from pathlib import Path

from embodied_arm.paths import ASSET_ROOT, CONFIG_EXCLUSION_ROOT, PACKAGE_ROOT, PROJECT_ROOT, RUNTIME_ROOT, WEB_ROOT
from embodied_arm.runtime import brain_ipc, ipc, so101_ipc, vision_ipc
from embodied_arm.runtime.start_so101 import commands


def test_source_paths_share_repository_runtime_and_config_boundary():
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert CONFIG_EXCLUSION_ROOT == PROJECT_ROOT
    assert RUNTIME_ROOT == PROJECT_ROOT / '.runtime'
    assert {p.parent for p in (ipc.IPC_ROOT, brain_ipc.BRAIN_IPC_ROOT,
                              so101_ipc.SO101_IPC_ROOT, vision_ipc.VISION_IPC_ROOT)} == {RUNTIME_ROOT}


def test_packaged_assets_and_pages_exist():
    assert (ASSET_ROOT / 'so101/so101.urdf').is_file()
    assert (ASSET_ROOT / 'so101/assets/base_so101_v2.stl').is_file()
    assert (WEB_ROOT / 'static_so101/index.html').is_file()
    assert (WEB_ROOT / 'static_so101/ui.js').is_file()
    assert (WEB_ROOT / 'static/index.html').is_file()


def test_root_has_no_implementation_or_shell_scripts():
    assert not list(PROJECT_ROOT.glob('*.py'))
    assert not list(PROJECT_ROOT.glob('*.sh'))
    assert not list(PROJECT_ROOT.glob('SO101-*.json'))


def test_no_import_path_hacks_or_legacy_top_level_imports():
    old_names={'app', 'app_so101', 'brain', 'adapters', 'runner', 'simulator',
               'so101_simulation', 'so101_robot', 'vision_adapter', 'camera_perception',
               'ipc', 'brain_ipc', 'vision_ipc', 'so101_ipc'}
    for path in PACKAGE_ROOT.rglob('*.py'):
        source=path.read_text()
        assert 'sys.path.' not in source, str(path)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert not any(alias.name in old_names for alias in node.names), str(path)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module not in old_names, str(path)


def test_workers_launch_as_package_modules(tmp_path):
    jobs=commands(live=True, config=str(tmp_path/'outside.json'))
    assert all(job[1]=='-m' for job in jobs)
    assert all(job[2].startswith('embodied_arm.runtime.') for job in jobs[1:])
