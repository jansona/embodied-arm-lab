"""Shared locations, independent of the module's subpackage and caller cwd.

Source checkouts keep the existing repository-private IPC queues. Installed
packages use a user-owned runtime directory, never write into site-packages.
Importing this module does not create directories or read configuration.
"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
_candidate = PACKAGE_ROOT.parent
PROJECT_ROOT = _candidate if (_candidate / 'pyproject.toml').is_file() else None
CONFIG_EXCLUSION_ROOT = PROJECT_ROOT or PACKAGE_ROOT
WORKSPACE_ROOT = PROJECT_ROOT or Path.cwd()
RUNTIME_ROOT = (PROJECT_ROOT / '.runtime' if PROJECT_ROOT else
                Path.home() / '.embodied-arm-runtime')
ASSET_ROOT = PACKAGE_ROOT / 'assets'
WEB_ROOT = PACKAGE_ROOT / 'web'
