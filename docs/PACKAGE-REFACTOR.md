# Package refactor verification — 2026-10-16

## Structure

Root files are now only `.gitignore`, `README.md`, `pyproject.toml`, and `requirements.txt`. Application code, robot/web assets, workers, and legacy Panda code live under `embodied_arm/`; configuration templates, scripts, tests, documentation, experiments and packaging tools have dedicated directories. Root module compatibility stubs and `sys.path` modifications are not used.

`python -m embodied_arm` and the installed `embodied-arm` console script share the same entry point. Worker subprocesses use `python -m embodied_arm.runtime.<worker>`. Existing loopback binding, preview default, worker budgets, explicit configuration, stop/reset and capability gates are preserved.

The root-level generated numerical reports were moved locally to `tests/fixtures/physics/`. These reports remain ignored and excluded from source/wheel distributions. Historical documents were preserved under `docs/`.

## IPC race diagnosed and fixed

The initial full run reproduced the earlier intermittent SO101 IPC failure: a safe `unavailable` response instead of the expected rejection reason. This was not a timeout shortage.

The worker atomically replaces `ready.json`. When replacement happens between a reader's `open()` and `fstat()`, the old, already-open inode has link count zero. The strict reader rejected it, status appeared offline, and the bridge cancelled a healthy request. A controlled replacement reproduced this deterministically; a repeated rejection probe also captured it.

The reader now reopens **only `ready.json`, at most once**, when that exact unlinked-snapshot case occurs. It never accepts an unlinked inode. Regular-file, ownership, private-permission, hardlink and size checks remain enforced. A second replacement fails closed. New deterministic tests cover both recovery and the retry bound.

## Verification

- Full offline suite after the IPC fix: **807 passed, 16 skipped, 121 subtests passed**, 243.08 seconds
- Skips: 10 opt-in browser scenarios and 6 missing historical fixture checks; these are not passing tests
- One pre-existing FastAPI/Starlette HTTPX TestClient deprecation warning
- IPC selection: 16 passed and 4 subtests passed
- Repeated rejection stress probe: 100 cycles / 400 roundtrips passed
- Latest layout/source-package selection after packaging hardening: 8 passed, including rejection of a symlinked configuration template before reading it
- Shell syntax, Python compilation and whitespace checks passed

The layout/source-package selection overlaps the full run and must not be added as unique tests. Its newly added symlink-template check ran after full-suite collection.

## Distribution and startup checks

Built a wheel without network/dependency fetching and installed it into the existing test environment. From an unrelated working directory, confirmed import from the installed package, loaded actual SO101 URDF/meshes, rendered stereo frames, started the default loopback application, received HTTP 200 for state/root/JS/CSS, verified manipulation remained disabled, and shut down cleanly with SIGINT.

An independently extracted source archive passed the same asset/preview HTTP/shutdown checks, invoked through its shell entry from an unrelated working directory. Source archives include project metadata and package resources; recorded JSON, runtime state, caches, screenshots and real configuration remain excluded. Wheel resources contain robot assets and both UIs, not source-only experiments/tests or configuration.

## Unchanged product limitations

This verifies structure, packaging and offline behavior. It does not establish live Qwen/Jev access, image-only grasp/place success or real-robot safety. No live API call, credential read, physics-policy change or production manipulation enablement was made during this refactor. See [the product checkpoint](PRODUCT-CHECKPOINT.md) for the independent perception and provider-access blockers.
