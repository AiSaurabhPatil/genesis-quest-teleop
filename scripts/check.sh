#!/usr/bin/env bash
set -euo pipefail
uv sync --locked
uv pip check
nvidia-smi >/dev/null
uv run python -m compileall -q src
uv run python - <<'PY'
import aiohttp, aiortc, genesis, numpy, scipy, torch, yaml
from genesis_quest_teleop.protocol import parse_quest_packet
from genesis_quest_teleop.input.frames import map_webxr_position_to_genesis
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.get_device_capability(0) == (12, 0)
y=torch.ones(256,device='cuda')*2; torch.cuda.synchronize(); assert y.is_cuda
print('imports-and-torch-cuda: ok')
PY
uv run python - <<'PY'
import genesis as gs
gs.init(backend=gs.cuda, logging_level='warning')
assert gs.backend == gs.cuda
scene=gs.Scene(show_viewer=False); scene.add_entity(gs.morphs.Plane()); scene.build()
for _ in range(10): scene.step()
print('genesis-cuda-smoke: ok')
PY
uv run ruff check src
