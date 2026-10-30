import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def git():
    def run(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {'commit': run('rev-parse', 'HEAD') or None, 'dirty': bool(run('status', '--porcelain', '--', 'mvs'))}
    except OSError:
        return {'commit': None, 'dirty': None}


def device_name(device):
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    if device.type == 'xpu':
        return torch.xpu.get_device_name(device)
    return platform.processor() or platform.machine()


def environment(device):
    return {**git(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version(),
            'platform': platform.platform(), 'device': str(device), 'device_name': device_name(device)}


def write(path, cfg, device, **fields):
    # one file per run: enough to rebuild the exact command and know which code produced the numbers
    doc = {**environment(device), 'config': asdict(cfg), **fields}
    Path(path).write_text(json.dumps(doc, indent=1, default=str))
    return doc
