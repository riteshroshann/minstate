import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mvs import chaos
from mvs.launcher import launch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = str(ROOT / 'scripts' / 'train.py')


def losses(path):
    out = {}
    for line in open(path):
        e = json.loads(line)
        if 'loss' in e:
            out[e['t']] = e['loss']
    return out


def events(path, name):
    return [json.loads(l) for l in open(path) if f'"event": "{name}"' in l]


@pytest.mark.skipif(not hasattr(signal, 'SIGTERM') or sys.platform == 'win32', reason='posix signal test')
def test_real_signal_drains_and_resumes_bitwise(overrides, tmp_path):
    ctrl, job = tmp_path / 'ctrl', tmp_path / 'job'
    subprocess.run([sys.executable, TRAIN, *overrides, f'out_dir={ctrl}'], check=True)
    cmd = [sys.executable, TRAIN, *overrides, f'out_dir={job}', 'strategy=full']
    p = subprocess.Popen(cmd)
    log = job / 'log.jsonl'
    while not (log.exists() and any(t >= 20 for t in losses(log))):
        time.sleep(0.02)
    p.send_signal(signal.SIGTERM)
    assert p.wait() == 3
    drain = events(log, 'drain')[0]
    assert drain['source'] == 'signal' and drain['strategy'] == 'full'
    subprocess.run(cmd + ['auto_resume=true'], check=True)
    assert events(log, 'resume')[0]['step'] == drain['t']
    assert losses(log) == losses(ctrl / 'log.jsonl')


def test_hard_crash_restarts_from_checkpoint(overrides, tmp_path):
    ctrl, job = tmp_path / 'ctrl', tmp_path / 'job'
    assert launch([sys.executable, TRAIN, *overrides, f'out_dir={ctrl}'], 2) == 0
    code = launch([sys.executable, TRAIN, *overrides, f'out_dir={job}', 'ckpt_every=20', 'ckpt_strategy=full',
                   'crash_at=35', 'auto_resume=true'], 2, max_restarts=2)
    assert code == 0 and events(job / 'log.jsonl', 'resume')[0]['step'] == 20
    assert losses(job / 'log.jsonl') == losses(ctrl / 'log.jsonl')


@pytest.mark.parametrize('via', ['signal', 'file'])
def test_chaos_resize_crash_resize_recovers(overrides, tmp_path, via):
    rep = chaos.run(None, [*overrides, 'device=cpu', 'max_steps=70', 'strategy=full', 'ckpt_every=10', 'ckpt_strategy=full'], 2,
                    'preempt@20:1,crash@40,preempt@55:2', tmp_path, via)
    s, rows = rep['summary'], rep['rows']
    assert [r['world'] for r in rows] == ['2->1', '1->1', '1->2']
    assert rows[0]['steps_redone'] == 0 and rows[1]['steps_redone'] > 0 and rows[1]['resumed_from'] == 40
    assert s['steps'] == 70 and s['max_loss_gap'] < 1e-5
    assert (tmp_path / 'recovery.md').exists()
