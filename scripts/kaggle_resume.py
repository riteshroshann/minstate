"""Runs on Kaggle: resume the laptop's payloads on a GPU in a data centre and keep the held-out loss of each.

The code and the payloads arrive as two attached datasets. Only the result files are left in the output.
"""
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

INPUT, WORK, OUT = Path('/kaggle/input'), Path('/kaggle/tmp/minstate'), Path('/kaggle/working')


def find(pattern):
    hits = sorted(INPUT.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f'{pattern} under {INPUT}: {[str(p) for p in INPUT.rglob("*")][:40]}')
    return hits


if __name__ == '__main__':
    # Kaggle unpacks an uploaded archive on its own; take the tree if it did and the zip if it did not
    packed = sorted(INPUT.rglob('minstate_code.zip'))
    if packed:
        WORK.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(packed[0]) as z:
            z.extractall(WORK)
    else:
        shutil.copytree(find('mvs/__init__.py')[0].parents[1], WORK)
    payloads = WORK / 'runs' / 'xdevice' / 'payloads'
    payloads.mkdir(parents=True)
    for p in find('*.mvs'):
        os.symlink(p, payloads / p.name)
    subprocess.run([sys.executable, '-u', 'scripts/xdevice.py', 'configs/experiments/xdevice.yaml', 'resume',
                    '--device', 'cuda', '--label', 't4'], cwd=WORK, check=True)
    shutil.copytree(WORK / 'runs' / 'xdevice' / 't4', OUT / 't4')
