import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs import xdevice

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='cross-device migration: encode on one device, resume on another')
    ap.add_argument('experiment')
    ap.add_argument('stage', choices=('encode', 'resume', 'report'))
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    e = xdevice.spec(a.experiment)
    if a.stage == 'report':
        print(f'{len(xdevice.report(e))} rows')
    else:
        getattr(xdevice, a.stage)(e, a.device)
