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
    ap.add_argument('--conditions', nargs='*', help='subset of the conditions, e.g. when bf16 is too slow on a CPU')
    ap.add_argument('--scales', nargs='*')
    ap.add_argument('--label', help='name for this destination in the report, when it is a second machine')
    a = ap.parse_args()
    e = xdevice.spec(a.experiment)
    if a.stage == 'report':
        print(f'{len(xdevice.report(e))} rows')
    elif a.stage == 'encode':
        xdevice.encode(e, a.device)
    else:
        xdevice.resume(e, a.device, a.conditions, a.scales, a.label)
