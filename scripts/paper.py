import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs import analysis, paper, xdevice

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='rebuild every table, figure and number the paper inputs')
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--reports', default='reports')
    ap.add_argument('--out', default='paper/gen')
    ap.add_argument('--xdevice', help='cross-device experiment yaml, to refresh its report first')
    a = ap.parse_args()
    for sweep in ('shakespeare_scales',):
        if (Path(a.runs) / sweep).exists():
            analysis.report(Path(a.runs) / sweep, Path(a.reports) / sweep)
    if a.xdevice:
        xdevice.report(xdevice.spec(a.xdevice))
    for k, v in paper.build(a.runs, a.reports, a.out).items():
        print(f'{k} = {v}')
