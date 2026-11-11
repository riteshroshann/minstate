import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs import link

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='migrate between two processes over a paced link; run on a quiet machine')
    ap.add_argument('mode', choices=['run', 'serve'], help='run drives the experiment; serve is the destination it starts')
    ap.add_argument('experiment')
    a = ap.parse_args()
    if a.mode == 'serve':
        link.serve(a.experiment, link.spec(a.experiment)['port'])
    else:
        link.run(a.experiment)
