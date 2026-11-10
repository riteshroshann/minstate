import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs import latency

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='drain and restore latency per strategy, repeated; run on a quiet machine')
    ap.add_argument('experiment')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    latency.run(a.experiment, a.device)
