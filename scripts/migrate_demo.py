import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs import chaos

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='real OS signal, drain, fresh-process resume, compare with control')
    ap.add_argument('--config')
    ap.add_argument('--strategy', default='w_vr1')
    ap.add_argument('--at', type=int, default=600)
    ap.add_argument('--out', default='runs/migrate_demo')
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    rep = chaos.run(a.config, [*a.overrides, f'strategy={a.strategy}'], 1, f'preempt@{a.at}', a.out, 'signal')
    print(chaos.markdown(rep))
