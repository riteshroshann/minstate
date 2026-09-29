import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs import chaos

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='inject preemptions and crashes into a live job and measure recovery')
    ap.add_argument('--config')
    ap.add_argument('--world', type=int, default=1)
    ap.add_argument('--events', default='preempt@600')
    ap.add_argument('--strategy', default='full')
    ap.add_argument('--ckpt-every', type=int, default=100)
    ap.add_argument('--ckpt-strategy', default='full')
    ap.add_argument('--via', choices=('signal', 'file'), default='signal')
    ap.add_argument('--out', default='runs/chaos')
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    ov = [*a.overrides, f'strategy={a.strategy}', f'ckpt_every={a.ckpt_every}', f'ckpt_strategy={a.ckpt_strategy}']
    rep = chaos.run(a.config, ov, a.world, a.events, a.out, a.via)
    print(chaos.markdown(rep))
