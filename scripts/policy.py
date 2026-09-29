import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs.policy import MODELS, decide, render

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='what to carry, and whether to move')
    ap.add_argument('summary')
    ap.add_argument('--model', default='llama-7b', choices=sorted(MODELS))
    ap.add_argument('--notice-s', type=float, default=30.0)
    ap.add_argument('--bandwidth-gbps', type=float, default=10.0)
    ap.add_argument('--frac', type=float, default=0.6)
    ap.add_argument('--step-s', type=float, default=6.0)
    ap.add_argument('--gpus', type=int, default=8)
    ap.add_argument('--price', type=float, default=1.2)
    ap.add_argument('--egress', type=float, default=0.02)
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--src-price', type=float)
    ap.add_argument('--remaining', type=int)
    a = ap.parse_args()
    print(render(decide(a.model, a.summary, a.frac, a.bandwidth_gbps, a.step_s, a.gpus, a.price, a.egress, a.notice_s,
                        a.ckpt_every, a.src_price, a.remaining)))
