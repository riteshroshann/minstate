import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs.launcher import launch

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='torchrun-compatible local launcher with restart semantics')
    ap.add_argument('--nproc', type=int, default=1)
    ap.add_argument('--max-restarts', type=int, default=0)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd
    sys.exit(launch([sys.executable, *cmd], a.nproc, a.max_restarts))
