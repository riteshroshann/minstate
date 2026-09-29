import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs.experiment import run

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='resumable sweep: controls, forks, one migration per strategy')
    ap.add_argument('experiment')
    run(ap.parse_args().experiment)
