import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs import trace

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='update size after arrival, per strategy, against the uninterrupted run')
    ap.add_argument('experiment')
    trace.run(ap.parse_args().experiment)
