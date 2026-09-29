import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs.experiment import diagnose

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='first post-migration update of every strategy against the true one')
    ap.add_argument('experiment')
    diagnose(ap.parse_args().experiment)
