import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs.analysis import read_csv, report, table

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='summary.csv, table and figures from a sweep, or the table from a summary')
    ap.add_argument('source', help='runs/<exp> directory or a summary.csv')
    ap.add_argument('--out', default='reports/latest')
    a = ap.parse_args()
    print(table(read_csv(a.source) if a.source.endswith('.csv') else report(a.source, a.out)))
