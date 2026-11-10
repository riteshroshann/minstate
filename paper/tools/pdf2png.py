"""Render PDF pages to PNG for visual review: pdf2png.py file.pdf [dpi] [first-last]."""
import sys
from pathlib import Path

import fitz

src = Path(sys.argv[1])
dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 110
doc = fitz.open(src)
lo, hi = (map(int, sys.argv[3].split('-')) if len(sys.argv) > 3 else (1, len(doc)))
out = src.parent / '_png'
out.mkdir(exist_ok=True)
for i in range(lo - 1, min(hi, len(doc))):
    path = out / f'{src.stem}_{i + 1}.png'
    doc[i].get_pixmap(dpi=dpi).save(path)
    print(path)
