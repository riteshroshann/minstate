"""Assemble every venue folder from paper/ and compile it: python submissions/tools/build.py [venue ...]."""
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER, OUT = ROOT / 'paper', ROOT / 'submissions'
NAME = 'MinimumViableState'
TECTONIC = shutil.which('tectonic') or str(Path.home() / '.local' / 'bin' / 'tectonic.exe')

# venue -> (folder, main.tex to use, extra files beside it, has supplement, has cover letter, running head)
VENUES = {
    'IEEE_TPDS': ('journals', PAPER / 'main.tex', [], True, True, 'IEEE Transactions on Parallel and Distributed Systems'),
    'IEEE_TCC': ('journals', PAPER / 'main.tex', [], True, True, 'IEEE Transactions on Cloud Computing'),
    'TMLR': ('journals', PAPER / 'venues' / 'tmlr' / 'main.tex',
             [PAPER / 'venues' / 'tmlr' / f for f in ('tmlr.sty', 'tmlr.bst', 'LICENSE')], False, False, None),
    'arxiv': ('arxiv', PAPER / 'main.tex', [], False, False, None),
}


def compile_tex(src, name, keep=False):
    flags = ['--keep-intermediates'] if keep else []
    r = subprocess.run([TECTONIC, '-X', 'compile', *flags, name], cwd=src, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'{src / name} failed to build:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}')


def assemble(venue):
    kind, main, extra, supplement, letter, head = VENUES[venue]
    dst = OUT / kind / (NAME if kind == 'arxiv' else f'{venue}/{NAME}')
    src = dst / 'source'
    if src.exists():
        shutil.rmtree(src)
    (src / 'gen' / 'figures').mkdir(parents=True)
    shutil.copytree(PAPER / 'parts', src / 'parts')
    for f in (PAPER / 'gen').glob('*.tex'):
        shutil.copy(f, src / 'gen')
    for f in (PAPER / 'gen' / 'figures').glob('*.pdf'):
        shutil.copy(f, src / 'gen' / 'figures')
    shutil.copy(PAPER / 'refs.bib', src)
    text = main.read_text(encoding='utf-8')
    if head:
        text = text.replace('\\markboth{}', f'\\markboth{{{head}}}')
    (src / 'main.tex').write_text(text, encoding='utf-8')
    for f in extra:
        shutil.copy(f, src)
    compile_tex(src, 'main.tex', keep=kind == 'arxiv')
    for junk in ('main.aux', 'main.log', 'main.blg', 'main.out', 'main.xdv'):  # arXiv wants the .bbl and nothing else
        (src / junk).unlink(missing_ok=True)
    shutil.move(src / 'main.pdf', dst / 'paper.pdf')
    if supplement:
        shutil.copy(PAPER / 'supplement.tex', src)
        compile_tex(src, 'supplement.tex')
        shutil.move(src / 'supplement.pdf', dst / 'supplement.pdf')
    if letter:
        stem = OUT / 'tools' / f'{venue}_cover_letter'
        shutil.copy(stem.with_suffix('.tex'), src / 'cover_letter.tex')
        shutil.copy(stem.with_suffix('.txt'), dst / 'cover_letter.txt')
        compile_tex(src, 'cover_letter.tex')
        shutil.move(src / 'cover_letter.pdf', dst / 'cover_letter.pdf')
    if kind == 'arxiv':
        with tarfile.open(dst / 'upload_to_arxiv.tar.gz', 'w:gz') as tar:
            for f in sorted(src.rglob('*')):
                if f.is_file():
                    tar.add(f, f.relative_to(src).as_posix())
    return dst


if __name__ == '__main__':
    subprocess.run([sys.executable, str(OUT / 'tools' / 'make_letters.py')], check=True)
    for venue in sys.argv[1:] or VENUES:
        print(assemble(venue))
