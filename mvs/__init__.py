import sys

__version__ = '1.0.0'

# tables use Δ, ×, ± etc.; a Windows console defaults to cp1252 and would raise on them
for _s in (sys.stdout, sys.stderr):
    if _s and hasattr(_s, 'reconfigure') and (_s.encoding or '').lower().replace('-', '') != 'utf8':
        _s.reconfigure(encoding='utf-8')
