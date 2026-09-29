from pathlib import Path

import pytest

from mvs.policy import decide, payload, shapes

SUMMARY = str(Path(__file__).resolve().parents[1] / 'reports' / 'paper' / 'summary.csv')


@pytest.mark.parametrize('name,gb', [('full', 80.9), ('w_mq8_vlog8', 40.7), ('w_m_vr1', 53.9), ('w_vr1', 27.0),
                                     ('wbf16_vr1', 13.5), ('w_fresh', 27.0)])
def test_llama_7b_payloads(name, gb):
    assert sum(payload(shapes('llama-7b'), name).values()) / 1e9 == pytest.approx(gb, abs=0.05)


def test_thirty_second_warning():
    d = decide('llama-7b', SUMMARY, notice=30)
    rows = {r['strategy']: r for r in d['table']}
    assert d['choice'] == 'wbf16_vr1' and rows['wbf16_vr1']['total'] == pytest.approx(0.39, abs=0.01)
    assert not rows['full']['fits'] and rows['full']['total'] == pytest.approx(8.0)
    assert rows['full']['xfer'] == pytest.approx(64.7, abs=0.05) and rows['w_fresh']['lam'] == 12.5


def test_two_minute_warning_and_opportunistic_move():
    d = decide('llama-7b', SUMMARY, notice=120, src_price=4.0, remaining=20000)
    rows = {r['strategy']: r for r in d['table']}
    assert all(r['fits'] for r in d['table']) and rows['full']['total'] == pytest.approx(1.79, abs=0.01)
    assert d['move'] and d['saving'] == pytest.approx(746.67, abs=0.01) and d['break_even_steps'] == 11


def test_qlora_payload_is_adapters_only():
    sh = shapes('llama-7b-qlora')
    assert sum(payload(sh, 'full').values()) == 12 * 8_388_608
