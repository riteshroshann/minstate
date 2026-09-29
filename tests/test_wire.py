import os

import pytest
import torch

from mvs.state import wire

TENSORS = {'a': torch.randn(3, 5), 'b': torch.randn(7).bfloat16(), 'c': torch.arange(9, dtype=torch.int64),
           'd': torch.tensor(2.5), 'e': torch.zeros(0), 'f': torch.randint(0, 255, (65,), dtype=torch.uint8),
           'g': torch.tensor([True, False])}


def test_roundtrip_every_dtype():
    meta, out = wire.loads(wire.dumps({'step': 3}, TENSORS))
    assert meta == {'step': 3}
    for k, t in TENSORS.items():
        assert out[k].dtype == t.dtype and out[k].shape == t.shape and torch.equal(out[k], t)


def test_alignment():
    buf = wire.dumps({}, TENSORS)
    head, start = wire.header(buf)
    assert start % 64 == 0 and all(e['offset'] % 64 == 0 for e in head['tensors'])


def test_corruption_is_rejected():
    buf = wire.dumps({}, TENSORS)
    head, start = wire.header(buf)
    buf[start + head['tensors'][0]['offset'] + 5] ^= 0xFF
    with pytest.raises(ValueError, match='checksum'):
        wire.loads(buf)


def test_truncation_is_rejected():
    buf = wire.dumps({}, {'a': torch.randn(1000)})
    with pytest.raises(ValueError, match='truncated'):
        wire.loads(buf[:-100])
    with pytest.raises(ValueError):
        wire.loads(buf[:8])


def test_bad_magic():
    with pytest.raises(ValueError, match='not an MVS'):
        wire.loads(bytearray(b'XXXX' + bytes(100)))


def test_atomic_write_and_peek(tmp_path):
    p = str(tmp_path / 'x.mvs')
    wire.write(p, wire.dumps({'step': 9, 'time': 1.0}, TENSORS))
    assert not os.path.exists(p + '.tmp') and wire.peek(p)['step'] == 9
    assert torch.equal(wire.loads(wire.read(p))[1]['a'], TENSORS['a'])
