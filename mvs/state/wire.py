import json
import math
import os
import struct
import zlib

import torch

MAGIC, VERSION, ALIGN = b'MVS\x01', 1, 64
DTYPES = {str(d): d for d in (torch.float32, torch.bfloat16, torch.float16, torch.float64, torch.int8, torch.uint8,
                              torch.int16, torch.int32, torch.int64, torch.bool)}


def raw(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()


def dumps(meta, tensors):
    table, blobs, off = [], [], 0
    for key, t in tensors.items():
        b = raw(t.cpu())
        table.append({'key': key, 'dtype': str(t.dtype), 'shape': list(t.shape), 'offset': off,
                      'nbytes': b.nbytes, 'crc': zlib.crc32(b)})
        blobs.append((off, b))
        off += b.nbytes + (-b.nbytes) % ALIGN
    head = json.dumps({'version': VERSION, 'meta': meta, 'tensors': table}, separators=(',', ':')).encode()
    start = 12 + len(head) + (-(12 + len(head))) % ALIGN
    buf = bytearray(start + off)
    buf[:12] = MAGIC + struct.pack('<Q', len(head))
    buf[12:12 + len(head)] = head
    view = memoryview(buf)
    for o, b in blobs:
        view[start + o:start + o + b.nbytes] = memoryview(b).cast('B')
    return buf


def header(buf):
    if len(buf) < 12 or bytes(buf[:4]) != MAGIC:
        raise ValueError('not an MVS payload')
    n = struct.unpack('<Q', bytes(buf[4:12]))[0]
    if 12 + n > len(buf):
        raise ValueError('truncated header')
    head = json.loads(bytes(buf[12:12 + n]))
    if head.get('version') != VERSION:
        raise ValueError(f'unsupported payload version {head.get("version")}')
    return head, 12 + n + (-(12 + n)) % ALIGN


def loads(buf, verify=True):
    head, start = header(buf)
    view, out = memoryview(buf), {}
    for e in head['tensors']:
        a = start + e['offset']
        z = a + e['nbytes']
        if z > len(buf):
            raise ValueError(f'truncated payload at {e["key"]}')
        if verify and zlib.crc32(view[a:z]) != e['crc']:
            raise ValueError(f'checksum mismatch at {e["key"]}')
        dt, n = DTYPES[e['dtype']], math.prod(e['shape'])
        t = torch.frombuffer(buf, dtype=dt, count=n, offset=a) if n else torch.empty(0, dtype=dt)
        out[e['key']] = t.view(e['shape'])
    return head['meta'], out


def write(path, buf):
    tmp = f'{path}.tmp'
    with open(tmp, 'wb') as f:
        f.write(buf)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read(path):
    with open(path, 'rb') as f:
        buf = bytearray(os.fstat(f.fileno()).st_size)
        f.readinto(buf)
    return buf


def peek(path):
    with open(path, 'rb') as f:
        pre = f.read(12)
        if len(pre) < 12 or pre[:4] != MAGIC:
            raise ValueError('not an MVS payload')
        n = struct.unpack('<Q', pre[4:])[0]
        return json.loads(f.read(n))['meta']
