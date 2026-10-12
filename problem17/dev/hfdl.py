#!/usr/bin/env python3
"""Parallel HF downloader writing straight into the hub cache layout (IPv4 forced)."""
import os, sys, json, socket, threading, time
import urllib3.util.connection as c
c.allowed_gai_family = lambda: socket.AF_INET
import requests

repo = sys.argv[1]
skip = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else set()
NCONN = int(os.environ.get('NCONN', '12'))
api = requests.get(f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=60).json()
sha = api['sha']
base = os.path.expanduser(f'~/.cache/huggingface/hub/models--{repo.replace("/", "--")}')
snap = os.path.join(base, 'snapshots', sha)
os.makedirs(snap, exist_ok=True); os.makedirs(os.path.join(base, 'refs'), exist_ok=True)
open(os.path.join(base, 'refs', 'main'), 'w').write(sha)

def first_zero_block(path, start, end, blk=1 << 16):
    """Resume point: first 64 KB block inside [start, end] that is entirely zero (unwritten)."""
    with open(path, 'rb') as f:
        pos = start
        while pos <= end:
            f.seek(pos); b = f.read(min(blk, end - pos + 1))
            if not any(b): return pos
            pos += len(b)
    return end + 1

def fetch_range(url, start, end, out, idx, prog):
    pos = first_zero_block(out, start, end); prog[idx] = pos - start
    for attempt in range(200):
        if pos > end: return
        try:
            r = requests.get(url, headers={'Range': f'bytes={pos}-{end}'}, stream=True, timeout=60)
            with open(out, 'r+b') as f:
                f.seek(pos)
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk); pos += len(chunk); prog[idx] += len(chunk)
            if pos > end: return
        except Exception as e:
            print('retry', idx, str(e)[:80], file=sys.stderr); time.sleep(2)

for s in api['siblings']:
    name, size = s['rfilename'], s.get('size') or 0
    if name in skip: continue
    dst = os.path.join(snap, name)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print('have', name); continue
    url = f'https://huggingface.co/{repo}/resolve/{sha}/{name}'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if size < 5_000_000:
        open(dst, 'wb').write(requests.get(url, timeout=120).content); print('ok', name); continue
    tmp = dst + '.part'
    if not (os.path.exists(tmp) and os.path.getsize(tmp) == size):
        with open(tmp, 'wb') as f: f.truncate(size)
    n = NCONN; step = size // n + 1
    prog = [0]*n; ths = []
    for i in range(n):
        a, b = i*step, min(size-1, (i+1)*step-1)
        if a > b: continue
        t = threading.Thread(target=fetch_range, args=(url, a, b, tmp, i, prog)); t.start(); ths.append(t)
    t0 = time.time()
    while any(t.is_alive() for t in ths):
        time.sleep(10); done = sum(prog)
        print(f'{name}: {done/1e6:.0f}/{size/1e6:.0f} MB  {done/1e6/(time.time()-t0):.2f} MB/s', flush=True)
    os.rename(tmp, dst); print('ok', name, flush=True)
print('DONE', repo)
