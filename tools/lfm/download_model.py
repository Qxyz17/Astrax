import urllib.request as u, pathlib, time
dest = pathlib.Path('data/LFM2-350M-Q4_K_M.gguf')
if dest.exists() and dest.stat().st_size > 200*1024*1024:
    print('already downloaded', dest.stat().st_size)
    raise SystemExit(0)
url = 'https://huggingface.co/LiquidAI/LFM2-350M-GGUF/resolve/main/LFM2-350M-Q4_K_M.gguf'
req = u.Request(url, headers={'User-Agent':'Mozilla/5.0'})
start = time.time(); total = 0
with u.urlopen(req, timeout=120) as r, dest.open('wb') as f:
    while True:
        c = r.read(1024*1024)
        if not c: break
        f.write(c); total += len(c)
        if total % (32*1024*1024) < 1024*1024:
            print(f'{total//(1024*1024)} MiB', flush=True)
print(f'done {total} bytes in {time.time()-start:.0f}s')
