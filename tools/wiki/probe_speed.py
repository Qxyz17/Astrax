import urllib.request as u, time

def measure(label, url):
    try:
        req = u.Request(url, headers={'User-Agent':'Mozilla/5.0 AstraxBot'})
        r = u.urlopen(req, timeout=20)
        start = time.time(); total = 0
        while total < 256*1024 and time.time()-start < 12:
            chunk = r.read(64*1024)
            if not chunk: break
            total += len(chunk)
        dt = max(0.001, time.time()-start)
        print(f'{label}: {total} bytes in {dt:.1f}s = {total/dt/1024:.0f} KiB/s')
    except Exception as e:
        print(f'{label}: ERROR {e}')

measure('wikimedia-dumps', 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles1.xml-p1p187712.bz2')
measure('wikipedia-api', 'https://zh.wikipedia.org/w/api.php?action=query&format=json&list=random&rnlimit=5')
measure('wikipedia-rest', 'https://zh.wikipedia.org/api/rest_v1/page/random/summary')
