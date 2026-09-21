import sys, io, urllib.request as u, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
for label, url in [
    ('dumps-head', 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2'),
    ('api-zh', 'https://zh.wikipedia.org/w/api.php?action=query&format=json&meta=siteinfo'),
    ('rest-zh', 'https://zh.wikipedia.org/api/rest_v1/page/random/summary'),
]:
    try:
        req = u.Request(url, method='HEAD', headers={'User-Agent':'AstraxBot/1.0'})
        start = time.time()
        r = u.urlopen(req, timeout=15)
        print(f'{label}: {r.status} in {time.time()-start:.1f}s')
    except Exception as e:
        print(f'{label}: {type(e).__name__} {e}')
