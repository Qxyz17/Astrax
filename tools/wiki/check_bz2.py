import bz2, pathlib
p = pathlib.Path('data/wiki_cache/enwiki-latest-pages-articles1.xml-p1p41242.bz2')
print('size', p.stat().st_size)
try:
    with bz2.open(p, 'rb') as f:
        data = f.read(1024*1024)
    print('bz2 OK, first read', len(data), 'bytes')
except Exception as e:
    print('bz2 ERROR', type(e).__name__, e)
