import urllib.request as u, re, sys
wiki = sys.argv[1]
req = u.Request(f'https://dumps.wikimedia.org/{wiki}/latest/', headers={'User-Agent':'AstraxResearchBot/1.0'})
p = u.urlopen(req, timeout=60).read().decode('utf-8','replace')
names = sorted(set(re.findall(r'href="(' + wiki + r'-latest-pages-articles[^"]*\.bz2)"', p)))
for n in names:
    print(n)
