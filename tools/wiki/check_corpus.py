import io
p = 'data/wiki_en.txt'
n = 0; total = 0; sample = []
with io.open(p, encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line or line.startswith('#'):
            continue
        total += len(line.encode('utf-8')) + 1
        if n < 2:
            sample.append(line[:150])
        n += 1
print('documents:', n)
print('bytes:', total)
for s in sample:
    print('sample:', s)
