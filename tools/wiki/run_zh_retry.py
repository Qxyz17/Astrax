"""Run the wiki extractor in a retry loop until it reports success."""
import subprocess, sys, time, pathlib
root = pathlib.Path(__file__).resolve().parents[2]
for attempt in range(1, 200):
    result = subprocess.run(
        [sys.executable, str(root / 'tools' / 'wiki' / 'extract_wiki_corpus.py'),
         '--language', 'zh', '--target-mib', '50'],
        cwd=str(root), capture_output=True, text=True, encoding='utf-8', errors='replace')
    tail = (result.stdout or '')[-500:] + (result.stderr or '')[-500:]
    with (root / 'artifacts' / 'wiki_zh.log').open('a', encoding='utf-8') as log:
        log.write(f'--- attempt {attempt} rc={result.returncode} ---\n{tail}\n')
    if result.returncode == 0 and 'output=' in (result.stdout or ''):
        break
    time.sleep(5)
