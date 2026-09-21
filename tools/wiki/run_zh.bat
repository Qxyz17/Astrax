@echo off
cd /d D:\Project\GitHub\Astrax
python tools\wiki\extract_wiki_corpus.py --language zh --target-mib 50 > artifacts\wiki_zh.log 2>&1
