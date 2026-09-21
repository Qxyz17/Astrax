@echo off
cd /d D:\Project\GitHub\Astrax
python tools\wiki\extract_wiki_corpus.py --language en --target-mib 50 > artifacts\wiki_en.log 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> artifacts\wiki_en.log
