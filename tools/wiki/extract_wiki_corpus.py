"""Download and extract real Wikipedia text into Astrax document corpora.

Downloads numbered pages-articles parts (small, resumable bz2 chunks) one at a
time, extracts clean plain text from each, and stops once the target byte size
is reached. Output is one complete document per line, matching the holistic
document objective.

This is real encyclopedic text, not a synthesized prompt/response table.
"""

from __future__ import annotations

import argparse
import bz2
import pathlib
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "wiki_cache"

USER_AGENT = "AstraxResearchBot/1.0 (local research; contact: astrax@example.com)"

RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_REF = re.compile(r"<ref[^>/]*?/>|<ref.*?</ref>", re.DOTALL | re.IGNORECASE)
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
RE_FILE_LINK = re.compile(r"\[\[[^\]]*?:(.*?)\]\]", re.DOTALL)
RE_WIKILINK = re.compile(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]")
RE_URL = re.compile(r"https?://\S+")
RE_TAG = re.compile(r"<[^>]+>")
RE_APOSTROPHE = re.compile(r"'{2,5}")
RE_WS = re.compile(r"[ \t\u00a0]+")
RE_MULTI_NL = re.compile(r"\n{3,}")
RE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
RE_LATIN = re.compile(r"[A-Za-z]")


def list_parts(wiki: str) -> list[str]:
    index = f"https://dumps.wikimedia.org/{wiki}/latest/"
    request = urllib.request.Request(index, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        page = response.read().decode("utf-8", "replace")
    pattern = rf'href="({wiki}-latest-pages-articles\d+\.xml-p\d+p\d+\.bz2)"'
    names = sorted(set(re.findall(pattern, page)))
    if not names:
        raise RuntimeError(f"no numbered parts found for {wiki}")
    return names


def download_part(index_url: str, name: str) -> pathlib.Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = CACHE / name
    url = index_url + name
    expected = expected_size(index_url, name)
    if expected and destination.exists() and destination.stat().st_size == expected:
        print(f"cached {name}: {expected // (1024*1024)} MiB", file=sys.stderr, flush=True)
        return destination
    attempt = 0
    while True:
        resume_at = destination.stat().st_size if destination.exists() else 0
        if expected and resume_at >= expected:
            break
        headers = {"User-Agent": USER_AGENT}
        if resume_at:
            headers["Range"] = f"bytes={resume_at}-"
        attempt += 1
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                # A server that ignores Range answers 200 with the whole file.
                # In that case restart the file; only 206 may be appended.
                partial = response.status == 206
                mode = "ab" if (resume_at and partial) else "wb"
                with destination.open(mode) as handle:
                    while True:
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        handle.flush()
                size = destination.stat().st_size
                print(f"  {name}: {size // (1024*1024)} MiB (attempt {attempt})",
                      file=sys.stderr, flush=True)
                if expected is None or size >= expected:
                    break
        except Exception as error:
            print(f"  {name}: retry after {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
        time.sleep(min(30, 3 * attempt))
    return destination


def expected_size(index_url: str, name: str) -> int | None:
    try:
        request = urllib.request.Request(index_url + name, method="HEAD",
                                         headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def clean_wikitext(text: str) -> str:
    text = RE_COMMENT.sub(" ", text)
    text = RE_REF.sub(" ", text)
    text = RE_TABLE.sub(" ", text)
    for _ in range(4):
        text = RE_TEMPLATE.sub(" ", text)
    text = RE_FILE_LINK.sub(" ", text)
    text = RE_WIKILINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = RE_URL.sub(" ", text)
    text = RE_TAG.sub(" ", text)
    text = RE_APOSTROPHE.sub("", text)
    lines = [RE_WS.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines
             if line and not line.startswith(("|", "!", "=", "*", "#", ";"))]
    text = "\n".join(lines)
    text = RE_MULTI_NL.sub("\n\n", text)
    return text.strip()


def is_target_language(text: str, language: str) -> bool:
    if len(text) < 200:
        return False
    if language == "zh":
        return len(RE_CJK.findall(text)) > len(RE_LATIN.findall(text)) * 0.15
    return len(RE_LATIN.findall(text)) > 200


def extract_part(dump_path: pathlib.Path, out, language: str,
                 written: int, target_bytes: int) -> tuple[int, int]:
    documents = 0
    with bz2.open(dump_path, "rb") as raw:
        for event, element in ET.iterparse(raw, events=("end",)):
            if not element.tag.endswith("page"):
                continue
            title = element.findtext(".//{*}title") or ""
            ns = element.findtext(".//{*}ns") or "0"
            text = element.findtext(".//{*}revision/{*}text") or ""
            element.clear()
            if ns != "0" or text.lower().startswith("#redirect"):
                continue
            cleaned = clean_wikitext(text)
            if not is_target_language(cleaned, language):
                continue
            document = f"{title} {cleaned}" if title else cleaned
            document = " ".join(document.split())
            out.write(document + "\n")
            written += len(document.encode("utf-8")) + 1
            documents += 1
            if written >= target_bytes:
                break
    return written, documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["zh", "en"], required=True)
    parser.add_argument("--target-mib", type=int, default=100)
    args = parser.parse_args()

    wiki = "zhwiki" if args.language == "zh" else "enwiki"
    index_url = f"https://dumps.wikimedia.org/{wiki}/latest/"
    target_bytes = args.target_mib * 1024 * 1024
    DATA.mkdir(parents=True, exist_ok=True)
    output_path = DATA / f"wiki_{args.language}.txt"

    written = 0
    documents = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(f"# Astrax {args.language} Wikipedia document corpus, UTF-8\n")
        out.write("# One complete document per line; real encyclopedic text.\n")
        for name in list_parts(wiki):
            if written >= target_bytes:
                break
            part = download_part(index_url, name)
            written, added = extract_part(part, out, args.language, written, target_bytes)
            documents += added
            print(f"{args.language}: {written // (1024*1024)} MiB, "
                  f"{documents} docs (after {name})", file=sys.stderr, flush=True)
            part.unlink(missing_ok=True)
    print(f"output={output_path} bytes={written} documents={documents}")


if __name__ == "__main__":
    main()
