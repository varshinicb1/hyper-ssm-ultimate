"""
prepare_c_corpus.py — Build an embedded C training corpus
==========================================================
Downloads targeted .c/.h from public MCU repos, assembles into c_corpus.txt.

Sources: FreeRTOS, STM32 HAL, ESP-IDF, Zephyr samples, NXP i.MX RT

Usage:  python data/prepare_c_corpus.py
"""

import os, re, zipfile, shutil
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(DATA_DIR, "c_corpus.txt")
TMP_DIR  = os.path.join(DATA_DIR, "_tmp")

# Targeted repos — smaller zips where possible
SOURCES = [
    {
        "name": "freertos_kernel",
        "url": "https://github.com/FreeRTOS/FreeRTOS-Kernel/archive/refs/heads/main.zip",
        "prefix": [],  # accept all .c/.h
    },
    {
        "name": "esp_idf",
        "url": "https://github.com/espressif/esp-idf/archive/refs/heads/master.zip",
        "prefix": ["examples/peripherals/", "components/driver/", "components/hal/"],
    },
    {
        "name": "nxp_imxrt",
        "url": "https://github.com/platformio/platform-nxpimxrt/archive/refs/heads/develop.zip",
        "prefix": [],
    },
]


def download(name: str, url: str) -> str:
    zip_path = os.path.join(TMP_DIR, f"{name}.zip")
    if os.path.exists(zip_path):
        size = os.path.getsize(zip_path) / 1e6
        print(f"  [CACHED] {name} ({size:.1f} MB)")
        return zip_path
    print(f"  [DOWNLOAD] {name} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, zip_path)
        size = os.path.getsize(zip_path) / 1e6
        print(f"OK ({size:.1f} MB)")
    except Exception as e:
        print(f"FAILED: {e}")
        return None
    return zip_path


def extract_c_files(zip_path: str, prefixes: list[str]) -> list[str]:
    """Extract .c/.h matching any prefix substring in the archive path."""
    texts = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not (info.filename.endswith(".c") or info.filename.endswith(".h")):
                continue

            # if prefixes specified, check if any prefix is inside the filepath
            if prefixes and not any(p in info.filename for p in prefixes):
                continue

            try:
                raw = zf.read(info.filename).decode("utf-8", errors="ignore")
                cleaned = _clean(raw, info.filename)
                if len(cleaned.strip()) > 80:
                    texts.append(cleaned)
            except Exception:
                continue
    return texts


def _clean(code: str, path: str) -> str:
    # strip leading block-comment license
    code = re.sub(r"^/\*.*?\*/", "", code, count=1, flags=re.DOTALL)
    code = re.sub(r"\n{3,}", "\n\n", code)
    name = os.path.basename(path)
    return f"<|file|>{name}\n{code.strip()}\n<|eof|>\n"


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    all_code = []

    for src in SOURCES:
        print(f"\n── {src['name']} ──")
        zip_path = download(src["name"], src["url"])
        if zip_path is None:
            continue
        files = extract_c_files(zip_path, src["prefix"])
        print(f"  Extracted {len(files)} C/H files")
        all_code.extend(files)

    # write
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_code))

    mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\n{'='*50}")
    print(f"Corpus: {OUT_FILE}")
    print(f"Files:  {len(all_code)}")
    print(f"Size:   {mb:.2f} MB")
    print(f"{'='*50}")

    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
