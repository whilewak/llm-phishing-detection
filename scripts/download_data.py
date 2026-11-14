#!/usr/bin/env python3
"""Download the Phishing Email Curated Datasets (Champa et al., Zenodo record 8339691, CC-BY-4.0).

Files are checked against the MD5 sums Zenodo publishes and saved to
data/raw/champa2023/ (git-ignored). Already-downloaded files that match are skipped.

    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORD = "8339691"
API = f"https://zenodo.org/api/records/{RECORD}"
# Nazario_5.csv and Nigerian_5.csv are left out: they repeat the Nazario and
# Nigerian phishing and add legitimate mail of unstated origin.
FILES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
    "TREC_05.csv",
    "TREC_06.csv",
    "TREC_07.csv",
]
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "champa2023"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(API, timeout=60) as resp:
        record = json.load(resp)
    by_name = {f["key"]: f for f in record["files"]}

    for name in FILES:
        meta = by_name[name]
        expected = meta["checksum"].removeprefix("md5:")
        dest = args.out / name
        if dest.exists() and md5(dest) == expected:
            print(f"ok    {name}")
            continue
        url = meta["links"]["self"]
        print(f"fetch {name} ({meta['size'] / 2**20:.1f} MB)")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        actual = md5(tmp)
        if actual != expected:
            tmp.unlink()
            print(f"MD5 mismatch for {name}: expected {expected}, got {actual}", file=sys.stderr)
            return 1
        tmp.rename(dest)

    info = {
        "record": RECORD,
        "doi": record.get("doi"),
        "title": record["metadata"]["title"],
        "license": record["metadata"].get("license", {}).get("id"),
        "files": {n: by_name[n]["checksum"] for n in FILES},
    }
    (args.out / "SOURCE.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"done: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
