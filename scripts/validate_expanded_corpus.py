"""Fact-checks expanded corpus file existence, page counts, and content hashes."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    """Fails if the qualified corpus no longer contains ten unmodified 90+-page PDFs."""

    manifest = yaml.safe_load(Path("data/corpus/expanded_manifest.yaml").read_text())
    verified = []
    for source in manifest["sources"]:
        path = Path(source["local_path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        pages = int(next(line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("Pages:")))
        if digest != source["sha256"] or pages != source["pages"] or pages < 90:
            raise SystemExit(f"Corpus validation failed: {source['source_id']}")
        verified.append({"source_id": source["source_id"], "pages": pages})
    if len(verified) < 10:
        raise SystemExit("At least ten qualifying PDFs are required.")
    print({"qualified_sources": len(verified), "total_pages": sum(item["pages"] for item in verified), "sources": verified})


if __name__ == "__main__":
    main()
