"""Download and verify `criteo-research-uplift-v2.1.csv.gz` from official publisher URLs.

Attempts:
  1. Primary Criteo AI Lab URL: http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
  2. Publisher-maintained Hugging Face mirror:
     https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz

Verifies exact byte size (311,422,618 bytes) and SHA-256 digest before accepting the download.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sampling import compute_sha256
from src.schema import CRITEO_V21_BYTES, CRITEO_V21_FILENAME, CRITEO_V21_SHA256

PRIMARY_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
MIRROR_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)


def download_with_fallback(dest_path: Path, timeout_sec: int = 120) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        digest = compute_sha256(dest_path)
        if digest == CRITEO_V21_SHA256:
            print(f"[OK] Verified existing archive at {dest_path} (SHA-256 matches official v2.1).")
            return dest_path
        print(f"[WARN] Existing file at {dest_path} failed SHA-256 check; re-downloading.")

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    errors: list[str] = []

    for label, url in [("Primary Criteo AI Lab", PRIMARY_URL), ("Hugging Face Mirror", MIRROR_URL)]:
        print(f"[INFO] Attempting download from {label}: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "criteo-uplift-portfolio/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp, temp_path.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)

            obs_bytes = temp_path.stat().st_size
            obs_sha = compute_sha256(temp_path)
            if obs_bytes != CRITEO_V21_BYTES or obs_sha != CRITEO_V21_SHA256:
                temp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Integrity mismatch from {label}: bytes={obs_bytes} (expected {CRITEO_V21_BYTES}), "
                    f"sha256={obs_sha} (expected {CRITEO_V21_SHA256})"
                )

            temp_path.replace(dest_path)
            print(f"[OK] Downloaded and verified {dest_path} ({obs_bytes:,} bytes, SHA-256 verified).")
            return dest_path
        except (urllib.error.URLError, OSError, ValueError) as exc:
            temp_path.unlink(missing_ok=True)
            msg = f"{label} ({url}) failed: {exc}"
            print(f"[WARN] {msg}")
            errors.append(msg)

    raise RuntimeError(
        "Could not obtain criteo-research-uplift-v2.1.csv.gz from primary or mirror URLs:\n  - "
        + "\n  - ".join(errors)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify Criteo Uplift v2.1 archive.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / CRITEO_V21_FILENAME,
        help="Target path for criteo-research-uplift-v2.1.csv.gz",
    )
    args = parser.parse_args()
    download_with_fallback(args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
