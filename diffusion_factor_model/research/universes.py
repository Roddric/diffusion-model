"""Materialize checksum-pinned, pre-confirmation index universes."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


SNAPSHOTS = {
    "csi300": {
        "url": (
            "https://yfiua.github.io/index-constituents/2023/12/"
            "constituents-csi300.csv"
        ),
        "sha256": (
            "fd04cb3a4be9fb6ec08c62d08073a8ae2fe53639004af8457a4813e2087a3490"
        ),
        "rows": 300,
    },
    "sp500": {
        "url": (
            "https://yfiua.github.io/index-constituents/2023/12/"
            "constituents-sp500.csv"
        ),
        "sha256": (
            "3dc7f3f14ce0a82a4fd5c43085059032abdbd48889ff271ea8ecfe7b4b79b502"
        ),
        "rows": 503,
    },
    "ftse100": {
        "url": (
            "https://yfiua.github.io/index-constituents/2023/12/"
            "constituents-ftse100.csv"
        ),
        "sha256": (
            "13d55d86de45153ef68cf630549a1f054998d1a64979868a1dca0e72d04dddcf"
        ),
        "rows": 100,
    },
    "hsi": {
        "url": (
            "https://yfiua.github.io/index-constituents/2023/12/"
            "constituents-hsi.csv"
        ),
        "sha256": (
            "87307877e58f780cdecc58d28c7c7174c977bc9702f20f1e7d63d80cc317de42"
        ),
        "rows": 80,
    },
    "dax": {
        "url": (
            "https://raw.githubusercontent.com/yfiua/index-constituents/"
            "main/docs/2023/12/constituents-dax.csv"
        ),
        "sha256": (
            "71af65f921febd5c5bf60b32fc99e5110d24d56ab393e4122e6c84b6235c277e"
        ),
        "rows": 40,
    },
}


def materialize_snapshot(name, output_dir):
    """Download one immutable archived snapshot and record its provenance."""
    if name not in SNAPSHOTS:
        raise ValueError(f"Unknown snapshot {name!r}; choose {sorted(SNAPSHOTS)}.")
    metadata = SNAPSHOTS[name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}_202312.csv"
    provenance_path = output_dir / f"{name}_202312.provenance.json"

    payload = urlopen(metadata["url"], timeout=60).read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError(
            f"Checksum mismatch for {name}: expected {metadata['sha256']}, "
            f"received {digest}."
        )

    temporary = csv_path.with_suffix(".csv.tmp")
    temporary.write_bytes(payload)
    frame = pd.read_csv(temporary)
    if list(frame.columns) != ["Symbol", "Name"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Unexpected columns for {name}: {list(frame.columns)}")
    if len(frame) != metadata["rows"] or not frame["Symbol"].is_unique:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Unexpected {name} universe: {len(frame)} rows, "
            f"{frame['Symbol'].nunique()} unique symbols."
        )
    temporary.replace(csv_path)

    provenance = {
        "index": name,
        "archive_month": "2023-12",
        "source_url": metadata["url"],
        "source_repository": "https://github.com/yfiua/index-constituents",
        "sha256": digest,
        "rows": len(frame),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Universe frozen before the 2024-01-01 confirmation boundary."
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path, provenance_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "indices", nargs="+", choices=sorted(SNAPSHOTS) + ["all"]
    )
    parser.add_argument(
        "--output-dir", default="./research_data/universes"
    )
    args = parser.parse_args()
    names = sorted(SNAPSHOTS) if "all" in args.indices else args.indices
    for name in names:
        paths = materialize_snapshot(name, args.output_dir)
        print(f"Materialized {name}: {paths[0]} ({paths[1]})")


if __name__ == "__main__":
    main()
