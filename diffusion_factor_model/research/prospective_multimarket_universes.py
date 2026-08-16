"""Materialize revision-pinned December 2023 international index universes.

This script retrieves only historical constituent pages.  It does not request
prices or any post-2023 outcomes.  Every derived manifest records the exact
MediaWiki revision ID, revision timestamp, source HTML hash, ticker conversion,
and rendered CSV hash.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


WIKI_API = "https://en.wikipedia.org/w/api.php"
REVISION_CUTOFF = "2023-12-01T00:00:00Z"
USER_AGENT = "diffusion-factor-research/1.0 (reproducible academic study)"

MARKETS = {
    "asx200": {
        "name": "S&P/ASX 200",
        "page": "S&P/ASX_200",
        "suffix": ".AX",
        "row_bounds": (150, 220),
    },
    "nifty50": {
        "name": "NIFTY 50",
        "page": "NIFTY_50",
        "suffix": ".NS",
        "row_bounds": (45, 55),
    },
    "ibex35": {
        "name": "IBEX 35",
        "page": "IBEX_35",
        "suffix": ".MC",
        "row_bounds": (30, 40),
    },
    "nikkei225": {
        "name": "Nikkei 225",
        "page": "Nikkei_225",
        "suffix": ".T",
        "row_bounds": (210, 235),
    },
    "cac40": {
        "name": "CAC 40",
        "page": "CAC_40",
        "suffix": ".PA",
        "row_bounds": (35, 45),
    },
    "smi": {
        "name": "Swiss Market Index",
        "page": "Swiss_Market_Index",
        "suffix": ".SW",
        "row_bounds": (18, 25),
    },
    "tsx60": {
        "name": "S&P/TSX 60",
        "page": "S&P/TSX_60",
        "suffix": ".TO",
        "internal_dot_replacement": "-",
        "row_bounds": (55, 65),
    },
}

SYMBOL_COLUMNS = (
    "ticker",
    "symbol",
    "code",
    "ticker symbol",
    "stock symbol",
    "epic",
)
NAME_COLUMNS = (
    "company",
    "company name",
    "constituent",
    "security",
    "name",
)


def _column_name(value):
    if isinstance(value, tuple):
        parts = [str(part) for part in value if not str(part).startswith("Unnamed")]
        value = parts[-1] if parts else value[-1]
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _find_column(frame, candidates):
    normalized = {_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _select_constituent_table(tables, row_bounds):
    lower, upper = row_bounds
    eligible = []
    for frame in tables:
        symbol_column = _find_column(frame, SYMBOL_COLUMNS)
        if symbol_column is None or not lower <= len(frame) <= upper:
            continue
        name_column = _find_column(frame, NAME_COLUMNS)
        eligible.append((frame, symbol_column, name_column))
    if len(eligible) != 1:
        schemas = [
            {"rows": len(frame), "columns": [_column_name(c) for c in frame.columns]}
            for frame in tables
        ]
        raise ValueError(
            f"Expected exactly one constituent table, found {len(eligible)}; "
            f"schemas={schemas}"
        )
    return eligible[0]


def _normalize_symbol(value, suffix, internal_dot_replacement=None):
    if pd.isna(value):
        raise ValueError("Constituent symbol is missing.")
    symbol = re.sub(r"\[[^]]*]", "", str(value)).strip()
    symbol = symbol.replace("\xa0", "").replace(" ", "")
    if ":" in symbol:
        symbol = symbol.split(":")[-1]
    symbol = symbol.split("/")[0].strip()
    if not symbol:
        raise ValueError(f"Invalid constituent symbol {value!r}.")
    if symbol.upper().endswith(suffix.upper()):
        base = symbol[: -len(suffix)]
    else:
        base = symbol
    if internal_dot_replacement is not None:
        base = base.replace(".", internal_dot_replacement)
    symbol = f"{base}{suffix}"
    return symbol


def _revision_metadata(session, page, cutoff):
    response = session.get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "revisions",
            "titles": page,
            "rvprop": "ids|timestamp",
            "rvlimit": 1,
            "rvstart": cutoff,
            "rvdir": "older",
            "format": "json",
            "formatversion": 2,
        },
        timeout=60,
    )
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    revisions = pages[0].get("revisions", [])
    if not revisions:
        raise ValueError(f"No revision found for {page!r} before {cutoff}.")
    return revisions[0]


def _fetch_revision(session, page, revision_id):
    url = f"https://en.wikipedia.org/w/index.php?title={page}&oldid={revision_id}"
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return url, response.text


def _render_manifest(html, specification):
    tables = pd.read_html(StringIO(html), keep_default_na=False)
    frame, symbol_column, name_column = _select_constituent_table(
        tables, specification["row_bounds"]
    )
    missing = frame[symbol_column].isna() | (
        frame[symbol_column].astype(str).str.strip() == ""
    )
    excluded_rows = []
    for index, row in frame.loc[missing].iterrows():
        excluded_rows.append(
            {
                "source_row_index": int(index),
                "values": {
                    _column_name(column): (
                        None if pd.isna(value) else str(value)
                    )
                    for column, value in row.items()
                },
            }
        )
    frame = frame.loc[~missing].copy()
    lower, upper = specification["row_bounds"]
    if not lower <= len(frame) <= upper:
        raise ValueError(
            f"Constituent count {len(frame)} after explicit missing-symbol "
            f"exclusions is outside [{lower}, {upper}]."
        )
    symbols = frame[symbol_column].map(
        lambda value: _normalize_symbol(
            value,
            specification["suffix"],
            specification.get("internal_dot_replacement"),
        )
    )
    if name_column is None:
        names = symbols.str.removesuffix(specification["suffix"])
    else:
        names = frame[name_column].astype(str).str.strip()
    output = pd.DataFrame({"Symbol": symbols, "Name": names})
    if output["Symbol"].duplicated().any():
        duplicates = output.loc[output["Symbol"].duplicated(), "Symbol"].tolist()
        raise ValueError(f"Normalized manifest contains duplicates: {duplicates}")
    return output.reset_index(drop=True), excluded_rows


def materialize_market(
    code,
    output_dir,
    cutoff=REVISION_CUTOFF,
    session=None,
    allow_overwrite=False,
):
    if code not in MARKETS:
        raise ValueError(f"Unknown market {code!r}; choose {sorted(MARKETS)}.")
    specification = MARKETS[code]
    output_dir = Path(output_dir)
    csv_path = output_dir / f"{code}_202312.csv"
    provenance_path = output_dir / f"{code}_202312.provenance.json"
    if (csv_path.exists() or provenance_path.exists()) and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing universe for {code}.")

    session = session or requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    revision = _revision_metadata(session, specification["page"], cutoff)
    source_url, html = _fetch_revision(
        session, specification["page"], revision["revid"]
    )
    manifest, excluded_rows = _render_manifest(html, specification)
    csv_payload = manifest.to_csv(index=False).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_payload)
    provenance = {
        "index": code,
        "index_name": specification["name"],
        "universe_as_of_cutoff_utc": cutoff,
        "source": "English Wikipedia historical revision via MediaWiki API",
        "source_page": specification["page"],
        "source_revision_id": revision["revid"],
        "source_revision_timestamp": revision["timestamp"],
        "source_url": source_url,
        "source_html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "ticker_suffix_rule": specification["suffix"],
        "internal_dot_replacement": specification.get("internal_dot_replacement"),
        "rows": len(manifest),
        "excluded_source_rows_missing_symbol": excluded_rows,
        "manifest_sha256": hashlib.sha256(csv_payload).hexdigest(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_boundary": (
            "Historical membership only; no price or post-2023 outcome requested."
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path, provenance_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("markets", nargs="+", choices=sorted(MARKETS) + ["all"])
    parser.add_argument("--output-dir", default="research_data/universes")
    parser.add_argument("--revision-cutoff", default=REVISION_CUTOFF)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    markets = sorted(MARKETS) if "all" in args.markets else args.markets
    for market in markets:
        paths = materialize_market(
            market,
            args.output_dir,
            cutoff=args.revision_cutoff,
            allow_overwrite=args.allow_overwrite,
        )
        print(f"Materialized {market}: {paths[0]} ({paths[1]})")


if __name__ == "__main__":
    main()
