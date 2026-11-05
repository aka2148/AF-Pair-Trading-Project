"""Build a local wide adjusted-close CSV from a pinned public research dataset.

The source license allows academic research but forbids redistribution. Keep the
generated CSV local; point readers to the upstream repository and this recipe.
"""

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests
import truststore


COMMIT = "bbe90ca7fb2a3d1ceb47ccaa158e12e81b9cfb40"
REPO = "https://github.com/aswinv90/indian-market-historical-data"
RAW = f"https://raw.githubusercontent.com/aswinv90/indian-market-historical-data/{COMMIT}/data/NSE"
SYMBOLS = """RELIANCE TCS HDFCBANK INFY ICICIBANK HINDUNILVR ITC SBIN BHARTIARTL
KOTAKBANK LT AXISBANK ASIANPAINT MARUTI SUNPHARMA TITAN ULTRACEMCO BAJFINANCE
NESTLEIND WIPRO HCLTECH POWERGRID NTPC TECHM ONGC COALINDIA JSWSTEEL TATASTEEL
GRASIM ADANIPORTS HINDALCO DRREDDY CIPLA DIVISLAB EICHERMOT HEROMOTOCO BRITANNIA
INDUSINDBK BPCL IOC SHREECEM TATACONSUM GAIL VEDL AMBUJACEM DABUR GODREJCP
PIDILITIND HAVELLS SIEMENS LUPIN AUROPHARMA BERGEPAINT COLPAL MARICO BIOCON
MOTHERSON""".split()
OUT = Path("data/public_nse_adjusted_2012_2025.csv")


def main():
    truststore.inject_into_ssl()
    session = requests.Session()
    columns, file_hashes, unavailable = {}, {}, []
    for symbol in SYMBOLS:
        url = f"{RAW}/{symbol}_NS.parquet"
        response = session.get(url, timeout=40)
        if response.status_code == 404:
            unavailable.append(symbol)
            continue
        response.raise_for_status()
        blob = response.content
        frame = pd.read_parquet(io.BytesIO(blob))
        if "Adj Close" not in frame.columns:
            raise ValueError(f"Adj Close missing for {symbol}")
        dates = pd.to_datetime(frame.index if frame.index.name == "Date" else frame["Date"])
        dates = dates.tz_localize(None) if dates.tz is not None else dates
        close = pd.Series(pd.to_numeric(frame["Adj Close"], errors="coerce").to_numpy(), index=dates)
        close = close.loc["2012-01-01":"2025-12-31"]
        columns[symbol + ".NS"] = close.loc[~close.index.duplicated(keep="last")]
        file_hashes[symbol] = hashlib.sha256(blob).hexdigest()
        print(f"{symbol}: {len(close)} rows")
    prices = pd.DataFrame(columns).sort_index()
    prices.index.name = "Date"
    if prices.shape[1] < 25 or len(prices) < 2500:
        raise RuntimeError(f"Insufficient coverage: {prices.shape}; unavailable={unavailable}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUT)
    manifest = {"source_repo": REPO, "source_commit": COMMIT,
                "upstream_license": f"{REPO}/blob/{COMMIT}/LICENSE",
                "dataset": str(OUT), "rows": len(prices), "stocks": prices.shape[1],
                "first_date": str(prices.index.min().date()), "last_date": str(prices.index.max().date()),
                "unavailable_symbols": unavailable, "source_file_sha256": file_hashes,
                "csv_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {OUT}: {prices.shape}; unavailable={unavailable}")


if __name__ == "__main__":
    main()
