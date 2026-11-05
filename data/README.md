# Dataset notes

This directory contains the dataset preparation recipe but not the market-price files. Generated CSVs and manifests are kept local and ignored by Git.

## Reproducible input

Run `python prepare_public_dataset.py` from the repository root. The script fetches the individual NSE parquet files from the pinned upstream commit and builds:

```text
data/public_nse_adjusted_2012_2025.csv
data/public_nse_adjusted_2012_2025.manifest.json
```

The generated dataset in the saved full-profile run has 3,453 trading-date rows and 57 ticker columns, covering 2012-01-02 to 2025-12-31. The first column is `Date`; each remaining column is a ticker such as `RELIANCE.NS`, and its values are adjusted closing prices. Unavailable observations are blank. The generated manifest records the source commit, coverage, unavailable symbols, and SHA-256 hashes for local reproducibility.

## Source and use terms

- Upstream repository: [Indian Market Historical Data](https://github.com/aswinv90/indian-market-historical-data)
- Pinned source commit: [`bbe90ca7fb2a3d1ceb47ccaa158e12e81b9cfb40`](https://github.com/aswinv90/indian-market-historical-data/tree/bbe90ca7fb2a3d1ceb47ccaa158e12e81b9cfb40)
- Terms: [upstream license at the pinned commit](https://github.com/aswinv90/indian-market-historical-data/blob/bbe90ca7fb2a3d1ceb47ccaa158e12e81b9cfb40/LICENSE)

The upstream project describes its dataset terms as non-commercial research and personal use and restricts redistribution. This repository therefore publishes the acquisition script and schema notes, not the downloaded CSV or source parquet files. Check the upstream terms directly and request permission from the data owner if your planned use requires rights not granted there.

The original 12-stock CSV used for a compact experiment is not included either: its source could not be verified. To run the compact profile, provide a CSV for which you have appropriate rights and document its provenance.

## Use a different CSV

The pipeline accepts a wide CSV with a `Date` column and one adjusted-close column per stock, for example:

```csv
Date,RELIANCE.NS,TCS.NS
2024-01-02,2584.85,3793.20
2024-01-03,2599.10,3811.45
```

Set `DATA_SOURCE='csv'` and `PRICE_CSV` to the file path. Do not commit a third-party dataset unless its license permits public redistribution.

## Research caveat

The selected 57-stock universe is a fixed list of present-day names, not a point-in-time index membership history. Backtests using it are subject to survivorship bias. The pipeline accepts historical membership snapshots through `MEMBERSHIP_CSV`, but this repository does not provide them.
