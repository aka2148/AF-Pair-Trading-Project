# Dataset notes

This directory contains the 57-stock adjusted-close dataset, its provenance manifest, and the full upstream license notice. The separate 12-stock CSV used for an exploratory compact experiment is not included because its original source could not be verified.

## Reproducible input

The included file is `public_nse_adjusted_2012_2025.csv`. To reproduce it, run `python prepare_public_dataset.py` from the repository root. The script fetches the individual NSE parquet files from the pinned upstream commit and builds:

```text
data/public_nse_adjusted_2012_2025.csv
data/public_nse_adjusted_2012_2025.manifest.json
```

The generated dataset in the saved full-profile run has 3,453 trading-date rows and 57 ticker columns, covering 2012-01-02 to 2025-12-31. The first column is `Date`; each remaining column is a ticker such as `RELIANCE.NS`, and its values are adjusted closing prices. Unavailable observations are blank. The generated manifest records the source commit, coverage, unavailable symbols, and SHA-256 hashes for local reproducibility.

## Source and use terms

- Upstream repository: [Indian Market Historical Data](https://github.com/aswinv90/indian-market-historical-data)
- Pinned source commit: [`bbe90ca7fb2a3d1ceb47ccaa158e12e81b9cfb40`](https://github.com/aswinv90/indian-market-historical-data/tree/bbe90ca7fb2a3d1ceb47ccaa158e12e81b9cfb40)
- Terms: [upstream license at the pinned commit](https://github.com/aswinv90/indian-market-historical-data/blob/bbe90ca7fb2a3d1ceb47ccaa158e12e81b9cfb40/LICENSE)

The license text supplied with this dataset is included in [`LICENSE-upstream.txt`](LICENSE-upstream.txt). It grants copying and use for personal, non-commercial, and educational research. It prohibits commercial use, including redistribution as part of a commercial offering, unless the copyright holder gives prior explicit written permission. The copyright notice, permission notice, and disclaimer must accompany copies or substantial portions. Keep this notice with the CSV when copying or downloading it for research use.

The original 12-stock CSV used for a compact experiment is not included: its source could not be verified. To run the compact profile, provide a CSV for which you have appropriate rights and document its provenance.

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
