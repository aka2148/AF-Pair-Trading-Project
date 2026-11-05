# %% [markdown]
# # Adaptive Pairs Trading on Indian Stocks (NSE)
# **Pipeline:** NSE prices → (clustering) → cointegration → rolling spread / z-score →
# **LSTM forecast** + **Markov state probabilities** → **PPO** position sizing → walk-forward backtest.
#
# Design rules (so the results are honest):
# * Everything is **walk-forward**. Pairs for year *Y* are chosen using only the 2 years before *Y*.
#   The full profile evaluates 50 pairs/year; the compact 12-stock profile evaluates 10. Exploratory fills are reported separately.
#   The LSTM and Markov matrix used in year *Y* are fitted only on data before *Y*.
# * Full-profile split: **train 2017-2020 / validation 2021-2023 / test 2024-latest**.
#   The compact profile uses train 2022-2023 / validation 2024 / test 2025.
# * Positions are decided at the close of day *t* and earn the return from *t* to *t+1*. Transaction costs are charged on every change in position.
# * Components are added one at a time (baseline → clustering → LSTM → Markov → PPO) so you can see what each one actually contributes.
#
# **Limitations you should state in your report**
# * Default universe = current large-cap NSE stocks → survivorship bias. Supply historical snapshots with `MEMBERSHIP_CSV` to use point-in-time constituents.
# * Indian cash-market short selling is intraday only; overnight short legs require single-stock futures. Costs are a configurable two-leg turnover estimate; futures basis, margin, taxes and market impact need local calibration.
# * Results are whatever this data gives. The numbers reported in the literature review are not targets.
#
# ## Run this in Kaggle
# 1. Create a Kaggle notebook and turn **Internet on** to download adjusted daily NSE prices with yfinance.
# 2. If Internet is off, attach a CSV named `prices.csv`: first column `Date`, remaining columns NSE tickers such as `RELIANCE.NS`.
#    Set `DATA_SOURCE = "csv"` and point `PRICE_CSV` to that input file. The price history should begin by 2012.
# 3. Run all cells. The all-pairs backtest selects 50 pairs/year in full mode or 10 pairs/year in compact mode.
# 4. For point-in-time membership, attach `constituents.csv` with `snapshot_date,ticker` columns and set `MEMBERSHIP_CSV`.
#
# For the supplied 12-stock CSV, set `PROFILE=compact`, `DATA_SOURCE=csv`, and `PRICE_CSV` to its path.
# `FAST=True` lowers PPO/LSTM training time but keeps pair discovery and the chosen basket size intact.
# `synthetic` is only a code smoke test.
# Results are written to `/kaggle/working`; download the results table, selected-pair ledger and plot from the Kaggle Output panel.

# %%
import os, sys, copy, time, itertools, random, warnings, hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch, torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, os.cpu_count() or 1))

# %% [markdown]
# ## 0. Configuration
# Change `DATA_SOURCE`, the dates, and the cost here. Everything else follows.

# %%
FAST = os.environ.get("FAST", "0") == "1"          # fewer training steps; pair discovery is unchanged
PROFILE = os.environ.get("PROFILE", "full")       # full: 50 pairs; compact: supplied 12-stock CSV
if PROFILE not in {"full", "compact"}:
    raise ValueError("PROFILE must be 'full' or 'compact'")
DATA_SOURCE = os.environ.get("DATA_SOURCE", "yfinance")   # "yfinance" | "csv" | "synthetic"
PRICE_CSV = os.environ.get("PRICE_CSV", "data/indian_stock_prices.csv")
MEMBERSHIP_CSV = os.environ.get("MEMBERSHIP_CSV", "")
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

START, END = "2012-01-01", os.environ.get("END_DATE") or None
LOOKBACK_YEARS = 2          # pair selection window before each trading year
FIRST_ACTIVE_YEAR = 2014    # first year with pairs (needs 2 years of history before it)
FIRST_ML_YEAR = 2017        # first year for which LSTM / Markov forecasts exist
TRAIN_YEARS = list(range(2017, 2021))
RULE_TUNE_YEARS = [2019, 2020]  # inner tuning window; validation remains untouched
VAL_YEARS = [2021, 2022, 2023]
TEST_START_YEAR = 2024      # test = TEST_START_YEAR .. last year in data

TOP_K = 50                  # full-universe annual basket
MAX_PER_STOCK = 4           # cap overlap in pair selection
MIN_TEST_PAIRS = TOP_K
FDR_Q = 0.10                # report BH-adjusted significance; exploratory fills are labeled
COINT_P = 0.05
HL_MIN, HL_MAX = 3, 60      # half-life filter (days)
ROLL_WIN = 126              # rolling regression / z-score window (days)
COST_BPS = 20.0             # assumed cost per unit of combined two-leg gross turnover, in bps
MAX_STOCK_GROSS_WEIGHT = 0.05  # aggregate daily gross exposure cap per stock
ANNUAL_SHORT_CARRY_BPS = 0.0   # set only for cash shorts; futures basis/margin are not modeled here
H = 5                       # LSTM forecast horizon (days)
SEQ_LEN = 20
PPO_SEEDS = [0, 1, 2] if not FAST else [0]
PPO_TOTAL_STEPS = 150_000 if not FAST else 12_000
PPO_EVAL_EVERY = 25_000 if not FAST else 6_000
LSTM_EPOCHS = 15 if not FAST else 3
GRID = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])   # PPO action -> target position in the spread
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

if PROFILE == "compact":
    # With 12 stocks, 50 pairs and the original 2017 training start are impossible.
    FIRST_ACTIVE_YEAR = 2020
    FIRST_ML_YEAR = 2022
    TRAIN_YEARS = [2022, 2023]
    RULE_TUNE_YEARS = [2022, 2023]
    VAL_YEARS = [2024]
    TEST_START_YEAR = 2025
    TOP_K = 10
    MIN_TEST_PAIRS = TOP_K
print(f"Profile={PROFILE} | FAST={FAST} | source={DATA_SOURCE} | target pairs/year={TOP_K}")

TICKERS = """RELIANCE TCS HDFCBANK INFY ICICIBANK HINDUNILVR ITC SBIN BHARTIARTL KOTAKBANK LT AXISBANK ASIANPAINT
MARUTI SUNPHARMA TITAN ULTRACEMCO BAJFINANCE NESTLEIND WIPRO HCLTECH POWERGRID NTPC TECHM M&M ONGC COALINDIA
JSWSTEEL TATASTEEL GRASIM ADANIPORTS HINDALCO BAJAJ-AUTO DRREDDY CIPLA DIVISLAB EICHERMOT HEROMOTOCO BRITANNIA
INDUSINDBK BPCL IOC SHREECEM TATACONSUM BAJAJFINSV UPL PNB BANKBARODA GAIL VEDL AMBUJACEM DABUR GODREJCP
PIDILITIND HAVELLS SIEMENS LUPIN AUROPHARMA BERGEPAINT COLPAL MARICO BIOCON MOTHERSON""".split()
TICKERS = [t + ".NS" for t in TICKERS]

# %% [markdown]
# ## 1. Data
# `yfinance` needs internet (on Kaggle: Settings → Internet → On). If you can't enable it, upload a CSV of daily adjusted
# closes and set `DATA_SOURCE = "csv"`. There is **no silent fallback to fake data**. The default list contains current NSE large-cap names;
# use historical constituent snapshots to reduce survivorship bias.

# %%
def make_synthetic(n=60, seed=0):
    """Fake market with a few cointegrated pairs (some of which break). ONLY for testing that the code runs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", "2026-10-16")
    T = len(dates)
    mkt = np.cumsum(rng.normal(0.0004, 0.010, T))
    sec = np.cumsum(rng.normal(0, 0.006, (T, 4)), axis=0)
    lp = {}
    for i in range(n):
        lp[i] = 4 + rng.uniform(0.5, 1.5) * mkt + rng.uniform(0.5, 1.5) * sec[:, i % 4] + np.cumsum(rng.normal(0, 0.012, T))
    for k in range(0, 20, 2):          # build cointegrated pairs
        i, j = k, k + 1
        ou = np.zeros(T)
        for t in range(1, T):
            ou[t] = ou[t - 1] * (1 - 0.06) + rng.normal(0, 0.012)
        broken_at = rng.integers(int(T * 0.4), T) if k % 4 == 0 else T + 1
        rw = np.cumsum(rng.normal(0, 0.012, T))
        spread = np.where(np.arange(T) < broken_at, ou, ou[min(broken_at, T - 1)] + rw - rw[min(broken_at, T - 1)])
        lp[j] = 0.3 + rng.uniform(0.6, 1.4) * lp[i] + spread
    px = pd.DataFrame({f"S{i:02d}.NS": np.exp(v) for i, v in lp.items()}, index=dates)
    px.iloc[:900, -3:] = np.nan        # a few late listings
    return px

def load_prices():
    if DATA_SOURCE == "yfinance":
        import yfinance as yf
        raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
        if raw is None or len(raw) == 0:
            raise RuntimeError("Download returned nothing. On Kaggle enable Internet in Settings, or use DATA_SOURCE='csv'.")
    elif DATA_SOURCE == "csv":
        raw = pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)
    elif DATA_SOURCE == "synthetic":
        print("!!! SYNTHETIC DATA - results below say NOTHING about real markets !!!")
        raw = make_synthetic()
    else:
        raise ValueError(DATA_SOURCE)
    return raw

raw = load_prices()
px = raw.sort_index().loc[START:END]
px = px.loc[~px.index.duplicated(keep="last")]
px = px.apply(pd.to_numeric, errors="coerce")
px = px.loc[:, ~px.columns.duplicated()]
px = px.where(px > 0).dropna(how="all").ffill(limit=5)
if px.empty:
    raise ValueError("The price dataset contains no positive observations")
if PROFILE == "compact" and px.shape[1] < 12:
    raise ValueError("The compact profile requires the supplied 12-stock dataset")
if PROFILE == "full" and px.shape[1] < 25:
    raise ValueError("The full profile needs at least 25 stocks; use PROFILE=compact for the supplied 12-stock CSV")
jumps = (px.pct_change().abs() > 0.25).sum()
print("Stocks:", px.shape[1], "| days:", px.shape[0], "|", px.index[0].date(), "->", px.index[-1].date())
print("Stocks with >25% 1-day moves (check for bad data):", jumps[jumps > 0].to_dict())
lp = np.log(px)
IDX = px.index
LAST_YEAR = IDX[-1].year
TEST_YEARS = list(range(TEST_START_YEAR, LAST_YEAR + 1))
YEAR_POS = {y: (int(np.argmax(IDX.year == y)), int(len(IDX) - np.argmax(IDX.year[::-1] == y))) for y in sorted(set(IDX.year))}
print("Train", TRAIN_YEARS, "| Val", VAL_YEARS, "| Test", TEST_YEARS)

MEMBERSHIP = None
if MEMBERSHIP_CSV:
    MEMBERSHIP = pd.read_csv(MEMBERSHIP_CSV, parse_dates=["snapshot_date"])
    required = {"snapshot_date", "ticker"}
    if not required.issubset(MEMBERSHIP.columns):
        raise ValueError("MEMBERSHIP_CSV needs snapshot_date and ticker columns")
    MEMBERSHIP = MEMBERSHIP.dropna(subset=["snapshot_date", "ticker"]).copy()
    MEMBERSHIP["ticker"] = MEMBERSHIP["ticker"].astype(str).str.upper().str.strip()
    MEMBERSHIP["ticker"] = MEMBERSHIP["ticker"].map(lambda t: t if t.endswith(".NS") else t + ".NS")
    MEMBERSHIP = MEMBERSHIP.sort_values("snapshot_date")
    print("Historical constituent snapshots loaded:", MEMBERSHIP.snapshot_date.min().date(), "to", MEMBERSHIP.snapshot_date.max().date())
else:
    print("Universe note: using the current ticker list (survivorship bias). Set MEMBERSHIP_CSV to historical snapshots to reduce it.")

def constituents_at(date):
    if MEMBERSHIP is None:
        return None
    prior = MEMBERSHIP[MEMBERSHIP.snapshot_date <= pd.Timestamp(date)]
    if prior.empty:
        raise ValueError(f"No constituent snapshot exists on or before {pd.Timestamp(date).date()}")
    snap = prior.snapshot_date.max()
    return set(MEMBERSHIP.loc[MEMBERSHIP.snapshot_date == snap, "ticker"])

# %% [markdown]
# ## 2. Pair selection (walk-forward)
# For each trading year *Y* we use the previous 2 years only. **Baseline** = test every pair of stocks.
# **Clustered** = PCA of returns → KMeans, then test only pairs inside each cluster.
# Both Engle-Granger orientations are tested; p-values are orientation-adjusted and BH-FDR corrected across candidates.
# The full all-pairs experiment ranks 50 pairs/year; compact ranks 10. Candidates that fail FDR,
# the half-life screen or the hedge-ratio screen are labeled exploratory.

# %%
def cluster_stocks(lpw, n_pcs=5):
    r = lpw.diff().dropna()
    r = (r - r.mean()) / r.std()
    load = PCA(n_components=min(n_pcs, r.shape[1] - 1), random_state=0).fit(r.values).components_.T
    load = (load - load.mean(0)) / (load.std(0) + 1e-12)
    best, best_s = None, -1
    for k in range(3, max(4, min(10, len(load) // 4)) + 1):
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(load)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(load, lab)
        if s > best_s:
            best, best_s = lab, s
    groups = {}
    for c, lab in zip(lpw.columns, best):
        groups.setdefault(int(lab), []).append(c)
    return groups

def ols_beta_hl(y, x):
    X = np.column_stack([np.ones_like(x), x])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    s = y - a - b * x
    ds, sl = np.diff(s), s[:-1] - s[:-1].mean()
    lam = (sl @ (ds - ds.mean())) / (sl @ sl)
    hl = -np.log(2) / np.log(1 + lam) if -1 < lam < 0 else np.inf
    return b, hl

def eg_p(a, b):
    return coint(a, b, trend="c", maxlag=1, autolag=None)[1]

def select_pairs(lpw, method):
    cols = [c for c in lpw.columns if lpw[c].notna().all()]
    lpw = lpw[cols]
    groups = cluster_stocks(lpw) if method == "clustered" else {0: cols}
    cands = [(a, b) for g in groups.values() for a, b in itertools.combinations(g, 2)]
    rows = []
    for a, b in cands:
        A, B = lpw[a].values, lpw[b].values
        p1, p2 = eg_p(A, B), eg_p(B, A)
        y, x, p = (a, b, p1) if p1 <= p2 else (b, a, p2)
        beta, hl = ols_beta_hl(lpw[y].values, lpw[x].values)
        rows.append((y, x, p, hl, beta))
    df = pd.DataFrame(rows, columns=["y", "x", "p", "hl", "beta"])
    # Correct for choosing the better of two Engle-Granger orientations, then
    # control false discoveries across the full candidate set for this year.
    df["p_oriented"] = np.minimum(1.0, 2.0 * df["p"])
    df["p_adj"] = multipletests(df["p_oriented"].fillna(1.0), method="fdr_bh")[1]
    df["fdr_significant"] = df["p_adj"] <= FDR_Q
    qualified = df[df.fdr_significant & df.hl.between(HL_MIN, HL_MAX) & (df.beta > 0.1)].copy()
    qualified["selection_tier"] = "fdr_screen_pass"
# Always evaluate the configured number of distinct candidates when the price
# universe permits. Any fill beyond the statistical screen is explicitly
    # labeled exploratory and remains visible in the outputs.
    pool = qualified.sort_values(["p_adj", "p"])
    # Include exploratory candidates after every screened candidate. The stock
    # overlap cap may otherwise leave the basket short even when len(pool)>=TOP_K.
    fill = df[(df.beta > 0.1) & ~df.index.isin(pool.index)].copy()
    fill["selection_tier"] = "exploratory_fill"
    pool = pd.concat([pool, fill.sort_values(["p_adj", "p"])])
    ok = pool
    pick, cnt = [], {}
    for r in ok.itertuples():
        if cnt.get(r.y, 0) < MAX_PER_STOCK and cnt.get(r.x, 0) < MAX_PER_STOCK:
            pick.append(dict(y=r.y, x=r.x, p=r.p, p_adj=r.p_adj,
                             fdr_significant=bool(r.fdr_significant),
                             selection_tier=r.selection_tier, hl=r.hl, beta=r.beta))
            cnt[r.y] = cnt.get(r.y, 0) + 1; cnt[r.x] = cnt.get(r.x, 0) + 1
        if len(pick) == TOP_K:
            break
    if method == "all" and len(pick) < MIN_TEST_PAIRS:
        raise RuntimeError(f"Only {len(pick)} usable pairs found; at least {MIN_TEST_PAIRS} are required. "
                           "Add more complete-history NSE stocks or inspect the price CSV.")
    return pick, len(cands), int(len(qualified))

SEL, SELSTATS = {"all": {}, "clustered": {}}, []
t0 = time.time()
years_active = [y for y in range(FIRST_ACTIVE_YEAR, LAST_YEAR + 1) if y in YEAR_POS]
for Y in years_active:
    i0 = YEAR_POS[Y][0]
    w_end = IDX[i0 - 1]
    lpw = lp.iloc[max(0, i0 - LOOKBACK_YEARS * 252): i0]
    members = constituents_at(w_end)
    if members is not None:
        lpw = lpw.loc[:, [c for c in lpw.columns if c in members]]
    if len(lpw) < LOOKBACK_YEARS * 252 * 0.9:
        continue
    minimum_stocks = 12 if PROFILE == "compact" else 25
    if lpw.shape[1] < minimum_stocks:
        raise RuntimeError(f"Only {lpw.shape[1]} eligible stocks are available for {Y}; at least {minimum_stocks} are required for the {PROFILE} profile.")
    for m in ["all", "clustered"]:
        pick, ncand, npass = select_pairs(lpw, m)
        SEL[m][Y] = pick
        SELSTATS.append(dict(year=Y, method=m, candidates=ncand, passed=npass, selected=len(pick),
                             fdr_selected=sum(p["fdr_significant"] for p in pick),
                             exploratory_fill=sum(p["selection_tier"] == "exploratory_fill" for p in pick)))
    print(f"{Y}: all={SELSTATS[-2]['candidates']} cands -> {SELSTATS[-2]['selected']} | clustered={SELSTATS[-1]['candidates']} cands -> {SELSTATS[-1]['selected']}  ({time.time()-t0:.0f}s)")
SELSTATS = pd.DataFrame(SELSTATS)
all_counts = SELSTATS[SELSTATS.method == "all"].set_index("year").selected
if all_counts.empty or (all_counts < MIN_TEST_PAIRS).any():
    raise RuntimeError(f"The all-pairs selector produced fewer than {MIN_TEST_PAIRS} pairs in at least one year, or no eligible years.")
print("Annual all-pairs basket size:", all_counts.to_dict())
print("Selected pairs passing FDR q<=%.2f by year:" % FDR_Q,
      SELSTATS[SELSTATS.method == "all"].set_index("year").fdr_selected.to_dict())
if (SELSTATS[SELSTATS.method == "all"].fdr_selected == 0).all():
    print("WARNING: No all-pairs selections pass FDR. Treat their trading results as exploratory, not validated cointegration evidence.")

# %% [markdown]
# ## 3. Spread, z-score and per-pair features
# Hedge ratio and z-score come from a **rolling regression** (window `ROLL_WIN`), so they adapt and only use past data.
# `sr_next[t]` = return of the market-neutral spread from close *t* to close *t+1* per unit of gross capital.

# %%
def rolling_reg(ly, lx, win):
    ly = ly - ly.dropna().iloc[0]; lx = lx - lx.dropna().iloc[0]
    mx, my = lx.rolling(win).mean(), ly.rolling(win).mean()
    vx = (lx ** 2).rolling(win).mean() - mx ** 2
    vy = (ly ** 2).rolling(win).mean() - my ** 2
    cxy = (lx * ly).rolling(win).mean() - mx * my
    beta = cxy / vx.clip(lower=1e-10)
    alpha = my - beta * mx
    rv = (vy - beta * cxy).clip(lower=1e-10)
    z = (ly - alpha - beta * lx) / np.sqrt(rv)
    return beta, z

PF = {}
def pair_frame(y, x):
    if (y, x) in PF:
        return PF[(y, x)]
    beta, z = rolling_reg(lp[y], lp[x], ROLL_WIN)
    Ry, Rx = px[y].pct_change(), px[x].pct_change()
    sr_next = (Ry.shift(-1) - beta * Rx.shift(-1)) / (1 + beta.abs())
    sr_prev = sr_next.shift(1)
    vol = sr_prev.rolling(60, min_periods=30).std()
    df = pd.DataFrame({"z": z, "dz1": z.diff(), "dz5": z.diff(5), "beta": beta,
                       "sr_next": sr_next, "sr_prev": sr_prev, "vol": vol})
    df["target"] = (sr_next[::-1].rolling(H).sum()[::-1]) / (vol * np.sqrt(H))   # uses FUTURE data: only ever a training label
    df["lstm"] = np.nan; df["mk_dz"] = np.nan; df["mk_center"] = np.nan; df["mk_ext"] = np.nan
    PF[(y, x)] = df
    return df

PAIR_YEARS = {}   # (y,x) -> set of years in which some method trades it
for m in SEL:
    for Y, picks in SEL[m].items():
        for r in picks:
            pair_frame(r["y"], r["x"])
            PAIR_YEARS.setdefault((r["y"], r["x"]), set()).add(Y)
UNION = sorted({(p, Y) for p, ys in PAIR_YEARS.items() for Y in ys}, key=lambda t: (t[1], t[0]))
print("distinct pairs:", len(PF), "| pair-years:", len(UNION))

# %% [markdown]
# ## 4. Backtester and the rule-based baseline
# A *policy* takes a list of segments (one pair × one year) and returns a position series in [-1, 1] for each.
# Net P&L includes both hedge legs' turnover and optional short carry; the portfolio applies a 5% per-stock gross exposure cap.
# Positions close at year-end. Portfolio return = sum over active pairs / `TOP_K`.

# %%
def seg_arrays(method, Y, r):
    df = PF[(r["y"], r["x"])]
    i0, i1 = YEAR_POS[Y]
    s = df.iloc[i0:i1]
    return dict(method=method, year=Y, pair=(r["y"], r["x"]), dates=s.index,
                z=np.nan_to_num(s.z.values), dz1=np.nan_to_num(s.dz1.values), dz5=np.nan_to_num(s.dz5.values),
                lstm=np.nan_to_num(s.lstm.values), mk_dz=np.nan_to_num(s.mk_dz.values),
                mk_center=np.nan_to_num(s.mk_center.values), mk_ext=np.nan_to_num(s.mk_ext.values),
                sr_next=np.nan_to_num(s.sr_next.values), beta=np.nan_to_num(s.beta.values, nan=1.0))

def build_segments():
    return {m: [seg_arrays(m, Y, r) for Y, picks in SEL[m].items() for r in picks] for m in SEL}

def zscore_policy(entry=2.0, exit_=0.5, stop=4.0):
    def pol(segs):
        out = []
        for s in segs:
            z = s["z"]; pos = np.zeros(len(z)); p = 0; blocked = False
            for t in range(len(z) - 1):
                if blocked and abs(z[t]) < entry:
                    blocked = False
                if p == 0:
                    if not blocked:
                        if z[t] > entry: p = -1
                        elif z[t] < -entry: p = 1
                else:
                    if abs(z[t]) > stop: p, blocked = 0, True
                    elif (p == -1 and z[t] < exit_) or (p == 1 and z[t] > -exit_): p = 0
                pos[t] = p
            out.append(pos)
        return out
    return pol

def cap_one_day(segs, rows):
    """Scale (segment index, day index, target) rows to the per-stock gross cap."""
    targets = [float(p) for _, _, p in rows]
    for _ in range(6):
        gross = {}; pairs = []
        for (i, t, _), p in zip(rows, targets):
            s = segs[i]; b = float(s["beta"][t]); denom = 1.0 + abs(b)
            wy, wx = 1.0 / denom, -b / denom
            pairs.append((s["pair"], p * wy / TOP_K, p * wx / TOP_K))
            gross[s["pair"][0]] = gross.get(s["pair"][0], 0.0) + abs(p * wy / TOP_K)
            gross[s["pair"][1]] = gross.get(s["pair"][1], 0.0) + abs(p * wx / TOP_K)
        over = {ticker: MAX_STOCK_GROSS_WEIGHT / amount for ticker, amount in gross.items()
                if amount > MAX_STOCK_GROSS_WEIGHT}
        if not over:
            break
        for j, (pair, _, _) in enumerate(pairs):
            targets[j] *= min(over.get(pair[0], 1.0), over.get(pair[1], 1.0))
    return targets

def cap_pair_positions(segs, positions):
    """Scale pair targets to enforce a daily per-stock gross exposure cap."""
    pos = [np.asarray(p, float).copy() for p in positions]
    date_rows = {}
    for i, s in enumerate(segs):
        for t, date in enumerate(s["dates"]):
            date_rows.setdefault(date, []).append((i, t))
    for rows in date_rows.values():
        raw = [(i, t, pos[i][t]) for i, t in rows]
        capped = cap_one_day(segs, raw)
        for (i, t), p in zip(rows, capped):
            pos[i][t] = p
    return pos

def backtest(segs, policy, years):
    segs = [s for s in segs if s["year"] in years]
    poss = cap_pair_positions(segs, policy(segs))
    parts, turn, entries, held = [], 0.0, 0, 0
    for s, pos in zip(segs, poss):
        pos = np.asarray(pos, float).copy(); pos[-1] = 0.0
        prev = np.concatenate([[0.0], pos[:-1]])
        beta = np.clip(s["beta"], -10, 10)
        denom = 1.0 + np.abs(beta)
        leg_y, leg_x = pos / denom, -pos * beta / denom
        prev_y, prev_x = np.concatenate([[0.0], leg_y[:-1]]), np.concatenate([[0.0], leg_x[:-1]])
        # Turnover includes both legs and hedge-ratio rebalancing.
        d = np.abs(leg_y - prev_y) + np.abs(leg_x - prev_x)
        short_gross = np.maximum(0.0, -leg_y) + np.maximum(0.0, -leg_x)
        carry = short_gross * ANNUAL_SHORT_CARRY_BPS / 1e4 / 252
        net = pos * s["sr_next"] - COST_BPS / 1e4 * d - carry
        parts.append(pd.Series(net / TOP_K, index=s["dates"]))
        turn += d.sum(); entries += int(((prev == 0) & (pos != 0)).sum()); held += int((pos != 0).sum())
    ret = pd.concat(parts).groupby(level=0).sum().sort_index() if parts else pd.Series(dtype=float)
    return dict(ret=ret, turnover=turn / max(1, len(years)) / TOP_K, entries=entries, avg_hold=held / max(1, entries))

def metrics(ret):
    if len(ret) < 20 or ret.std() == 0:
        return dict(Sharpe=np.nan, AnnRet=np.nan, AnnVol=np.nan, MaxDD=np.nan)
    eq = (1 + ret).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(ret)) - 1 if eq.iloc[-1] > 0 else np.nan
    peaks = np.maximum.accumulate(np.r_[1.0, eq.values])[1:]
    return dict(Sharpe=ret.mean() / ret.std() * np.sqrt(252), AnnRet=cagr, AnnVol=ret.std() * np.sqrt(252),
                MaxDD=(eq.values / peaks - 1).min())

def boot_sharpe_ci(ret, n=2000, block=10, seed=0):
    rng = np.random.default_rng(seed); r = ret.values; T = len(r)
    if T < 50: return (np.nan, np.nan)
    nb = int(np.ceil(T / block)); out = []
    for _ in range(n):
        st = rng.integers(0, T, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        x = r[idx]; out.append(x.mean() / (x.std() + 1e-12) * np.sqrt(252))
    return tuple(np.percentile(out, [2.5, 97.5]))

SEG = build_segments()
# --- tune entry/exit only on the inner training window; validation is kept for model selection
GRID_RULES = [(e, x) for e in (1.5, 2.0, 2.5) for x in (0.0, 0.5)]
BEST_RULE, tune_rows = {}, []
for m in SEG:
    best = None
    for e, x in GRID_RULES:
        r = backtest(SEG[m], zscore_policy(e, x), RULE_TUNE_YEARS)
        sh = metrics(r["ret"])["Sharpe"]; tune_rows.append(dict(method=m, entry=e, exit=x, Sharpe_inner_train=sh))
        if best is None or (np.nan_to_num(sh, nan=-9) > best[0]): best = (np.nan_to_num(sh, nan=-9), e, x)
    BEST_RULE[m] = best[1:]
print(pd.DataFrame(tune_rows).pivot_table(index=["entry", "exit"], columns="method", values="Sharpe_inner_train").round(2))
print("chosen (entry, exit):", BEST_RULE)

# %% [markdown]
# ## 5. Stage-2 comparison: does clustering help pair selection?
# Two views: (a) how many selected pairs are *still cointegrated* in the year they are traded (out-of-sample persistence),
# (b) the z-score baseline performance for each selection method.

# %%
persist = []
for m in SEL:
    for Y, picks in SEL[m].items():
        i0, i1 = YEAR_POS[Y]
        for r in picks:
            a, b = lp[r["y"]].iloc[i0:i1].values, lp[r["x"]].iloc[i0:i1].values
            if len(a) > 60 and not (np.isnan(a).any() or np.isnan(b).any()):
                persist.append(dict(method=m, year=Y, p_oos=eg_p(a, b)))
persist = pd.DataFrame(persist)
sel_cmp = persist.groupby("method").agg(pairs=("p_oos", "size"), median_p_oos=("p_oos", "median"),
                                        still_coint_5pct=("p_oos", lambda s: (s < 0.05).mean())).round(3)
print("Out-of-sample cointegration persistence of selected pairs"); print(sel_cmp)
print(SELSTATS.groupby("method")[["candidates", "passed", "selected"]].mean().round(1))

# %% [markdown]
# ## 6. LSTM forecaster (walk-forward)
# Input: last 20 days of [z, Δz, 5-day Δz, last spread return / vol]. Target: next-5-day spread return scaled by trailing volatility.
# For each year *Y* a fresh model is trained on **pair-years before Y only** (targets never reach into Y), with the last 15% (in time) used for early stopping.

# %%
def seq_feats(df):
    return np.column_stack([np.clip(df.z.values, -4, 4) / 2, np.clip(df.dz1.values, -2, 2),
                            np.clip(df.dz5.values, -4, 4) / 2, np.clip(df.sr_prev.values / df.vol.values, -4, 4) / 2])

class LSTMNet(nn.Module):
    def __init__(self, d=4, h=32):
        super().__init__(); self.lstm = nn.LSTM(d, h, batch_first=True); self.fc = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.lstm(x); return self.fc(o[:, -1]).squeeze(-1)

def windows(df, positions):
    F = seq_feats(df); out_x, out_t, keep = [], [], []
    for i in positions:
        if i - SEQ_LEN + 1 < 0: continue
        w = F[i - SEQ_LEN + 1: i + 1]
        if np.isnan(w).any(): continue
        out_x.append(w); keep.append(i)
    return (np.array(out_x, dtype=np.float32), np.array(keep)) if keep else (np.zeros((0, SEQ_LEN, 4), np.float32), np.array([], int))

def train_lstm(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); m = LSTMNet(); opt = torch.optim.Adam(m.parameters(), 1e-3); lossf = nn.HuberLoss()
    Xtr, ytr, Xva, yva = map(torch.tensor, (Xtr, ytr, Xva, yva))
    best, best_state, bad = 1e9, None, 0
    for ep in range(LSTM_EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; opt.zero_grad(); lossf(m(Xtr[b]), ytr[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): v = lossf(m(Xva), yva).item()
        if v < best - 1e-5: best, best_state, bad = v, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= 3: break
    m.load_state_dict(best_state); m.eval(); return m

LSTM_DIAG = []
ml_years = [Y for Y in years_active if Y >= FIRST_ML_YEAR]
for Y in ml_years:
    cutoff = YEAR_POS[Y][0] - 1                       # last index of data we may look at
    Xs, ys, ts = [], [], []
    for (p, Yp) in UNION:
        if Yp >= Y: continue
        df = PF[p]; i0, i1 = YEAR_POS[Yp]
        pos = [i for i in range(i0, i1) if i + H - 1 <= cutoff]
        X, keep = windows(df, pos)
        if len(keep) == 0: continue
        y = df.target.values[keep]; ok = ~np.isnan(y)
        Xs.append(X[ok]); ys.append(np.clip(y[ok], -4, 4).astype(np.float32)); ts.append(keep[ok])
    X, y, t = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ts)
    split = np.quantile(t, 0.85)
    tr, va = t <= split - H, t > split
    model = train_lstm(X[tr], y[tr], X[va], y[va], SEED + Y)
    preds_y, preds_t = [], []
    for (p, Yp) in UNION:
        if Yp != Y: continue
        df = PF[p]; i0, i1 = YEAR_POS[Y]
        Xp, keep = windows(df, list(range(i0, i1)))
        if len(keep) == 0: continue
        with torch.no_grad(): pr = model(torch.tensor(Xp)).numpy()
        df.iloc[keep, df.columns.get_loc("lstm")] = pr
        tg = df.target.values[keep]; ok = ~np.isnan(tg)
        preds_y.append(tg[ok]); preds_t.append(pr[ok])
    if preds_y:
        a, b = np.concatenate(preds_y), np.concatenate(preds_t)
        LSTM_DIAG.append(dict(year=Y, n=len(a), corr=np.corrcoef(a, b)[0, 1], dir_acc=(np.sign(a) == np.sign(b)).mean()))
    print(f"LSTM {Y}: trained on {tr.sum()} samples, evaluated on {len(a)} | corr={LSTM_DIAG[-1]['corr']:.3f}")
LSTM_DIAG = pd.DataFrame(LSTM_DIAG)
print(LSTM_DIAG.round(3).to_string(index=False))
print("Out-of-sample correlation > ~0.03 sustained across years would be a meaningful signal; ~0 means the LSTM is not predictive here.")

# %% [markdown]
# ## 7. Markov state model (walk-forward)
# The z-score is discretised into 5 states (<-2, -2..-0.5, -0.5..0.5, 0.5..2, >2). The one-step transition matrix is estimated from **pair-years before Y**,
# raised to the power `H` to get 5-day-ahead probabilities. Features: expected z-change, P(back to centre), P(extreme state).

# %%
BINS = np.array([-2.0, -0.5, 0.5, 2.0])
def markov_fit(Y):
    C = np.ones((5, 5)); zs = [[] for _ in range(5)]
    for (p, Yp) in UNION:
        if Yp >= Y: continue
        i0, i1 = YEAR_POS[Yp]; z = PF[p].z.values[i0:i1]; z = z[~np.isnan(z)]
        s = np.digitize(z, BINS)
        for a, b in zip(s[:-1], s[1:]): C[a, b] += 1
        for k in range(5): zs[k].append(z[s == k])
    P = C / C.sum(1, keepdims=True)
    means = np.array([np.concatenate(zs[k]).mean() if sum(len(v) for v in zs[k]) else [-2.5, -1.2, 0, 1.2, 2.5][k] for k in range(5)])
    return P, np.linalg.matrix_power(P, H), means

MARKOV = {}
for Y in ml_years:
    P1, PH, means = markov_fit(Y); MARKOV[Y] = (P1, PH, means)
    for (p, Yp) in UNION:
        if Yp != Y: continue
        df = PF[p]; i0, i1 = YEAR_POS[Y]; z = df.z.values[i0:i1]; s = np.digitize(z, BINS)
        sl = slice(i0, i1)
        df.iloc[sl, df.columns.get_loc("mk_dz")] = PH[s] @ means - z
        df.iloc[sl, df.columns.get_loc("mk_center")] = PH[s, 2]
        df.iloc[sl, df.columns.get_loc("mk_ext")] = PH[s, 0] + PH[s, 4]
print("One-step transition matrix estimated up to the last pre-test year:\n", pd.DataFrame(MARKOV[max(ml_years)][0]).round(3))

SEG = build_segments()     # rebuild so segments carry LSTM / Markov features

# %% [markdown]
# ## 8. LSTM-only strategy (no RL)
# Position = LSTM forecast scaled by the 90th percentile of |forecast| on the training years, rounded to {-1,-0.5,0,0.5,1}.

# %%
def lstm_policy(scale):
    def pol(segs):
        return [np.clip(np.round(np.clip(s["lstm"] / scale, -1, 1) * 2) / 2, -1, 1) for s in segs]
    return pol

# Fix the downstream selection method in advance; do not choose it by looking at validation returns.
PPO_METHOD = "all"
print("Downstream pair selector fixed in advance:", PPO_METHOD)

tr_abs = np.concatenate([np.abs(s["lstm"][s["lstm"] != 0]) for s in SEG[PPO_METHOD] if s["year"] in TRAIN_YEARS])
LSTM_SCALE = np.percentile(tr_abs, 90) if len(tr_abs) else 1.0

# %% [markdown]
# ## 9. PPO agent
# **State:** z, Δz, 5-day Δz, current position (+ LSTM forecast and/or Markov features depending on the variant).
# **Action:** target position in {-1, -0.5, 0, 0.5, 1} of the spread. **Reward:** position × next-day spread return − trading cost.
# Trained on the train years only (LSTM/Markov inputs are out-of-sample there, so PPO learns how much to trust them).
# Checkpoints are chosen on the **validation** years; test is evaluated once at the end. Four variants + 3 seeds each.

# %%
VARIANTS = {"PPO z-only": (False, False), "PPO + LSTM": (True, False), "PPO + Markov": (False, True), "PPO + LSTM + Markov": (True, True)}

def obs_matrix(s, variant):
    use_l, use_m = VARIANTS[variant]
    cols = [np.clip(s["z"] / 2, -3, 3), np.clip(s["dz1"], -2, 2), np.clip(s["dz5"] / 2, -3, 3)]
    if use_l: cols.append(np.clip(s["lstm"] / 2, -2, 2))
    if use_m: cols += [np.clip(s["mk_dz"] / 2, -3, 3), s["mk_center"], s["mk_ext"]]
    return np.column_stack(cols).astype(np.float32)

class PairEnv(gym.Env):
    def __init__(self, segs, variant):
        super().__init__()
        self.segs = segs; self.O = [obs_matrix(s, variant) for s in segs]; d = self.O[0].shape[1] + 1
        self.observation_space = spaces.Box(-5, 5, (d,), np.float32); self.action_space = spaces.Discrete(len(GRID))
        self.rng = np.random.default_rng(0)
    def _obs(self): return np.append(self.O[self.k][self.t], self.pos).astype(np.float32)
    def reset(self, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.k = int(self.rng.integers(len(self.segs))); self.t = 0; self.pos = 0.0
        return self._obs(), {}
    def step(self, a):
        s = self.segs[self.k]; n = len(s["z"]); newpos = 0.0 if self.t == n - 1 else float(GRID[a])
        beta = float(np.clip(s["beta"][self.t], -10, 10))
        prev_beta = float(np.clip(s["beta"][max(0, self.t - 1)], -10, 10))
        den, prev_den = 1.0 + abs(beta), 1.0 + abs(prev_beta)
        turnover = abs(newpos / den - self.pos / prev_den) + abs(-newpos * beta / den + self.pos * prev_beta / prev_den)
        short_gross = max(0.0, -newpos / den) + max(0.0, newpos * beta / den)
        carry = short_gross * ANNUAL_SHORT_CARRY_BPS / 1e4 / 252
        r = 100 * (newpos * s["sr_next"][self.t] - COST_BPS / 1e4 * turnover - carry)
        self.pos = newpos; done = self.t == n - 1
        if not done: self.t += 1
        return self._obs(), r, done, False, {}

def ppo_policy(model, variant):
    def pol(segs):
        O = [obs_matrix(s, variant) for s in segs]
        pos = [np.zeros(len(s["z"])) for s in segs]
        prev = np.zeros(len(segs))
        date_rows = {}
        for i, s in enumerate(segs):
            for t, date in enumerate(s["dates"]):
                date_rows.setdefault(date, []).append((i, t))
        for rows_for_date in (date_rows[d] for d in sorted(date_rows)):
            obs = np.stack([np.append(O[i][t], prev[i]) for i, t in rows_for_date]).astype(np.float32)
            a, _ = model.predict(obs, deterministic=True)
            rows = [(i, t, 0.0 if t == len(segs[i]["z"]) - 1 else float(GRID[a[j]]))
                    for j, (i, t) in enumerate(rows_for_date)]
            capped = cap_one_day(segs, rows)
            for (i, t, _), p in zip(rows, capped):
                pos[i][t] = p; prev[i] = p
        return pos
    return pol

train_segs = [s for s in SEG[PPO_METHOD] if s["year"] in TRAIN_YEARS]
print("PPO training segments:", len(train_segs), "(~", sum(len(s['z']) for s in train_segs), "daily steps)")
PPO_RES = {v: {} for v in VARIANTS}
for v in VARIANTS:
    for sd in PPO_SEEDS:
        t0 = time.time(); env = PairEnv(train_segs, v)
        model = PPO("MlpPolicy", env, seed=sd, device="cpu", verbose=0, n_steps=2048, batch_size=256, n_epochs=5,
                    learning_rate=3e-4, ent_coef=0.01, gamma=0.99, policy_kwargs=dict(net_arch=[64, 64]))
        best_val, best_state, done_steps = -1e9, None, 0
        while done_steps < PPO_TOTAL_STEPS:
            model.learn(PPO_EVAL_EVERY, reset_num_timesteps=False); done_steps += PPO_EVAL_EVERY
            vr = backtest(SEG[PPO_METHOD], ppo_policy(model, v), VAL_YEARS)["ret"]
            sh = np.nan_to_num(metrics(vr)["Sharpe"], nan=-9)
            if sh > best_val: best_val, best_state = sh, copy.deepcopy(model.policy.state_dict())
        model.policy.load_state_dict(best_state)
        PPO_RES[v][sd] = dict(model=model, val_sharpe=best_val)
        print(f"{v:22s} seed {sd}: best val Sharpe {best_val:5.2f}  ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## 10. Final evaluation (validation and TEST)
# Test years were not used for pair choice, rule tuning, LSTM/Markov fitting, PPO training or checkpoint selection.

# %%
rows, curves = [], {}
def add_row(name, res_val, res_test, extra=None):
    mv, mt = metrics(res_val["ret"]), metrics(res_test["ret"]); lo, hi = boot_sharpe_ci(res_test["ret"])
    rows.append(dict(Strategy=name, Val_Sharpe=mv["Sharpe"], Test_Sharpe=mt["Sharpe"], Test_CI_lo=lo, Test_CI_hi=hi,
                     Test_AnnRet=mt["AnnRet"], Test_AnnVol=mt["AnnVol"], Test_MaxDD=mt["MaxDD"],
                     Test_turnover_per_yr=res_test["turnover"], Test_entries=res_test["entries"], **(extra or {})))
    curves[name] = res_test["ret"]

for m in SEG:
    pol = zscore_policy(*BEST_RULE[m])
    add_row(f"Z-score ({'all pairs' if m=='all' else 'clustered'})", backtest(SEG[m], pol, VAL_YEARS), backtest(SEG[m], pol, TEST_YEARS))
pol = lstm_policy(LSTM_SCALE)
add_row("LSTM-only rule", backtest(SEG[PPO_METHOD], pol, VAL_YEARS), backtest(SEG[PPO_METHOD], pol, TEST_YEARS))

for v in VARIANTS:
    vals, tests, test_sh = [], [], []
    for sd, d in PPO_RES[v].items():
        pol = ppo_policy(d["model"], v)
        vals.append(backtest(SEG[PPO_METHOD], pol, VAL_YEARS)); tests.append(backtest(SEG[PPO_METHOD], pol, TEST_YEARS))
        test_sh.append(metrics(tests[-1]["ret"])["Sharpe"])
        PPO_RES[v][sd]["test_sharpe"] = test_sh[-1]
    ens = lambda rs: dict(ret=pd.concat([r["ret"] for r in rs], axis=1).fillna(0).mean(axis=1), turnover=np.mean([r["turnover"] for r in rs]),
                          entries=int(np.mean([r["entries"] for r in rs])))
    suffix = " (single seed)" if len(test_sh) == 1 else " (seed ensemble)"
    add_row(v + suffix, ens(vals), ens(tests), dict(Seed_Sharpe_mean=np.mean(test_sh),
                                                    Seed_Sharpe_std=np.std(test_sh) if len(test_sh) > 1 else np.nan))

RESULTS = pd.DataFrame(rows).set_index("Strategy")
pd.set_option("display.width", 220)
print(RESULTS.round(3).to_string())
RESULTS.round(4).to_csv(f"{OUT_DIR}/results_table.csv"); sel_cmp.to_csv(f"{OUT_DIR}/selection_comparison.csv"); LSTM_DIAG.to_csv(f"{OUT_DIR}/lstm_diagnostics.csv", index=False)
pd.DataFrame(curves).to_csv(f"{OUT_DIR}/test_daily_returns.csv", index_label="Date")
SELSTATS.to_csv(f"{OUT_DIR}/selection_counts.csv", index=False)
pd.DataFrame([{**{"year": Y, "method": m}, **p} for m, years in SEL.items() for Y, picks in years.items() for p in picks]).to_csv(f"{OUT_DIR}/selected_pairs.csv", index=False)
manifest = dict(profile=PROFILE, fast=FAST, source=DATA_SOURCE, price_csv=PRICE_CSV if DATA_SOURCE == "csv" else None,
                price_sha256=hashlib.sha256(open(PRICE_CSV, "rb").read()).hexdigest() if DATA_SOURCE == "csv" else None,
                stocks=int(px.shape[1]), rows=int(len(px)), first_date=str(IDX[0].date()), last_date=str(IDX[-1].date()),
                formation_years=sorted(SEL["all"]), train_years=TRAIN_YEARS, validation_years=VAL_YEARS,
                test_years=TEST_YEARS, pairs_per_year=TOP_K, fdr_q=FDR_Q, cost_bps=COST_BPS,
                annual_short_carry_bps=ANNUAL_SHORT_CARRY_BPS, max_stock_gross_weight=MAX_STOCK_GROSS_WEIGHT,
                ppo_seeds=PPO_SEEDS, ppo_training_steps=PPO_TOTAL_STEPS, lstm_epochs=LSTM_EPOCHS)
with open(f"{OUT_DIR}/run_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for name, r in curves.items():
    ax[0].plot((1 + r).cumprod(), label=name, lw=1.4)
ax[0].set_title("Test-period equity curves (beta-hedged, after costs)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].barh(RESULTS.index, RESULTS["Test_Sharpe"], xerr=[np.maximum(0, RESULTS["Test_Sharpe"] - RESULTS["Test_CI_lo"]),
                                                         np.maximum(0, RESULTS["Test_CI_hi"] - RESULTS["Test_Sharpe"])], color="#4c78a8")
ax[1].axvline(0, c="k", lw=.8); ax[1].set_title("Test Sharpe with 95% block-bootstrap CI"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/test_results.png", dpi=130); plt.show()

# %% [markdown]
# ## 11. Save the final model
# The "final model" is whichever PPO variant has the best **validation** Sharpe (mean over configured seeds); its best-validation seed is saved.

# %%
val_mean = {v: np.mean([d["val_sharpe"] for d in PPO_RES[v].values()]) for v in VARIANTS}
FINAL_VARIANT = max(val_mean, key=val_mean.get)
FINAL_SEED = max(PPO_RES[FINAL_VARIANT], key=lambda s: PPO_RES[FINAL_VARIANT][s]["val_sharpe"])
PPO_RES[FINAL_VARIANT][FINAL_SEED]["model"].save(f"{OUT_DIR}/final_ppo_model")
print("Validation Sharpe by variant:", {k: round(v, 2) for k, v in val_mean.items()})
print(f"FINAL MODEL = {FINAL_VARIANT} (seed {FINAL_SEED}), selection = {PPO_METHOD}, saved to {OUT_DIR}/final_ppo_model.zip")
print("Saved final model seed test Sharpe:", round(PPO_RES[FINAL_VARIANT][FINAL_SEED]["test_sharpe"], 2),
      "| aggregate test Sharpe:", round(RESULTS.loc[FINAL_VARIANT + (' (single seed)' if len(PPO_SEEDS) == 1 else ' (seed ensemble)'), 'Test_Sharpe'], 2),
      "| z-score baseline:", round(RESULTS.loc["Z-score (all pairs)", "Test_Sharpe"], 2))
