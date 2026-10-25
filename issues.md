# issues.md: Data collection fixes (Issue #1 and Issue #2)

**Project:** ml-alpha-portfolio-model
**Notebook reviewed:** `notebooks/01_domain_setup_data_collection.ipynb` (with knock-on effects in `02_feature_engineering.ipynb`)
**Review date:** 18 Oct 2026
**Data reviewed:** the current files in `data/raw/` (100 price files, last date 10 Apr 2026) and a sample of 12 files in `data/news_raw/`

> **No script has been changed.** This file only describes what to change, where, and why.

---

## How to read the locations in this file

- Locations use the **Step label written in the cell's header comment** (for example `# Step 4: Fetch Price + News`), plus the **line number inside that cell**.
- To see line numbers in Jupyter, click into the cell and press **Shift + L** (or use *View → Show Line Numbers*).
- The first code cell of notebook 01 has no Step label. It is called **"Imports cell"** below.
- Line numbers refer to the notebooks exactly as they are on disk today. **Make the changes from the bottom of a cell upward** so earlier line numbers stay valid while you edit.

### Map of notebook 01 cells

| Cell header in notebook 01 | What it does today |
|---|---|
| Imports cell | imports `nselib`, `pandas`, `feedparser`, `transformers`, … |
| `# Step 0: Project Setup` | **deletes the entire `data/` folder** and recreates `data/raw`, `data/news_raw` |
| `# Step 1: Load FinBERT` | loads the `ProsusAI/finbert` pipeline |
| `# Step 2: Fetch NIFTY 100 tickers` | reads the Nifty 100 list from the NSE archive |
| `# Step 3: News Fetch Function` | Google News RSS + FinBERT scoring → `fetch_news_sentiment()` |
| `# Step 4: Fetch Price + News` | `fetch_all_data()`: NSE prices via `nselib`, saves CSV, calls the news function |
| `# Step 5: Execute` | runs `fetch_all_data()` from 2020-01-01 to today |

---

## Summary

| # | Issue | Severity | Where it starts | What it breaks |
|---|---|---|---|---|
| 1 | Prices are **not adjusted** for splits, bonuses and demergers | **Critical** | NB01 Step 4 (saves raw NSE prices as they are) | Every return, momentum, volatility, RSI, MACD and EMA feature, the 21-day target (NB03), and the backtest P&L (NB04) |
| 2 | FinBERT **neutral** headlines are scored as **strongly negative** | **High** | NB01 Step 3, line 42 | All 13 news/sentiment features plus `sent_mom_interaction` |

---

# Issue #1: Prices are not adjusted for corporate actions

## 1.1 What is wrong

`capital_market.price_volume_and_deliverable_position_data()` (NB01 Step 4, lines 13–17) returns the **actual traded prices** for each day. When a company splits its shares or issues bonus shares, the share price drops mechanically on the ex-date. No shareholder actually lost money, but the price series shows a crash.

I also checked whether NSE's own `prevclose` column could be used to detect these events (on exchanges that adjust it, `prevclose` on the ex-date ≠ the previous day's close). **It can't: in these files `prevclose` is also unadjusted.** For example, BAJFINANCE on 16 Jun 2025 has `prevclose` = the old pre-split close. The adjustment has to come from a separate corporate-actions source.

Nothing in `02_feature_engineering.ipynb` corrects this. Step 3 (lines 34–46) renames `closeprice` → `Close`, and Step 2 (lines 9–68) calculates every feature directly from `Close`, `High`, `Low` and `Volume`.

## 1.2 Evidence: every jump found in the current data

I scanned all 100 files in `data/raw/` for day-on-day close-to-close moves larger than ±20%.

> The **% move and the date are measured from your files.** The "likely cause" column is from my knowledge of these events. Each cause fits the size of the move (for example, a 1:1 bonus = exactly −50%), but **confirm them against the corporate-actions file that the fix below downloads** (`data/corporate_actions.csv`).

### A. Fake moves caused by corporate actions (36 events, 29 stocks): must be adjusted

| Symbol | Ex-date | Raw 1-day move | Likely cause (verify) | Price factor that should apply to all earlier prices |
|---|---|---|---|---|
| ADANIPOWER | 2025-09-22 | −76.0% | Split ₹10 → ₹2 | 0.20 |
| BAJAJFINSV | 2022-09-13 | −89.6% | Split ₹5 → ₹1 + bonus 1:1 | 0.10 |
| BAJFINANCE | 2025-06-16 | −89.9% | Split ₹2 → ₹1 + bonus 4:1 | 0.10 |
| BEL | 2022-09-15 | −66.9% | Bonus 2:1 | 0.333 |
| BPCL | 2024-06-21 | −50.9% | Bonus 1:1 | 0.50 |
| CANBK | 2024-05-15 | −79.0% | Split ₹10 → ₹2 | 0.20 |
| DRREDDY | 2024-10-28 | −79.9% | Split ₹5 → ₹1 | 0.20 |
| EICHERMOT | 2020-08-24 | −90.0% | Split ₹10 → ₹1 | 0.10 |
| GAIL | 2022-09-06 | −32.0% | Bonus 1:2 | 0.667 |
| HAL | 2023-09-28 | −50.8% | Split ₹10 → ₹5 | 0.50 |
| HDFCAMC | 2025-11-26 | −49.8% | Bonus 1:1 | 0.50 |
| HDFCBANK | 2025-08-26 | −50.4% | Bonus 1:1 | 0.50 |
| IOC | 2022-06-30 | −32.4% | Bonus 1:2 | 0.667 |
| KOTAKBANK | 2026-01-14 | −80.3% | Split ₹5 → ₹1 | 0.20 |
| LODHA | 2023-05-31 | −48.1% | Bonus 1:1 | 0.50 |
| MAZDOCK | 2024-12-27 | −51.0% | Split ₹10 → ₹5 | 0.50 |
| MOTHERSON | 2022-01-14 | −21.0% | Demerger (wiring-harness business) | ≈ 0.830 (ex-date open ÷ previous close) |
| MOTHERSON | 2022-10-03 | −33.1% | Bonus 1:2 | 0.667 |
| MOTHERSON | 2025-07-18 | −34.4% | Bonus 1:2 | 0.667 |
| NESTLEIND | 2024-01-05 | −90.2% | Split ₹10 → ₹1 | 0.10 |
| NESTLEIND | 2025-08-08 | −50.9% | Bonus 1:1 | 0.50 |
| PIDILITIND | 2025-09-23 | −51.0% | Bonus 1:1 | 0.50 |
| POWERGRID | 2021-07-29 | −26.6% | Bonus 1:3 | 0.75 |
| POWERGRID | 2023-09-12 | −27.7% | Bonus 1:3 | 0.75 |
| RECLTD | 2022-08-17 | −23.2% | Bonus 1:3 | 0.75 |
| RELIANCE | 2024-10-28 | −49.8% | Bonus 1:1 | 0.50 |
| SHRIRAMFIN | 2025-01-10 | −81.1% | Split ₹10 → ₹2 | 0.20 |
| SIEMENS | 2025-04-07 | −42.9% | Demerger of energy business (→ ENRIN) | ≈ 0.497 (ex-date open ÷ previous close) |
| TATASTEEL | 2022-07-28 | −89.5% | Split ₹10 → ₹1 | 0.10 |
| TMPV | 2025-10-14 | −40.2% | Tata Motors demerger (CV business → TMCV) | ≈ 0.605 (ex-date open ÷ previous close) |
| TORNTPHARM | 2022-07-08 | −50.3% | Bonus 1:1 | 0.50 |
| VBL | 2021-06-10 | −31.4% | Bonus 1:2 | 0.667 |
| VBL | 2022-06-06 | −33.4% | Bonus 1:2 | 0.667 |
| VBL | 2023-06-15 | −48.8% | Split ₹10 → ₹5 | 0.50 |
| VBL | 2024-09-12 | −58.9% | Split ₹5 → ₹2 | 0.40 |
| WIPRO | 2024-12-03 | −50.1% | Bonus 1:1 | 0.50 |

### B. Corporate actions smaller than ±20% (not caught by a threshold scan): must also be adjusted

| Symbol | Ex-date | Raw move at the open | Likely cause |
|---|---|---|---|
| ITC | 2025-01-06 | −5.4% (open 455.60 vs previous close 481.60) | Demerger of ITC Hotels |
| RELIANCE | 2023-07-20 | −9.2% (open 2580.00 vs previous close 2841.85) | Demerger of Jio Financial (→ JIOFIN) |

These two show why a "find big jumps" threshold rule is **not** a valid fix. Small demergers, rights issues and dividends slip under any threshold. The fix must be driven by a **corporate-actions list**.

### C. Large moves that look genuine: must NOT be adjusted

These match real market events. They are listed so they don't get "fixed" by mistake.

| Symbol(s) | Date | Move | Likely real event |
|---|---|---|---|
| AXISBANK, BAJAJFINSV, BAJFINANCE, CANBK, CHOLAFIN, SHRIRAMFIN, TRENT | 2020-03-23 | −20% to −30% | COVID crash |
| ADANIPOWER, MOTHERSON | 2020-03-12 | −26.5%, −24.3% | COVID crash |
| ADANIENT | 2023-02-01, 02-02, 02-08 | −28.2%, −26.7%, +20.0% | Hindenburg report |
| ADANIENT / ADANIGREEN | 2024-11-21 / 2024-11-29 | −22.6% / +21.8% | US indictment news and rebound |
| ADANIPORTS, PFC, RECLTD | 2024-06-04 | −21% to −25% | General-election results day |
| ADANIENT | 2020-08-25 | +23.7% | Real move |
| JINDALSTEL | 2020-04-07 | +28.9% | Real move |
| GODREJCP | 2021-05-12 | +21.9% | Real move |
| TMPV | 2021-10-13 | +20.4% | TPG investment in the EV arm |
| VEDL | 2020-10-12 | −20.4% | Failed delisting offer |

## 1.3 How the fix is designed (read this before editing)

1. **Keep NSE's raw columns exactly as they are.** Add new `adj_*` columns next to them. Raw prices are still needed for delivery %, turnover and audit.
2. **Download corporate actions from NSE** using `capital_market.corporate_actions_for_equity()`, which already exists in `nselib` (you are on 2.5.1; the function is in `nselib/capital_market`). Save it as `data/corporate_actions.csv`.
3. **Turn each action into a price factor `f`.** Multiply every price **before** the ex-date by `f`.
   - Split from face value `X` to `Y`: `f = Y / X` (₹10 → ₹2 gives 0.2)
   - Bonus `a:b` (a new shares for every b held): `f = b / (a + b)` (1:1 gives 0.5, 1:2 gives 0.667, 2:1 gives 0.333)
   - Demerger: `f = open on the ex-date ÷ close on the day before`. On a demerger ex-date NSE runs a special pre-open session to find the parent's new price, so the ex-date open is the market's estimate of the value left in the parent.
   - Dividend (optional, see 1.5): `f = 1 − dividend per share ÷ close on the day before`
   - Rights issue: printed for manual review, not auto-adjusted (rare in Nifty 100 and usually small)
4. **Several actions on the same date multiply together.** BAJFINANCE: split 0.5 × bonus 0.2 = 0.1.
5. **Volume must be adjusted the opposite way.** Volume and delivery quantity are **divided** by the share-count factor, because after a 1:1 bonus twice as many shares trade for the same money. Only splits and bonuses change the share count. Demergers and dividends do not.
6. **Recalculate the adjustment from the raw columns on every run.** When a new split happens, *all* earlier adjusted prices change. This matters for the CI/CD update job: never append new adjusted rows onto old adjusted rows.
7. **Manual override table.** If NSE's text for an event can't be parsed, or an event is missing, add the factor by hand in `MANUAL_CA_FACTORS`.

## 1.4 Code changes for Issue #1 (exact locations)

### Change 1-A: NB01, Imports cell, after line 14

**Add** two imports at the end of the cell:

```python
import re
import numpy as np
```

### Change 1-B: NB01, `# Step 4: Fetch Price + News`, insert after line 33

Lines 26–33 are the "Keep only EQ series" block. **Directly after line 33** (`continue`, at the same indentation as line 26), insert:

```python
            # --- Clean types and sort (required for corporate-action adjustment) ---
            df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y")
            num_cols = [
                "prevclose", "openprice", "highprice", "lowprice", "lastprice",
                "closeprice", "averageprice", "totaltradedquantity", "turnoverinrs",
                "no.oftrades", "deliverableqty", "%dlyqttotradedqty",
            ]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(
                        df[c].astype(str).str.replace(",", "", regex=False),
                        errors="coerce",
                    )
            df = (
                df.sort_values("date")
                .drop_duplicates(subset="date", keep="last")
                .reset_index(drop=True)
            )
```

**Why:** today the CSV stores prices as text with commas (`"5,200.50"`) and rows in mixed date order (30-Dec, 29-Dec, …). The adjustment needs real numbers in date order. Lines 35–36 (the save) stay unchanged. NB02 Step 3 lines 55–61 still work on the cleaned file, because `to_datetime` reads ISO dates and the comma-stripping does nothing on plain numbers.

### Change 1-C: NB01, new cell `# Step 4.5: Corporate Actions & Price Adjustment`

**Insert a new code cell between `Step 4` and `Step 5`** containing:

```python
# -----------------------------
# Step 4.5: Corporate Actions & Price Adjustment
# -----------------------------
CA_FILE = DATA_DIR / "corporate_actions.csv"
ADJ_LOG_FILE = DATA_DIR / "adjustment_log.csv"

ADJUST_DIVIDENDS = False   # False = price-adjusted; True = total-return-adjusted (see issues.md 1.5)

# Manual fixes: (SYMBOL, "YYYY-MM-DD" ex-date) -> price factor.
# Replaces any automatic factor for that symbol and date. Example:
# MANUAL_CA_FACTORS = {("TMPV", "2025-10-14"): 0.6054}
MANUAL_CA_FACTORS = {}

PRICE_COLS = ["openprice", "highprice", "lowprice", "lastprice", "closeprice", "averageprice"]
QTY_COLS = ["totaltradedquantity", "deliverableqty"]
RUPEE = r"(?:rs|re)\.?\s*"


def fetch_corporate_actions(start, end):
    """Download NSE corporate actions in 1-year chunks and save them to CA_FILE."""
    frames = []
    for y in range(start.year, end.year + 1):
        f = max(start, date(y, 1, 1))
        t = min(end, date(y, 12, 31))
        for attempt in range(3):
            try:
                ca = capital_market.corporate_actions_for_equity(
                    from_date=f.strftime("%d-%m-%Y"), to_date=t.strftime("%d-%m-%Y")
                )
                frames.append(ca)
                break
            except Exception as e:
                print(f" CA fetch failed for {y} (attempt {attempt + 1}): {e}")
                time.sleep(3)
        time.sleep(1)

    ca = pd.concat(frames, ignore_index=True)
    ca.columns = ca.columns.str.lower().str.strip()
    # On the first run, confirm the column names:
    # print(ca.columns.tolist())  -> expect 'symbol', 'subject', 'exdate', ...
    ca["exdate"] = pd.to_datetime(ca["exdate"], format="mixed", dayfirst=True, errors="coerce")
    ca = ca.dropna(subset=["exdate"]).drop_duplicates(subset=["symbol", "exdate", "subject"])
    ca.to_csv(CA_FILE, index=False)
    print(f" Corporate actions saved: {len(ca)} rows")
    return ca


def parse_actions(subject):
    """Turn NSE 'subject' text into a list of (kind, value)."""
    s = str(subject).lower()
    out = []

    # Split, e.g. "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
    m = re.search(r"from\s*" + RUPEE + r"([\d.]+).*?to\s*" + RUPEE + r"([\d.]+)", s)
    if ("split" in s or "sub-division" in s or "sub division" in s) and m:
        out.append(("split", float(m.group(2)) / float(m.group(1))))

    # Bonus a:b, e.g. "Bonus 1:1"
    for a, b in re.findall(r"bonus\s*(\d+)\s*:\s*(\d+)", s):
        out.append(("bonus", int(b) / (int(a) + int(b))))

    if "demerger" in s or "scheme of arrangement" in s:
        out.append(("demerger", None))

    if "rights" in s:
        out.append(("rights", None))

    # Dividend, e.g. "Interim Dividend - Rs 5.50 Per Share / Special Dividend - Rs 2 Per Share"
    divs = re.findall(r"dividend[^/]*?" + RUPEE + r"([\d.]+)", s)
    if divs:
        out.append(("dividend", sum(float(x) for x in divs)))

    return out


def build_events(symbol, px, ca):
    """Return a list of (exdate, price_factor, share_factor, kind) for one symbol."""
    px = px.sort_values("date").reset_index(drop=True)   # "day before" / "ex-date open" need date order
    events = []
    for _, r in ca[ca["symbol"] == symbol].iterrows():
        ex = r["exdate"]
        before = px[px["date"] < ex]
        on_or_after = px[px["date"] >= ex]
        if before.empty or on_or_after.empty:
            continue
        prev_close = before["closeprice"].iloc[-1]

        for kind, val in parse_actions(r["subject"]):
            if kind in ("split", "bonus"):
                events.append((ex, val, val, kind))            # price AND share count change
            elif kind == "demerger":
                f = on_or_after["openprice"].iloc[0] / prev_close
                events.append((ex, f, 1.0, kind))              # price only
            elif kind == "dividend" and ADJUST_DIVIDENDS and val:
                events.append((ex, 1 - val / prev_close, 1.0, kind))
            elif kind == "rights":
                print(f"  [review] {symbol} rights issue {ex.date()} not adjusted: {r['subject']}")

    # Manual overrides replace automatic events on the same date
    for (sym, d), f in MANUAL_CA_FACTORS.items():
        if sym == symbol:
            d = pd.Timestamp(d)
            events = [e for e in events if e[0] != d]
            events.append((d, f, f, "manual"))
    return events


def apply_adjustments(px, events):
    """Add adj_* columns. Raw NSE columns are left untouched."""
    px = px.sort_values("date").reset_index(drop=True)
    price_cum = np.ones(len(px))
    share_cum = np.ones(len(px))
    for ex, f_price, f_share, _ in events:
        mask = (px["date"] < ex).to_numpy()
        price_cum[mask] *= f_price
        share_cum[mask] *= f_share
    for c in PRICE_COLS:
        px[f"adj_{c}"] = px[c] * price_cum
    for c in QTY_COLS:
        px[f"adj_{c}"] = px[c] / share_cum
    px["adj_factor"] = price_cum
    return px


def adjust_all_prices(ca):
    log = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        symbol = f.stem
        px = pd.read_csv(f, parse_dates=["date"])
        px = px.drop(columns=[c for c in px.columns if c.startswith("adj_")])  # safe to re-run
        events = build_events(symbol, px, ca)
        px = apply_adjustments(px, events)
        px.to_csv(f, index=False)
        for ex, fp, fs, kind in events:
            log.append({"symbol": symbol, "exdate": ex.date(), "kind": kind,
                        "price_factor": round(fp, 6), "share_factor": round(fs, 6)})
    pd.DataFrame(log).to_csv(ADJ_LOG_FILE, index=False)
    print(f" Adjusted {len(list(RAW_DIR.glob('*.csv')))} files, {len(log)} events applied.")
```

Notes on this cell:
- `prevclose` is **deliberately not adjusted**. It belongs to the previous day, so it would need the previous day's factor, and NB02 does not use it.
- `%dlyqttotradedqty` and `turnoverinrs` need no adjustment. A percentage and a rupee value are unaffected by splits.
- I could not call NSE from here, so the column names `symbol`, `subject` and `exdate` come from the `nselib` source code and NSE's API format. **On the first run, uncomment the `print(ca.columns.tolist())` line and confirm them.**

### Change 1-D: NB01, `# Step 5: Execute`, insert after line 7

After line 7 (`fetch_all_data(tickers, START_DATE, END_DATE)`), insert:

```python
ca = fetch_corporate_actions(START_DATE, END_DATE)
adjust_all_prices(ca)
```

### Change 1-E: NB01, new cell `# Step 6: Validate Adjusted Prices`

**Insert a new code cell after Step 5**:

```python
# -----------------------------
# Step 6: Validate Adjusted Prices
# -----------------------------
# 1) Known corporate-action dates must no longer show a crash
KNOWN_EVENTS = [
    ("BAJFINANCE", "2025-06-16"), ("HDFCBANK", "2025-08-26"), ("RELIANCE", "2024-10-28"),
    ("TMPV", "2025-10-14"), ("SIEMENS", "2025-04-07"), ("EICHERMOT", "2020-08-24"),
    ("NESTLEIND", "2024-01-05"), ("KOTAKBANK", "2026-01-14"), ("VBL", "2024-09-12"),
    ("ITC", "2025-01-06"), ("RELIANCE", "2023-07-20"),
]
for sym, d in KNOWN_EVENTS:
    fp = RAW_DIR / f"{sym}.csv"
    if not fp.exists():
        continue
    px = pd.read_csv(fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    i = px.index[px["date"] == pd.Timestamp(d)]
    if len(i) == 0:
        print(f" {sym} {d}: date not found")
        continue
    i = i[0]
    raw_r = px.loc[i, "closeprice"] / px.loc[i - 1, "closeprice"] - 1
    adj_r = px.loc[i, "adj_closeprice"] / px.loc[i - 1, "adj_closeprice"] - 1
    status = "OK" if abs(adj_r) < 0.15 else "STILL BROKEN"
    print(f" {sym:<11} {d}  raw {raw_r:+.1%}  ->  adjusted {adj_r:+.1%}   {status}")

# 2) Every remaining |1-day move| > 20% goes to a review file
flags = []
for fp in sorted(RAW_DIR.glob("*.csv")):
    px = pd.read_csv(fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    r = px["adj_closeprice"].pct_change()
    gap = px["date"].diff().dt.days
    for i in px.index[r.abs() > 0.20]:
        flags.append({"symbol": fp.stem, "date": px.loc[i, "date"].date(),
                      "adj_ret": round(r[i], 4), "gap_days": int(gap[i])})
flags = pd.DataFrame(flags)
flags.to_csv(DATA_DIR / "large_moves_review.csv", index=False)
print(f"\n {len(flags)} large moves left for review -> data/large_moves_review.csv")
```

**Expected result after the fix:** every row in part (1) prints `OK`. The review file should contain only the genuine moves from table 1.2-C, plus the gap rows described in the Addendum.

### Change 1-F: NB02 (`02_feature_engineering.ipynb`), `# Step 3: Process Each Ticker`, replace lines 34–46

Replace the whole `col_map = { ... }` block (lines 34–46) with:

```python
        col_map = {
            "date": "Date",
            "adj_openprice": "Open",
            "adj_highprice": "High",
            "adj_lowprice": "Low",
            "adj_closeprice": "Close",
            "adj_totaltradedquantity": "Volume",
        }
```

**Why:** NB02 must read the **adjusted** columns. The old mappings (`"openprice": "Open"`, `"closeprice": "Close"`, …) must be **removed, not kept alongside the new ones**. Otherwise pandas creates two columns both named `Close`. Line 32 already lower-cases the column names, so `adj_closeprice` matches. Lines 50–53 (the required-column check) then confirm the adjusted columns exist. If NB01 was not re-run, NB02 stops with a clear "missing column" error instead of silently using bad prices.

**Nothing else in NB02, NB03 or NB04 needs a code change for Issue #1.**
- NB02 Step 2 (lines 9–68) calculates every feature from `Close`/`High`/`Low`/`Volume`, so it picks up adjusted values automatically.
- NB03 `STEP 1` line 7 builds the target `future_return_21d` from `close` (NB03 `STEP 0` line 12 lower-cases `Close` → `close`), so the target is fixed automatically.
- NB04 `STEP 1` reads `panel_phase4.csv`, which NB03 builds from `close`, so backtest returns are fixed automatically.
- NB03 `STEP 2` uses an explicit `FEATURES` list, so the extra raw/`adj_*` columns in the files do not leak into the model.

## 1.5 Decision you need to make: adjust for dividends or not?

| Option | `ADJUST_DIVIDENDS` | What the prices mean | Effect |
|---|---|---|---|
| Price-adjusted (default) | `False` | Like NSE/Yahoo "close adjusted for splits" | On every ex-dividend date, high-dividend stocks (COALINDIA, ONGC, PFC, RECLTD, POWERGRID, IOC, …) show a 2–5% drop that isn't a real loss. Their 21-day target and momentum are slightly understated around those dates. |
| Total-return-adjusted | `True` | Like a total-return index: dividends treated as reinvested | Returns measure what an investor actually earned. **Recommended for a strategy that ranks stocks by future return**, so dividend payers aren't penalised. |

My recommendation is `True`. It is left as `False` in the code above so you choose it deliberately.

## 1.6 What changes in the dataset: finance view

1. **Returns become economically true.** A shareholder of HDFCBANK did not lose 50% on 26 Aug 2025. They received one extra share for every share held. After adjustment, the return series shows the real ±1–2% move for that day.
2. **The bias against winners disappears.** Companies usually split or issue bonuses **after a big price run-up**. With unadjusted data, the stocks with the strongest uptrends (the ones a momentum strategy wants) are exactly the ones that show fake −50% to −90% "crashes". The model was learning "strong past performers suddenly collapse". That is a false anti-momentum lesson, and it works directly against your project goal.
3. **The 21-day target is corrected for about 36 events × 21 days ≈ 750 stock-days.** Around each ex-date, the label said "this stock will fall 50–90% next month". Those were among the most extreme labels in the dataset, and all of them were fake.
4. **Backtest P&L (NB04) is corrected.** If a stock was in the top 10 on its ex-date, the backtest booked a fake 50–90% loss on a 10% weight, i.e. a **−5% to −9% portfolio day that never happened**. Strategy returns, drawdown and Sharpe ratio in `results/` were all affected.
5. **Volatility and risk measures are corrected.** `vol_30d` and `vol_90d` contained a single −90% day for 30–90 days after each event. Those stocks looked like the riskiest in the index when they weren't.
6. **Technical levels become comparable over time.** RSI, MACD, EMA-20/50 crossovers and the 20-day range position were all distorted for weeks after each split. For example, the EMA-50 sat far above the new price, which falsely signalled a deep downtrend.
7. **Demergers are handled economically.** For TMPV, SIEMENS, RELIANCE (Jio Financial), ITC (ITC Hotels) and MOTHERSON, the part of the value that moved to the new company is treated as returned to the shareholder, not lost.

## 1.7 What changes in the dataset: data-science view

1. **New columns in every `data/raw/*.csv`:** `adj_openprice`, `adj_highprice`, `adj_lowprice`, `adj_lastprice`, `adj_closeprice`, `adj_averageprice`, `adj_totaltradedquantity`, `adj_deliverableqty`, `adj_factor`. Raw NSE columns are unchanged, except that they are now numeric (no commas) and sorted by date.
2. **New files:** `data/corporate_actions.csv` (the source), `data/adjustment_log.csv` (every factor applied, for audit), `data/large_moves_review.csv` (what's left to check).
3. **About 5,000 feature rows change value.** Each of the ~38 events affects up to 126 rows per feature (the longest window, `momentum_6m`). Total rows after NB02 ≈ 133,000, so ≈ 3–4% of rows change, but these are the most **extreme** values in each feature.
4. **Heavy tails disappear.** `ret_1d`, `ret_5d`, `ret_21d`, `momentum_*`, `ret_skew_*` and `ret_kurt_*` lose their fake −50% to −90% tails. Kurtosis and skew features in particular were dominated by single fake days.
5. **The target distribution changes.** `future_return_21d` loses roughly 750 extreme negative labels. Expect a smaller standard deviation and a less negative left tail. Correlation/IC numbers in NB03 (STEP 1 heatmap) will change.
6. **Volume features change sign around events.** Before the fix, volume jumped 2× to 10× permanently after a split or bonus. `volume_change`, `volume_zscore` and `volume_momentum` read that as a huge volume spike. After the fix, volume is continuous.
7. **Everything trained so far is invalid.** Saved models (`notebooks/data/models/`), `predictions.csv`, `panel_phase4.csv`, the results and the plots must be regenerated. Do not compare old and new metrics as like-for-like.
8. **Re-run order:** NB01 → NB02 → NB03 → NB04 → NB05.

---

# Issue #2: Neutral headlines scored as strongly negative

## 2.1 What is wrong

FinBERT (`ProsusAI/finbert`) outputs **three** labels: `positive`, `negative`, `neutral`. The `score` is the model's confidence in **that label**.

NB01 `# Step 3: News Fetch Function`, **line 42**:

```python
score = sent["score"] if sent["label"] == "positive" else -sent["score"]
```

Anything that is not "positive" gets a negative score. A headline that FinBERT is **94% sure is neutral** becomes **−0.94**, which looks like very bad news.

## 2.2 Evidence from your files

Sample: 1,094 headlines from 12 files in `data/news_raw/`.

| Measure | Value |
|---|---|
| Headlines with a negative score | **76.3%** |
| Headlines scored below −0.7 | **65.3%** |
| Mean score | **−0.457** |
| Median score | **−0.884** |

Examples of clearly neutral headlines that are stored as strongly negative:

| Ticker | Headline | Stored score |
|---|---|---|
| AXISBANK | "Axis Bank East Zonal Office … IFSC Code, MICR Code, Address & Phone Number" | −0.94 |
| BPCL | "Check out Bharat Petroleum Corporation Ltd's stock price (BPCL-IN) in real time" | −0.95 |
| SBIN | "SBI Q1 results date, time, analysts expectations: All you need to know" | −0.93 |
| INFY | "TCS, Infy, HCL Tech: How to trade IT stocks ahead of Q3 results? Key levels" | −0.94 |
| AXISBANK | "Axis Bank to Raise $6.4 Billion Through Debt and Equity" | −0.85 |
| RELIANCE | "Reliance Industries to launch Vantara University dedicated to wildlife conservation" | −0.84 |

## 2.3 How the fix is designed

- Ask FinBERT for **all three probabilities** on every headline.
- Score = **P(positive) − P(negative)**. This always falls between −1 and +1:
  - a neutral headline gives ≈ 0
  - a clearly positive one gives near +1
  - a clearly negative one gives near −1
  - a mixed one gives something in between
- Also save `p_positive`, `p_negative`, `p_neutral` and `sentiment_label`, so you can use or filter them later without re-running FinBERT.

## 2.4 Code changes for Issue #2 (exact locations)

### Change 2-A: NB01, `# Step 1: Load FinBERT`, replace lines 4–8

Replace:

```python
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    framework="pt",
)
```

with:

```python
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    framework="pt",
    top_k=None,        # return ALL three labels with probabilities, not just the top one
    truncation=True,   # long headlines are cut to the model's max length instead of erroring
)
```

### Change 2-B: NB01, `# Step 3: News Fetch Function`, insert after line 6

After `clean_text()` (lines 4–5) and the blank line 6, **insert** this new function:

```python
def score_headline(text):
    """FinBERT probabilities -> continuous score in [-1, +1]. Neutral ≈ 0."""
    out = sentiment_model(text)
    if out and isinstance(out[0], list):   # some transformers versions nest the output
        out = out[0]
    probs = {d["label"].lower(): float(d["score"]) for d in out}
    p_pos = probs.get("positive", 0.0)
    p_neg = probs.get("negative", 0.0)
    p_neu = probs.get("neutral", 0.0)
    return {
        "sentiment_label": max(probs, key=probs.get),
        "p_positive": p_pos,
        "p_negative": p_neg,
        "p_neutral": p_neu,
        "sentiment_score": p_pos - p_neg,
    }
```

### Change 2-C: NB01, `# Step 3: News Fetch Function`, replace lines 40–53

(Original line numbers. Make this change **before** Change 2-B, because 2-B inserts lines above and shifts these numbers.)

Replace:

```python
        try:
            sent = sentiment_model(text)[0]
            score = sent["score"] if sent["label"] == "positive" else -sent["score"]
        except:
            continue

        records.append(
            {
                "ticker": symbol,
                "date": date_pub.date(),
                "headline": text,
                "sentiment_score": score,
            }
        )
```

with:

```python
        try:
            s = score_headline(text)
        except Exception as e:
            print(f"  [sentiment failed] {symbol}: {e}")
            continue

        records.append(
            {
                "ticker": symbol,
                "date": date_pub.date(),
                "headline": text,
                **s,   # sentiment_label, p_positive, p_negative, p_neutral, sentiment_score
            }
        )
```

The bare `except:` is also replaced, so failures are printed instead of silently dropping headlines.

### Change 2-D: NB01, `# Step 3: News Fetch Function`, replace lines 55–56

(Original line numbers. Make this change **before** Changes 2-C and 2-B.)

Replace:

```python
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["headline"])
```

with:

```python
    if not records:
        return pd.DataFrame(columns=["ticker", "date", "headline", "sentiment_label",
                                     "p_positive", "p_negative", "p_neutral", "sentiment_score"])
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["headline"])
```

**Why:** this is a small existing bug on the same lines. If a symbol gets zero headlines, `pd.DataFrame([])` has no `headline` column, `drop_duplicates` raises `KeyError`, and Step 4 (lines 46–48) then marks the stock as **failed** even though its prices were saved correctly.

### Change 2-E: one-off re-scoring of the headlines you already have

**Important:** NB01 `Step 0` (lines 9–10) **deletes the whole `data/` folder**. If you simply re-run NB01, the old headlines are lost, and Google News will return a **different, newer** set. Your current files hold headlines going back to Feb 2026 and some older ones. To fix the scores **without losing those headlines**:

1. **Back up** `data/news_raw/` to another folder first (for example `data_backup/news_raw/`).
2. After Changes 2-A and 2-B, run this once (temporary cell, not part of the pipeline):

```python
# One-off: re-score existing headlines with the corrected method
BACKUP_NEWS = PROJECT_ROOT / "data_backup" / "news_raw"
for fp in sorted(BACKUP_NEWS.glob("*_news.csv")):
    df = pd.read_csv(fp)
    scored = pd.DataFrame([score_headline(h) for h in df["headline"].astype(str)])
    df = pd.concat([df[["ticker", "date", "headline"]], scored], axis=1)
    df.to_csv(fp, index=False)
print("Re-scored.")
```

3. After re-running NB01 (which recreates `data/news_raw/`), decide whether to **merge** the backed-up re-scored headlines with the new ones (concatenate, then `drop_duplicates(subset=["headline"])`). This keeps the most news history.

### Change 2-F: NB02, no code change required, but read this

- `# Step 2.5: News Feature Engineering` (lines 28–36) averages `sentiment_score` per day. It will now average a proper −1…+1 score. No change needed.
- `# Step 3: Process Each Ticker`, **lines 84–85** (`ffill(limit=3)` then `fillna(0)`): after the fix, **0 means two different things**: "no news" or "neutral news". Previously 0 could only mean "no news". The model can still tell them apart through `news_count` / `news_event`, which are already in the `FEATURES` list (NB03 STEP 2). **No change required.** Just don't drop `news_count`/`news_event` from the feature list later.
- `# Step 2: Technical Indicator Functions`, **line 52** (`sent_mom_interaction = sent_7d * momentum_1m`): no change, but its meaning changes (see 2.6).

## 2.5 What changes in the dataset: finance view

1. **Sentiment now measures tone, not coverage.** Before, any headline was roughly −0.9 unless clearly positive, so `news_sent` and `sent_7d` were essentially **"how much a stock was in the news" with a negative sign**. Heavily covered large caps (RELIANCE, HDFCBANK, INFY, TCS) looked permanently gloomy, and lightly covered names looked neutral. After the fix, routine items (results dates, price pages, branch addresses, "how to trade" pieces) score ≈ 0, and only real good or bad news moves the score.
2. **A real negative signal becomes possible.** "Profit falls 53%" and "IFSC code page" both scored about −0.95 before. Now the first stays strongly negative and the second goes to ≈ 0. Genuinely bad news can finally stand out.
3. **The bullish/bearish balance is restored.** The average headline went from −0.46 to something expected near 0, which matches reality: most financial headlines are informational.

## 2.6 What changes in the dataset: data-science view

1. **The distribution of `sentiment_score` changes shape.** Today it is two spikes, about 65% of values below −0.7 and the rest near +0.9. After the fix it becomes a continuous distribution centred near 0, with tails at ±1.
2. **These features change values:** `news_sent`, `sent_3d`, `sent_7d`, `sent_momentum`, `sent_surprise`, `sent_weighted`, `surprise_weighted`, `sent_mom_interaction`. `news_count`, `news_event`, `days_since_news`, `news_decay_weight`, `news_intensity` and `is_recent_news` are **unchanged** because they only count headlines.
3. **Collinearity drops.** Before, `news_sent` ≈ −0.9 on most news days, so it was strongly (negatively) correlated with `news_event` and `news_count`. The model was effectively seeing the same information twice. After the fix they carry different information.
4. **`sent_mom_interaction` becomes a real interaction.** Before, `sent_7d` was ≈ −0.9 most of the time, so `sent_7d × momentum_1m` ≈ `−0.9 × momentum_1m`: a sign-flipped copy of momentum. That may explain any odd importance this feature showed. After the fix it measures "momentum confirmed or contradicted by news tone".
5. **Feature importances, IC and the NB03 correlation heatmap for news features will change.** Retrain from scratch. Old SHAP/importance results for sentiment features should not be trusted.
6. **New columns in `data/news_raw/*_news.csv`:** `sentiment_label`, `p_positive`, `p_negative`, `p_neutral`. NB02 ignores them today, but they are useful later (for example, keep only headlines with `p_neutral < 0.5`).

### Known limit that this fix does NOT solve

Google News RSS returns only about 100 recent items per stock, so news features still exist mainly for the last few months (issue #3 from the review). Fixing the score makes those months **correct**. It does not fill 2020–2025.

---

# Addendum: two related findings from the same scan

These are not part of #1 or #2, but they cause the **same symptom** (fake price jumps or noise), so they are noted here.

**A. Multi-month gaps from the EQ-only filter** (NB01 Step 4, lines 26–33). When NSE moved a stock to the **BE** (trade-for-trade) series, those days were dropped. The next EQ day is then treated as one day after a months-old close:

| Symbol | Gap | Raw "1-day" move across the gap |
|---|---|---|
| CGPOWER | 2021-09-15 → 2022-01-04 (111 days) | **+129.9%** |
| ADANIPOWER | 2021-06-11 → 2021-09-20 (101 days) | −35.6% |
| ADANIENSOL | 2021-06-08 → 2021-09-13 (97 days) | +17.5% |
| DMART | 2020-03-03 → 2020-05-27 (85 days) | −1.5% |

There are 12 such gap rows in total, in ADANIENSOL, ADANIGREEN, ADANIPOWER, CGPOWER and DMART. Possible fixes: keep `BE` rows as well (`series in ["EQ", "BE"]`), or set returns to NaN when the gap is more than 10 calendar days. Step 6 above already lists them, because it prints `gap_days`.

**B. Junk headlines.** Pages like "IFSC Code, MICR Code, Address" and "Check out … stock price in real time" are not news. After Fix #2 they score ≈ 0, which limits the damage, but they still inflate `news_count`. A simple title filter (for example, drop titles containing "IFSC", "stock price in real time" or "share price live") would clean this up.

---

# Checklist (in order)

- [ ] Back up `data/news_raw/` (before anything else, because Step 0 deletes `data/`)
- [ ] 1-A: NB01 Imports cell: add `import re`, `import numpy as np`
- [ ] 1-B: NB01 Step 4: insert type cleaning and sorting after line 33
- [ ] 1-C: NB01: add new cell Step 4.5 (corporate actions + adjustment)
- [ ] 1-D: NB01 Step 5: add two calls after line 7
- [ ] 1-E: NB01: add new cell Step 6 (validation)
- [ ] 1-F: NB02 Step 3: replace `col_map` (lines 34–46)
- [ ] Decide `ADJUST_DIVIDENDS` (recommended: `True`)
- [ ] 2-D, 2-C, 2-B, 2-A: NB01 Steps 3 and 1, edited bottom-up
- [ ] 2-E: re-score the backed-up headlines (one-off)
- [ ] Run NB01. Check that Step 6 prints `OK` for every known event and review `large_moves_review.csv`
- [ ] Check `corporate_actions.csv` against table 1.2-A (every listed event should appear in `adjustment_log.csv`)
- [ ] Re-run NB02 → NB03 → NB04 → NB05 and treat all results as new (no like-for-like comparison with old metrics)


---
---

# PART 2: Pipeline split and Issues #3 to #6

> **Added 18 Oct 2026. Nothing above this line has been changed.**
> This part assumes every change from Issue #1 and Issue #2 has already been made.

## How Part 2 is organised

1. **Logic first.** Sections S, 3, 4, 5 and 6 explain *what* changes and *why*, including the finance and data-science effects. These sections contain no code.
2. **Code last.** The section "Code changes" has one block per step (**Changes in Step 0, 1, 2, 3, 4, 4.5, 5, 6**), then blocks for files outside notebook 01. The blocks contain only code.
3. **Your step numbers are kept.** The code moves out of the notebook into two scripts, but every section keeps its notebook step number, so "Step 4" still means "fetch prices".
4. **Tags in the code** show why a line exists: `[#3]`, `[#4]`, `[#5]`, `[#6]`, `[split]`, `[fix]`, `[Addendum A]`. Untagged code is carried over unchanged from Issue #1 / #2.
5. Steps 0 and 1 also have blocks because Issue #4 (Step 0) and the split (Step 1) require changes there. Step 4.5 has a block because the corporate-action download becomes incremental.

### What was tested and what was not

I assembled both scripts exactly from the blocks below and ran them on a copy of your project. NSE, Google News and FinBERT were simulated, and the simulated NSE served your real April price files.

**Confirmed in that test:**
- Your old-format files (text prices with commas, `30-Dec-2020` dates) are migrated automatically.
- An incremental run from 31 Jan to 10 Apr 2026 rebuilt exactly the same 1,559 rows as your full download.
- A second run on the same day downloads nothing.
- Corporate actions listed under an old symbol (`TATAMOTORS`) are applied to the new one (`TMPV`).
- All known events come out within ±3% after adjustment.
- An index addition and a removal are both detected.
- A former member with no data (HDFC) gives a warning, not a failed run.
- Legacy headlines are re-scored once with the Issue #2 method, junk and pre-2020 headlines are dropped, and duplicates are skipped.
- FinBERT is loaded only when there is something to score.
- `--full-refresh` backs up `data/raw` before re-downloading.
- The new notebook 02 and 03 snippets mask pre-coverage news correctly and mark index membership by date correctly.

**Not verifiable from here, so check on the first live run:**
- the column names NSE returns for corporate actions (as in Issue #1)
- whether Google News honours `after:` / `before:` in the optional backfill
- whether NSE accepts requests from the machine that runs the pipeline (see S.5)

---

## S. Split into an incremental price-update script and a separate news script

### S.1 Why split

| | Prices | News |
|---|---|---|
| Source | NSE (`nselib`) | Google News RSS + FinBERT |
| How often it should run | Every trading day after market close | Daily, or a few times a week |
| What the result is | Facts that can always be downloaded again | **An archive that cannot be downloaded again**: RSS only shows recent items |
| What happens if it fails | Must fail loudly, because model inputs depend on it | Can be retried tomorrow, since coverage simply has a gap |
| Heavy dependencies | None | `torch` + `transformers` (≈ 1–2 GB) |

Today both run in one loop (NB01 Step 4). A news timeout can mark a stock's **price** download as failed. A full run also needs FinBERT loaded even when you only want fresh prices. Splitting them gives each part its own schedule, failure rules and dependencies.

### S.2 New folder layout

```
ml-alpha-portfolio-model/
├── scripts/                         NEW
│   ├── pipeline_common.py           Step 0  (shared paths, logging, retry, safe writes, membership loader)
│   ├── update_prices.py             Steps 2, 4, 4.5, 6 + main()
│   ├── update_news.py               Steps 1, 3 + main()
│   └── run_daily_update.bat         optional, for Windows Task Scheduler
├── notebooks/
│   └── 01_domain_setup_data_collection.ipynb   becomes a 3-cell runner
├── data/
│   ├── raw/                         prices (raw NSE + adj_* columns)            Issue #1
│   ├── news_raw/                    append-only news archive                    Issue #3
│   ├── nifty100_membership.csv      point-in-time index membership              Issue #5
│   ├── nifty100_membership_backfill.csv   you create this once (5.2)            Issue #5
│   ├── corporate_actions.csv, adjustment_log.csv, large_moves_review.csv        Issue #1
│   ├── news_coverage.csv            per-ticker news coverage start              Issue #3
│   └── price_update_report.csv, news_update_report.csv,
│       data_quality_report.csv, failed_tickers.csv                              Issue #6
├── data_backup/                     NEW: backups made before any full refresh   Issue #4
└── logs/                            NEW: one log file per script per day        Issue #6
```

### S.3 Where each notebook step goes

| NB01 step | Goes to | What it becomes |
|---|---|---|
| Imports cell | top of each script (Step 5 block) | each script imports only what it needs; the price script no longer imports `torch`/`transformers` |
| Step 0: Project Setup | `scripts/pipeline_common.py` | shared by both scripts; no deletion (Issue #4) |
| Step 1: Load FinBERT | `scripts/update_news.py` | loaded lazily, only if there are headlines to score |
| Step 2: Fetch NIFTY 100 tickers | `scripts/update_prices.py` | membership history (Issue #5) |
| Step 3: News Fetch Function | `scripts/update_news.py` | company-name search, append-only archive, coverage (Issue #3) |
| Step 4: Fetch Price + News | `scripts/update_prices.py` | **prices only**, incremental, validated (Issue #6) |
| Step 4.5: Corporate Actions | `scripts/update_prices.py` | incremental corporate-action download |
| Step 5: Execute | `main()` in each script + notebook runner + scheduler | two independent entry points |
| Step 6: Validate | `scripts/update_prices.py` | quality gate that passes or fails the run |

### S.4 What one run does

**`python scripts/update_prices.py`**
1. Step 0: find the project root from the script's own location, create missing folders, open the log file.
2. Step 2: download today's Nifty 100 list and compare it with `nifty100_membership.csv`. Record additions and removals with today's date.
3. Step 4: for every symbol that was ever a member:
   - if the file is up to date, skip it
   - otherwise download only from *(last saved date − 7 days)*, merge, de-duplicate, validate, and write the file safely
4. Step 4.5: download corporate actions only for the last ~60 days, then recalculate the `adj_*` columns for every file from the raw columns (the Issue #1 rule).
5. Step 6: run the quality checks and write `data_quality_report.csv`. **The exit code is 0 (pass) or 1 (fail).**

**`python scripts/update_news.py`**
1. Step 0 as above. It reads the membership file written by the price script.
2. Step 3: for every current member, search Google News by **company name**, keep only headlines not already in the archive, score only those (FinBERT is loaded only now), append, and save.
3. Update `news_coverage.csv`. The exit code is 1 only if more than 20% of symbols failed.

Optional flags:
- `update_prices.py --full-refresh`: back up `data/raw` and download everything again.
- `update_prices.py --end-date YYYY-MM-DD`: update up to a given date.
- `update_news.py --symbols RELIANCE TCS`: run for selected symbols only.
- `update_news.py --backfill-start 2020-01-01`: the one-off history backfill in 3.2(f).

### S.5 Scheduling (the CI/CD part)

- **Order:** prices first, then news. The news script needs the membership file the price script writes.
- **Time:** after NSE publishes end-of-day data. A weekday evening such as 19:15 IST is a safe default.
- **Where to run it:** NSE often blocks requests from cloud data-centre IP addresses, including GitHub-hosted runners. Run the pipeline on a machine whose connection NSE accepts, which today is your own PC. Two options are in the Step 5 block:
  - **Windows Task Scheduler** with `run_daily_update.bat`. Simplest: no git or CI service needed.
  - **GitHub Actions on a self-hosted runner** (your PC registered as the runner). Use this if the project is on GitHub and you want run history in the Actions tab.
- **The data folder must persist between runs.** The news archive cannot be re-downloaded. With a self-hosted runner, the runner's checkout folder keeps `data/` between runs (`clean: false`). Keep `data/`, `data_backup/` and `logs/` in `.gitignore` and back them up separately.
- **Pass/fail:** both scripts return a non-zero exit code on critical problems, so Task Scheduler and GitHub Actions both show the run as failed. Details are in `logs/` and `data/*_report.csv`.
- **If you run Python from a virtual environment or conda,** replace `python` in the `.bat` file and the workflow with the full path to that environment's `python.exe`.

### S.6 Notebook 01 after the split

NB01 becomes a short runner (Step 5 block) that calls the two scripts with `runpy`. The notebook and the pipeline then always run **the same code**. There is no second copy to keep in sync.

### S.7 What changes in the dataset

- **Finance view:** the split itself changes no numbers. Its effect is timeliness: prices can update every trading day without waiting for news, so the model and portfolio always see the latest close.
- **Data-science view:** reproducibility. Every run leaves a log, a run report and a quality report, and a failed run no longer half-overwrites files (Issue #6, safe writes). Price files are always complete, validated and sorted. News files carry `fetched_at`, so you can always tell when a headline entered the dataset.

---

# Issue #3: News cannot support a backtest from 2020

## 3.1 What is wrong

1. **Google News RSS returns only about 100 of the most relevant recent items per query.** Your April run produced headlines mostly from February–April 2026, plus a few stray old ones (RELIANCE 2006, JINDALSTEL 2017). For 2020–2025 the news features are empty, so NB02 fills them with 0.
2. **The search uses the NSE symbol, not the company name.** NB01 Step 3, line 11 searches for `CUMMINSIND stock OR CUMMINSIND earnings …`. Headlines say "Cummins India", not "CUMMINSIND". That is a main reason why 15 stocks got 10 or fewer headlines: ADANIENSOL 6, BAJAJFINSV 9, BAJAJHLDNG 5, CUMMINSIND 3, EICHERMOT 9, GODREJCP 10, INDHOTEL 9, JINDALSTEL 2, NESTLEIND 6, PIDILITIND 3, SHREECEM 3, SHRIRAMFIN 8, TATACONSUM 4, TORNTPHARM 5, UNITDSPR 6.
3. **Every run overwrites the news files** (NB01 Step 4, line 43). Headlines that have dropped out of the RSS feed are lost for good, so history can never build up.
4. **Junk items** ("IFSC code", "stock price in real time") are counted as news (Addendum B).
5. **The model can't tell "no news" from "no data".** NB02 Step 3 (lines 84–87) writes 0 for every day before the first headline. In 2020–2025, "0" means "we didn't collect news then", but the model reads it as "neutral, quiet news day".

## 3.2 How the fix works

**(a) Search by company name.** The NSE Nifty 100 file already has a `Company Name` column. It is saved in the membership file (Step 2) and cleaned (`Cummins India Ltd.` → `Cummins India`). The search becomes `"Cummins India" (site:reuters.com OR …)`. `NEWS_QUERY_OVERRIDES` lets you set the phrase by hand for odd names, for example `M&M` → `Mahindra & Mahindra`.

**(b) Append-only archive.** Each run reads the existing `{SYMBOL}_news.csv`, adds only headlines not already there, and saves. Nothing is ever removed except junk and out-of-range dates. New columns:
- `fetched_at`: the day the headline entered the archive (legacy rows get the file's last-modified date, i.e. 12 Apr 2026)
- `fetch_mode`: `legacy` / `live` / `backfill`
- `source`: the publisher

**(c) Only new headlines go through FinBERT, in batches of 16.** Legacy rows scored the old way (no `p_neutral` column) are **re-scored automatically once** with the Issue #2 method. This replaces the manual one-off step in Change 2-E. Backing up `data/news_raw/` first is still a good precaution.

**(d) Clean dates and junk.** Only headlines dated between 2020-01-01 and today are kept, and titles matching the junk patterns are dropped.

**(e) Coverage file.** `news_coverage.csv` stores, per ticker, the date from which news can be trusted to be "collected":
- `coverage_start` = first `fetched_at` − 30 days (RSS returns roughly the last few weeks for well-covered names), or
- the backfill start date, if a backfill was done.

**(f) Optional one-off backfill (not for the daily job).** Google News search accepts `after:` and `before:` date filters. `update_news.py --backfill-start 2020-01-01` walks month by month and asks for each window separately.
- This is about 80 windows × 100 symbols ≈ 8,000 requests, roughly 3–4 hours with the built-in pause. Run it once on your PC, not in the scheduled job.
- **Test it first on 2–3 symbols.** Google may cap or throttle results, and older windows return fewer articles (some are no longer online).
- After a backfill, `coverage_start` moves back to the backfill date for that ticker.

**(g) Model side (notebooks 02 and 03).**
- NB02: every news feature is set to **NaN** (unknown) before that ticker's `coverage_start`, and a new 0/1 column `has_news_coverage` is added. LightGBM handles NaN natively as "missing", which is what it truly is.
- NB03: a switch `USE_NEWS_FEATURES` (default **False**) keeps news features out of the main model until there is at least ~2 years of coverage. You can still switch it on to test news features on the covered period only. `sent_mom_interaction` moves into the news group because it is built from `sent_7d`.

## 3.3 What changes in the dataset: finance view

- The model no longer learns from 5 years of fake "quiet news" days. Until now, the news features mostly separated **Feb–Apr 2026 from everything before**, a time effect rather than a stock-picking signal. Any apparent alpha from them in the 2020–2025 backtest was not real.
- Coverage becomes much more even across stocks. Mid-caps and companies with unusual symbols (Cummins, Pidilite, Shree Cement, Jindal Steel) get their actual news instead of 2–6 random items.
- History starts building today. Every day the pipeline runs adds headlines that RSS would otherwise lose. In 1–2 years you will have a proper point-in-time news history for a real out-of-sample test of news alpha.

## 3.4 What changes in the dataset: data-science view

- **New columns in `data/news_raw/*_news.csv`:** `source`, `fetched_at`, `fetch_mode` (plus the Issue #2 probability columns, now present on every row).
- **Rows:** pre-2020 and junk rows are removed. Rows accumulate over time instead of being replaced.
- **NB02 features:** news features become **NaN before coverage** instead of 0, and `has_news_coverage` is added. Expect roughly 95%+ NaN in news columns at first. That is correct: it is the true amount of missing information.
- **NB03:** with `USE_NEWS_FEATURES = False` the main model uses 18 technical features. The 13 news features plus `sent_mom_interaction` and `has_news_coverage` are kept aside. Expect the feature importance chart to change, and compare IC with and without news on the covered period only.
- **Runtime:** a daily news run scores only a handful of new headlines per stock instead of ~100.

---

# Issue #4: Step 0 deletes the whole data folder

## 4.1 What is wrong

- NB01 Step 0, lines 9–10: `if DATA_DIR.exists(): shutil.rmtree(DATA_DIR)` deletes **all** of `data/`: features, portfolio inputs, the 94 MB `combined_features.csv`, and the news archive that Issue #3 makes irreplaceable.
- NB01 Step 0, line 5: `PROJECT_ROOT = Path().resolve().parent` depends on the folder the notebook is started from. Started from `notebooks/`, it is the project. Started from the project root, it becomes `ml_alpha_portfolio/`, and Step 0 deletes `ml_alpha_portfolio/data`, **outside this project**.
- The same working-directory guess is in NB02 Step 1 line 5, NB03 CONFIGURATION line 8, NB04 STEP 0 line 12 and NB05 cell 2 line 6. NB02's own Step 0 (`shutil.rmtree(FEATURE_DIR)`, lines 7–9) has the same risk.

## 4.2 How the fix works

1. **Find the project root by its content, not by guessing.** Walk up from the script (or the notebook's folder) until a folder contains both `requirements.txt` and `src/ml_alpha/`. If none is found, stop with an error instead of guessing.
2. **Never delete.** `ensure_dirs()` only creates missing folders.
3. **A full refresh is a backup plus a re-download, not a delete.** `--full-refresh` copies `data/raw` to `data_backup/raw_YYYYMMDD_HHMMSS/` and then downloads everything again, overwriting file by file.
4. **A guard on every write.** `assert_inside_project()` refuses to write anywhere outside the project root, so a path mistake becomes an error instead of damage.
5. **Safe writes.** Every file is written to `name.csv.tmp` first and then swapped in. If a run is killed halfway, the old file stays intact instead of a half-written one replacing it.
6. **Notebooks 02–05** use the same root-finder (blocks at the end).

## 4.3 What changes in the dataset

- **Finance view:** no numbers change. What changes is that history is protected. The news archive and the membership history (Issue #5) can't be rebuilt from public sources once lost.
- **Data-science view:** runs become repeatable regardless of where they are started. A crash can no longer leave a corrupt file, and every full refresh leaves a dated backup, so you can diff "before vs after" for any change.

---

# Issue #5: Survivorship bias (today's Nifty 100 used for all of history)

## 5.1 What is wrong

- NB01 Step 2 downloads **today's** list and uses it for 2020–2026. Stocks that were in the Nifty 100 earlier but have since dropped out are missing.
  - Your older raw folder (`notebooks/data/raw`, March 2026) has 6 such symbols: HAVELLS, ICICIGI, LICI, NAUKRI, JSWENERGY, BAJAJHFL.
  - Going back to 2020 there are more. For example, HDFC Ltd. was in the index until it merged into HDFC Bank in July 2023.
- **Today's members are, by construction, the winners.** Stocks enter the index after rising and leave after falling. A backtest that only picks among today's members gets those winners for free.
- Recent entrants (TATACAP from Oct 2025, TMCV from Nov 2025, ENRIN from Jul 2025, HYUNDAI from Oct 2024) are treated as if they had always been index members.
- NB02 Step 6 (cell "Filter Stocks With Minimum 1000 Rows", line 5) deletes every stock with under 1,000 rows. That removes recent listings entirely, which is another selection effect.

Confirmed from your files: **NSE returns pre-rename history under the new symbol.** ETERNAL (formerly ZOMATO) has data from its 2021-07-23 listing, and TMPV and LTM have data from 2020-01-01. So a renamed stock needs only its current symbol.

## 5.2 How the fix works

1. **Membership history file `data/nifty100_membership.csv`** with one row per membership period: `symbol, company_name, industry, start_date, end_date, source`. An empty `end_date` means "still a member".
2. **One-off backfill, done by you:** `data/nifty100_membership_backfill.csv`, same columns.
   - NSE Indices publishes every Nifty 100 constituent change in its index-rebalancing press releases (niftyindices.com). Changes normally take effect after the last trading day of March and September.
   - Since 2020 that is about 12 reviews with a handful of changes each.
   - Start from the current list with `start_date = 2020-01-01`, and for every change since then either set `end_date` (removed) or add a row (added).
   - Example rows: `HDFC,Housing Development Finance Corporation Ltd.,Financial Services,2020-01-01,2023-07-13` and `TATACAP,Tata Capital Ltd.,Financial Services,<date added>,`.
   - The first price run reads this file and saves it as `nifty100_membership.csv`.
3. **If no backfill is provided,** the first run seeds the history with today's list from 2020. The quality report then shows a **warning** on every run (`membership_history`) until you add the backfill, so the remaining bias is never silent.
4. **Automatic tracking from now on.** Every price run compares NSE's current list with the open rows. Removals get `end_date = today` and additions get a new row (`source = detected_change`). Running daily keeps these dates within a day of the real change.
5. **Prices for every stock that was ever a member.** Removed stocks are downloaded up to 45 days after removal, enough to finish the 21-day target window and any open position.
   - A stock NSE no longer serves (e.g. a merged company) is reported as a **warning**, not a failed run.
   - `SYMBOL_RENAMES` maps old symbols to new ones in the membership file and the corporate-actions list. `ZOMATO→ETERNAL` and `TATAMOTORS→TMPV` are known renames. `LTIM→LTM` is inferred from your data (LTM has history from 2020 and LTIM is not in today's list), so confirm it.
6. **Point-in-time universe in notebook 03.**
   - STEP 0 adds `in_index` (1 if the stock was a member on that date).
   - STEP 1 keeps only `in_index == 1` rows **after** the 21-day target is calculated. If the filter came first, `shift(-21)` would jump across the gap when a stock leaves the index.
   - Training groups, cross-sectional ranks, IC and the final top-10 all then use the stocks actually investable on each date.
7. **NB02 minimum-history filter:** `MIN_DAYS` goes from 1000 to 63. Rows without enough history are already removed by the existing `dropna` on `momentum_6m` / `vol_90d`, so a large row minimum only throws away recent entrants.
8. **Things to update by hand in notebook 03:** the hard-coded `sector_map` (cell starting `sector_map = {`) must include every symbol in the membership file. Otherwise added or removed stocks get `sector = NaN`.

**Known remaining limit:** in NB04, if a held stock leaves the index in the middle of a holding period, its daily returns after the exit date are missing in `df` and count as 0 for those days. Index exits happen twice a year and rarely hit a top-10 holding. Fixing it needs NB04 to take returns from the full price panel; I've left that out to keep this change contained.

## 5.3 What changes in the dataset: finance view

- **Backtest returns will most likely go down,** possibly by a lot. That is the bias disappearing, not the strategy getting worse. With today's list, every stock the model could pick in 2020 was guaranteed to still be large in 2026.
- **Momentum results become honest.** Stocks that ran up and then collapsed out of the index (the classic momentum crash) are back in the history, so the model has to learn to avoid them.
- **Universe size is correct over time.** It is always about 100 members on any date, never "today's 100 projected backwards".

## 5.4 What changes in the dataset: data-science view

- **More files in `data/raw/`:** one per former member, about 15–30 extra depending on the backfill.
- **New file `nifty100_membership.csv`;** NB03 gets a new column `in_index`.
- **Row counts:** the training panel **drops** rows for current members before they joined, and **gains** rows for former members while they were members. Daily cross-sections are about 100 stocks each.
- **Fewer stocks removed by NB02 Step 6** (MIN_DAYS 63 instead of 1000). HYUNDAI, JIOFIN and the 2025 listings now appear from the date their features become valid.
- **Ranking labels (`cs_rank`) and IC are calculated within the true daily universe,** so earlier and new metrics are not comparable.

---

# Issue #6: Pipeline issues

## 6.1 Incremental download (NB01 Step 5 re-downloaded 6 years every time)

- For each symbol, the last saved date is read and only *(last date − 7 days)* to today is downloaded.
  - The 7-day overlap re-downloads the last week, so late NSE corrections are picked up.
  - The rows are then merged and de-duplicated by date, with the newest download winning.
- A symbol already up to date is skipped without any request.
- Removed members stop being downloaded 45 days after removal (Issue #5).
- Corporate actions are downloaded only for the last ~60 days and merged into the saved list (Step 4.5).
- In the test, this incremental path reproduced your full 2020–2026 files exactly.

## 6.2 Speed: a correction to my earlier review

In my first review I wrote that Step 5 was slow *mostly because of news and FinBERT*. **The timestamps of your April files show the opposite.** The 100 symbols took 29.1 minutes:
- about **20 minutes downloading prices** (median 11.6 s per symbol, including the 1 s pause)
- about **9 minutes on news + FinBERT** (median 6.8 s per symbol)

The price part was slow because it downloaded 6 years per symbol on every run. After the changes:
- prices: one small request per symbol per day (1–2 minutes in total on a normal day)
- news: only new headlines are scored, in batches, and FinBERT is not even loaded if nothing is new

## 6.3 Sorting, number types and validation at save time

Every price file is cleaned before saving:
- lower-case columns
- numbers without commas
- real dates (saved as `YYYY-MM-DD`)
- sorted by date
- one row per date

Then it is checked:
- **Critical (file is not written, symbol reported as `invalid`):** required columns missing, duplicate dates, unsorted dates, close ≤ 0.
- **Warning (written, but noted in the report):** missing closes, high < low, close outside the high–low range.

**Addendum A is included:** `ALLOWED_SERIES = ["EQ", "BE"]` keeps trade-for-trade (BE) days, so the multi-month gaps in CGPOWER, ADANIPOWER, ADANIENSOL, ADANIGREEN and DMART disappear on the next `--full-refresh`. Set it back to `["EQ"]` if you prefer EQ-only data. Step 6 still reports any remaining gap in `large_moves_review.csv` (`gap_days` column).

## 6.4 Retries, errors and logging

- `with_retry()` retries every network call (NSE prices, NSE list, corporate actions, Google News) up to 4 times, waiting 2 s, 4 s and 8 s in between.
- Every bare `except:` is gone. Errors are written to the log with the symbol and reason, and the symbol is marked `failed` in the run report.
- `logs/update_prices_YYYYMMDD.log` and `logs/update_news_YYYYMMDD.log` keep every run.
- **Quality gate (Step 6).** The run **fails** (exit code 1) if:
  - a known corporate-action date is still broken,
  - a current member has no price file,
  - more than 5% of current members failed, or
  - the index size is outside 95–105.

  It **warns** on stale symbols (more than 5 days behind the newest date), former members without data, remaining large moves, and seeded-only membership.

## 6.5 `requirements.txt` was incomplete

- **Added:** `nselib`, `requests`, `feedparser`, `transformers`, `torch` (notebook 01), plus everything notebooks 02–05 import that was missing: `tqdm`, `lightgbm`, `seaborn`, `plotly`, `sentence-transformers`, `openai`, `python-dotenv`, `nb_black`.
- **Minimum versions where the new code needs them:** `transformers>=4.30` for `top_k=None` in the pipeline constructor. The existing `pandas==2.1.4` already supports `format="mixed"`.
- After installing, run `pip freeze` in your working environment and pin the exact versions of the new packages, so the scheduled job always builds the same environment.

## 6.6 The stale `notebooks/data/` folder and files written into `notebooks/`

- `notebooks/data/` (features, models, news_raw, raw from March 2026) is left over from an older version of the notebooks. Nothing in the current notebooks reads it. Move it to `data_backup/notebooks_data_old/` (or delete it) so nobody mistakes it for current data.
- Four notebook cells write results to whichever folder the notebook runs in, which is how stray files end up in `notebooks/`. They now write to `results/`:
  - NB02: `phase2_metadata.csv`
  - NB03: `phase3_top10_next_month_5Apr.csv` (now named with the latest data date)
  - NB04: `portfolio_decisions.csv`
  - NB05: `latest_portfolio_explanations.csv`

## 6.7 Local data stops at 10 Apr 2026

There is no separate code change. The first run of the new price script fills the gap. Make it a **full refresh** (`--full-refresh`) because:
- the gap is 6 months anyway,
- it picks up the BE-series days (Addendum A),
- it re-applies every corporate action since April (Step 4.5 fetches them all on a first run).

## 6.8 What changes in the dataset

- **Finance view:** prices are current to the last trading day on every run, and corporate actions after April 2026 are applied automatically. Delisting gaps in trade-for-trade periods are filled (Addendum A), which removes fake jumps such as CGPOWER's +130% "one-day" return.
- **Data-science view:**
  - Files are typed, sorted and de-duplicated on disk, so NB02's own comma-stripping becomes a harmless no-op.
  - Each run leaves an auditable trail (`price_update_report.csv`, `news_update_report.csv`, `data_quality_report.csv`, logs).
  - A failed or half-finished run never corrupts existing data.
  - The environment can be rebuilt from `requirements.txt`.

---

# Order of work for Part 2

1. Back up `data/` and `notebooks/data/` once by hand.
2. Create `scripts/` and assemble the three files from the blocks: **Step 0** → `pipeline_common.py`; **Step 5 (prices)** with Steps 2, 4, 4.5, 6 pasted where marked → `update_prices.py`; **Step 5 (news)** with Steps 1, 3 pasted where marked → `update_news.py`.
3. Replace `requirements.txt` and reinstall.
4. *(Recommended)* Create `data/nifty100_membership_backfill.csv` (5.2 step 2).
5. Run `python scripts/update_prices.py --full-refresh`. Check `data/data_quality_report.csv`, `large_moves_review.csv` and `corporate_actions.csv` (confirm the column names, as in Issue #1).
6. Run `python scripts/update_news.py`. Check `news_update_report.csv` and `news_coverage.csv`.
7. *(Optional)* Test the backfill on 2–3 symbols: `python scripts/update_news.py --symbols RELIANCE CUMMINSIND --backfill-start 2025-01-01`. Run it for all symbols only if the results look right.
8. Replace notebook 01 with the runner, and apply the notebook 02–05 blocks.
9. Re-run NB02 → NB03 → NB04 → NB05 and treat all results as new.
10. Schedule the daily run (Step 5 block: Task Scheduler or GitHub Actions on your PC).

---

# Code changes

## Changes in Step 0

```python
# =====================================================================
# FILE: scripts/pipeline_common.py
# Step 0: Project Setup (shared by update_prices.py and update_news.py)
# =====================================================================
import logging
import os
import shutil
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def find_project_root(start):  # [#4]
    start = Path(start).resolve()
    for cand in [start, *start.parents]:
        if (cand / "requirements.txt").is_file() and (cand / "src" / "ml_alpha").is_dir():
            return cand
    raise RuntimeError(
        f"Project root not found above {start} "
        "(expected a folder containing requirements.txt and src/ml_alpha/)."
    )


PROJECT_ROOT = find_project_root(Path(__file__).parent)  # [#4]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NEWS_DIR = DATA_DIR / "news_raw"
BACKUP_DIR = PROJECT_ROOT / "data_backup"  # [#4]
LOG_DIR = PROJECT_ROOT / "logs"  # [#6]

MEMBERSHIP_FILE = DATA_DIR / "nifty100_membership.csv"  # [#5]
MEMBERSHIP_BACKFILL_FILE = DATA_DIR / "nifty100_membership_backfill.csv"  # [#5]
CA_FILE = DATA_DIR / "corporate_actions.csv"  # [#1]
ADJ_LOG_FILE = DATA_DIR / "adjustment_log.csv"  # [#1]
NEWS_COVERAGE_FILE = DATA_DIR / "news_coverage.csv"  # [#3]
PRICE_RUN_REPORT = DATA_DIR / "price_update_report.csv"  # [#6]
NEWS_RUN_REPORT = DATA_DIR / "news_update_report.csv"  # [#6]
QUALITY_REPORT = DATA_DIR / "data_quality_report.csv"  # [#6]
FAILED_FILE = DATA_DIR / "failed_tickers.csv"

START_DATE = date(2020, 1, 1)
MEMBERSHIP_COLUMNS = ["symbol", "company_name", "industry", "start_date", "end_date", "source"]  # [#5]

SYMBOL_RENAMES = {  # [#5] old NSE symbol -> current symbol
    "ZOMATO": "ETERNAL",
    "TATAMOTORS": "TMPV",
    "LTIM": "LTM",
}


def assert_inside_project(path):  # [#4]
    path = Path(path).resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise RuntimeError(f"Refusing to write to {path}: it is outside {PROJECT_ROOT}")
    return path


def ensure_dirs():  # [#4] replaces shutil.rmtree(DATA_DIR)
    for d in (DATA_DIR, RAW_DIR, NEWS_DIR, BACKUP_DIR, LOG_DIR):
        assert_inside_project(d).mkdir(parents=True, exist_ok=True)


def backup_folder(folder):  # [#4]
    folder = assert_inside_project(folder)
    if not folder.exists() or not any(folder.iterdir()):
        return None
    dest = BACKUP_DIR / f"{folder.name}_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copytree(folder, dest)
    return dest


def setup_logging(name):  # [#6]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(LOG_DIR / f"{name}_{date.today():%Y%m%d}.log", encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)
    for h in (file_handler, console_handler):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def with_retry(fn, *args, tries=4, base_delay=2.0, what="call", logger=None, **kwargs):  # [#6]
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == tries:
                raise
            wait = base_delay * 2 ** (attempt - 1)
            if logger:
                logger.warning(f"{what} failed (attempt {attempt}/{tries}): {e} -> retry in {wait:.0f}s")
            time.sleep(wait)


def atomic_write_csv(df, path, **kwargs):  # [#6]
    path = assert_inside_project(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)


def load_membership():  # [#5]
    src = MEMBERSHIP_FILE if MEMBERSHIP_FILE.exists() else MEMBERSHIP_BACKFILL_FILE
    if not src.exists():
        return None
    m = pd.read_csv(src)
    for c in MEMBERSHIP_COLUMNS:
        if c not in m.columns:
            m[c] = pd.NA
    m["symbol"] = m["symbol"].astype(str).str.strip().replace(SYMBOL_RENAMES)
    m["start_date"] = pd.to_datetime(m["start_date"])
    m["end_date"] = pd.to_datetime(m["end_date"])
    m["source"] = m["source"].fillna("backfill" if src == MEMBERSHIP_BACKFILL_FILE else "unknown")
    return m[MEMBERSHIP_COLUMNS]
```

## Changes in Step 1

```python
# =====================================================================
# FILE: scripts/update_news.py
# Step 1: Load FinBERT (lazy: only loaded when there are headlines to score)
# =====================================================================
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 16  # [#6]
_sentiment_model = None


def get_sentiment_model():  # [split] [#6]
    global _sentiment_model
    if _sentiment_model is None:
        from transformers import pipeline

        _sentiment_model = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            framework="pt",
            top_k=None,  # [#2]
            truncation=True,  # [#2]
        )
    return _sentiment_model
```

## Changes in Step 2

```python
# =====================================================================
# FILE: scripts/update_prices.py
# Step 2: Nifty 100 universe with membership history (point-in-time)
# =====================================================================
NIFTY100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
EXPECTED_INDEX_SIZE = (95, 105)  # [#5]
REMOVED_FETCH_BUFFER_DAYS = 45  # [#5]


def fetch_current_nifty100(logger):  # [#5] [#6]
    df = with_retry(pd.read_csv, NIFTY100_URL, what="Nifty 100 list", logger=logger)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Symbol": "symbol", "Company Name": "company_name", "Industry": "industry"})
    df["symbol"] = df["symbol"].astype(str).str.strip().replace(SYMBOL_RENAMES)
    lo, hi = EXPECTED_INDEX_SIZE
    if not lo <= len(df) <= hi:
        raise ValueError(f"Nifty 100 list has {len(df)} rows (expected {lo}-{hi}); refusing to update membership")
    return df[["symbol", "company_name", "industry"]].drop_duplicates("symbol")


def update_membership(today, logger):  # [#5]
    current = fetch_current_nifty100(logger)
    today_ts = pd.Timestamp(today)
    m = load_membership()

    if m is None:
        m = current.assign(start_date=pd.Timestamp(START_DATE), end_date=pd.NaT, source="seed_current_list")
        logger.warning(
            "No membership history found: seeded with TODAY's Nifty 100 list from START_DATE. "
            "Survivorship bias remains until data/nifty100_membership_backfill.csv is provided."
        )
    else:
        open_mask = m["end_date"].isna()
        open_syms = set(m.loc[open_mask, "symbol"])
        cur_syms = set(current["symbol"])
        removed = sorted(open_syms - cur_syms)
        added = sorted(cur_syms - open_syms)
        if removed:
            m.loc[open_mask & m["symbol"].isin(removed), "end_date"] = today_ts
            logger.info(f"Index removals detected: {removed}")
        if added:
            new_rows = current[current["symbol"].isin(added)].assign(
                start_date=today_ts, end_date=pd.NaT, source="detected_change"
            )
            m = pd.concat([m, new_rows[MEMBERSHIP_COLUMNS]], ignore_index=True)
            logger.info(f"Index additions detected: {added}")
        info = current.set_index("symbol")
        open_mask = m["end_date"].isna()
        for col in ("company_name", "industry"):
            m.loc[open_mask, col] = m.loc[open_mask, "symbol"].map(info[col]).fillna(m.loc[open_mask, col])

    m = m.sort_values(["symbol", "start_date"]).reset_index(drop=True)
    atomic_write_csv(m, MEMBERSHIP_FILE, date_format="%Y-%m-%d")
    logger.info(f"Membership: {m['end_date'].isna().sum()} current members, {m['symbol'].nunique()} symbols ever")
    return m


def fetch_plan(membership, today):  # [#5] symbol -> last date that needs prices
    plan = {}
    for sym, g in membership.groupby("symbol"):
        if g["end_date"].isna().any():
            plan[sym] = today
        else:
            last_needed = (g["end_date"].max() + pd.Timedelta(days=REMOVED_FETCH_BUFFER_DAYS)).date()
            plan[sym] = min(today, last_needed)
    return plan


def members_on(membership, when):  # [#5]
    when = pd.Timestamp(when)
    mask = (membership["start_date"] <= when) & (
        membership["end_date"].isna() | (membership["end_date"] > when)
    )
    return sorted(membership.loc[mask, "symbol"].unique())
```

## Changes in Step 3

```python
# =====================================================================
# FILE: scripts/update_news.py
# Step 3: News fetch, scoring and append-only archive
# =====================================================================
NEWS_START = date(2020, 1, 1)  # [#3]
NEWS_REQUEST_PAUSE = 1.5  # [#6]
COVERAGE_LOOKBACK_DAYS = 30  # [#3]
NEWS_SITES = [
    "reuters.com", "bloomberg.com", "cnbc.com", "economictimes.indiatimes.com",
    "business-standard.com", "livemint.com", "moneycontrol.com", "ndtvprofit.com",
    "financialexpress.com", "thehindubusinessline.com",
]
NEWS_QUERY_OVERRIDES = {  # [#3] symbol -> exact search phrase when the company name does not work well
    "M&M": "Mahindra & Mahindra",
}
JUNK_PATTERN = re.compile(  # [#3]
    r"ifsc|micr code|stock price .*in real time|share price live|share price today|price chart",
    flags=re.IGNORECASE,
)
NEWS_COLUMNS = [  # [#2] [#3]
    "ticker", "date", "headline", "source", "fetched_at", "fetch_mode",
    "sentiment_label", "p_positive", "p_negative", "p_neutral", "sentiment_score",
]


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def company_search_name(company_name, symbol):  # [#3]
    if company_name is None or pd.isna(company_name) or not str(company_name).strip():
        return symbol
    name = re.sub(r"\((india)\)", " ", str(company_name), flags=re.IGNORECASE)
    name = re.sub(r"\b(ltd|limited)\b\.?", " ", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip(" .,")


def build_news_url(search_name, after=None, before=None):  # [#3]
    q = f'"{search_name}" (' + " OR ".join(f"site:{s}" for s in NEWS_SITES) + ")"
    if after is not None:
        q += f" after:{after:%Y-%m-%d}"
    if before is not None:
        q += f" before:{before:%Y-%m-%d}"
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-IN&gl=IN&ceid=IN:en"


def _http_get(url):
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.content


def fetch_feed_items(url, symbol, fetched_at, fetch_mode, logger):  # [#3] [#6]
    content = with_retry(_http_get, url, what=f"news {symbol}", logger=logger)
    feed = feedparser.parse(content)
    rows = []
    for e in feed.entries:
        if not getattr(e, "published_parsed", None):
            continue
        src = e.get("source", {})
        rows.append(
            {
                "ticker": symbol,
                "date": date(*e.published_parsed[:3]).isoformat(),
                "headline": clean_text(e.title),
                "source": src.get("title", "") if hasattr(src, "get") else "",
                "fetched_at": fetched_at.isoformat(),
                "fetch_mode": fetch_mode,
            }
        )
    return pd.DataFrame(rows, columns=["ticker", "date", "headline", "source", "fetched_at", "fetch_mode"])


def filter_news(df, latest_day):  # [#3]
    if df.empty:
        return df
    d = pd.to_datetime(df["date"], errors="coerce")
    keep = d.notna() & (d >= pd.Timestamp(NEWS_START)) & (d <= pd.Timestamp(latest_day))
    keep &= ~df["headline"].astype(str).str.contains(JUNK_PATTERN)
    return df[keep]


def score_headlines(texts):  # [#2] batched
    if not texts:
        return []
    outputs = get_sentiment_model()(list(texts), batch_size=FINBERT_BATCH_SIZE)
    results = []
    for out in outputs:
        if isinstance(out, dict):
            out = [out]
        probs = {d["label"].lower(): float(d["score"]) for d in out}
        p_pos, p_neg, p_neu = probs.get("positive", 0.0), probs.get("negative", 0.0), probs.get("neutral", 0.0)
        results.append(
            {
                "sentiment_label": max(probs, key=probs.get),
                "p_positive": p_pos,
                "p_negative": p_neg,
                "p_neutral": p_neu,
                "sentiment_score": p_pos - p_neg,
            }
        )
    return results


def _headline_key(s):
    return s.astype(str).str.lower().str.strip()


def load_news_archive(path):  # [#3]
    if not path.exists():
        return pd.DataFrame(columns=NEWS_COLUMNS)
    df = pd.read_csv(path)
    if "fetched_at" not in df.columns:  # files written before this change
        df["fetched_at"] = date.fromtimestamp(path.stat().st_mtime).isoformat()
    if "fetch_mode" not in df.columns:
        df["fetch_mode"] = "legacy"
    for c in NEWS_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df[NEWS_COLUMNS]


def merge_score_save(symbol, archive, fresh, latest_day):  # [#2] [#3]
    known = set(_headline_key(archive["headline"]))
    fresh = fresh[~_headline_key(fresh["headline"]).isin(known)]
    fresh = fresh.drop_duplicates(subset="headline")
    for c in NEWS_COLUMNS:
        if c not in fresh.columns:
            fresh[c] = np.nan
    parts = [p for p in (archive, fresh[NEWS_COLUMNS]) if not p.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=NEWS_COLUMNS)
    combined = filter_news(combined, latest_day)

    to_score = combined["p_neutral"].isna()  # new rows + rows scored with the old (#2) method
    if to_score.any():
        scores = pd.DataFrame(score_headlines(combined.loc[to_score, "headline"].tolist()),
                              index=combined.index[to_score])
        for c in scores.columns:
            combined[c] = combined[c].astype(object)
            combined.loc[to_score, c] = scores[c]

    combined = combined.sort_values(["date", "headline"]).reset_index(drop=True)
    atomic_write_csv(combined[NEWS_COLUMNS], NEWS_DIR / f"{symbol}_news.csv")
    return len(fresh), len(combined), int(to_score.sum())


def update_news_for_symbol(symbol, company_name, today, logger):  # [#3] [split]
    archive = load_news_archive(NEWS_DIR / f"{symbol}_news.csv")
    name = NEWS_QUERY_OVERRIDES.get(symbol) or company_search_name(company_name, symbol)
    fresh = fetch_feed_items(build_news_url(name), symbol, today, "live", logger)
    fresh = filter_news(fresh, today)
    n_new, n_total, n_scored = merge_score_save(symbol, archive, fresh, today)
    return {"ticker": symbol, "query": name, "status": "ok", "new_headlines": n_new,
            "total_headlines": n_total, "scored": n_scored}


def backfill_news_for_symbol(symbol, company_name, start, end, window_days, today, logger):  # [#3] optional
    archive = load_news_archive(NEWS_DIR / f"{symbol}_news.csv")
    name = NEWS_QUERY_OVERRIDES.get(symbol) or company_search_name(company_name, symbol)
    frames = []
    w_start = start
    while w_start < end:
        w_end = min(w_start + timedelta(days=window_days), end)
        items = fetch_feed_items(build_news_url(name, after=w_start, before=w_end), symbol, today, "backfill", logger)
        if not items.empty:
            d = pd.to_datetime(items["date"])
            frames.append(items[(d >= pd.Timestamp(w_start)) & (d <= pd.Timestamp(w_end))])
        time.sleep(NEWS_REQUEST_PAUSE)
        w_start = w_end
    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NEWS_COLUMNS)
    n_new, n_total, n_scored = merge_score_save(symbol, archive, filter_news(fresh, today), today)
    set_backfill_start(symbol, start)
    return {"ticker": symbol, "query": name, "status": "backfilled", "new_headlines": n_new,
            "total_headlines": n_total, "scored": n_scored}


def _read_coverage():
    if NEWS_COVERAGE_FILE.exists():
        return pd.read_csv(NEWS_COVERAGE_FILE)
    return pd.DataFrame(columns=["ticker", "coverage_start", "first_fetch", "backfill_start",
                                 "n_headlines", "last_headline_date"])


def set_backfill_start(symbol, start):  # [#3]
    cov = _read_coverage()
    cov["backfill_start"] = cov["backfill_start"].astype(object)
    if symbol in set(cov["ticker"]):
        cov.loc[cov["ticker"] == symbol, "backfill_start"] = start.isoformat()
    else:
        cov = pd.concat([cov, pd.DataFrame([{"ticker": symbol, "backfill_start": start.isoformat()}])],
                        ignore_index=True)
    atomic_write_csv(cov, NEWS_COVERAGE_FILE)


def update_news_coverage():  # [#3]
    prev = _read_coverage().set_index("ticker")["backfill_start"].to_dict()
    rows = []
    for path in sorted(NEWS_DIR.glob("*_news.csv")):
        symbol = path.name[: -len("_news.csv")]
        df = load_news_archive(path)
        if df.empty:
            continue
        first_fetch = pd.to_datetime(df["fetched_at"]).min()
        start = first_fetch - pd.Timedelta(days=COVERAGE_LOOKBACK_DAYS)
        bf = prev.get(symbol)
        if bf is not None and pd.notna(bf):
            start = min(start, pd.Timestamp(bf))
        rows.append(
            {
                "ticker": symbol,
                "coverage_start": start.date().isoformat(),
                "first_fetch": first_fetch.date().isoformat(),
                "backfill_start": bf if bf is not None and pd.notna(bf) else "",
                "n_headlines": len(df),
                "last_headline_date": df["date"].max(),
            }
        )
    cov = pd.DataFrame(rows)
    atomic_write_csv(cov, NEWS_COVERAGE_FILE)
    return cov
```

## Changes in Step 4

```python
# =====================================================================
# FILE: scripts/update_prices.py
# Step 4: Incremental price fetch, cleaning and validation
# =====================================================================
OVERLAP_DAYS = 7  # [#6]
PRICE_REQUEST_PAUSE = 1.0
ALLOWED_SERIES = ["EQ", "BE"]  # [Addendum A]
NUM_COLS = [  # [#1]
    "prevclose", "openprice", "highprice", "lowprice", "lastprice",
    "closeprice", "averageprice", "totaltradedquantity", "turnoverinrs",
    "no.oftrades", "deliverableqty", "%dlyqttotradedqty",
]
REQUIRED_PRICE_COLS = ["date", "openprice", "highprice", "lowprice", "closeprice", "totaltradedquantity"]


def fetch_prices(symbol, start, end, logger):  # [#6]
    return with_retry(
        capital_market.price_volume_and_deliverable_position_data,
        symbol=symbol,
        from_date=start.strftime("%d-%m-%Y"),
        to_date=end.strftime("%d-%m-%Y"),
        what=f"prices {symbol}",
        logger=logger,
    )


def clean_prices(df):  # [#1] [#6]
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df = df.drop(columns=[c for c in df.columns if c.startswith("adj_")])
    if "series" in df.columns:
        df = df[df["series"].astype(str).str.strip().str.upper().isin(ALLOWED_SERIES)]
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)


def validate_prices(df):  # [#6]
    missing = [c for c in REQUIRED_PRICE_COLS if c not in df.columns]
    if missing:
        return [("critical", f"missing columns {missing}")]
    problems = []
    if df["date"].duplicated().any():
        problems.append(("critical", "duplicate dates"))
    if not df["date"].is_monotonic_increasing:
        problems.append(("critical", "dates not sorted"))
    if (df["closeprice"] <= 0).any():
        problems.append(("critical", f"{int((df['closeprice'] <= 0).sum())} non-positive closes"))
    n_nan = int(df["closeprice"].isna().sum())
    if n_nan:
        problems.append(("warning", f"{n_nan} missing closes"))
    n_hl = int((df["highprice"] < df["lowprice"]).sum())
    if n_hl:
        problems.append(("warning", f"{n_hl} rows with high < low"))
    outside = (df["closeprice"] > df["highprice"] * 1.0001) | (df["closeprice"] < df["lowprice"] * 0.9999)
    if outside.any():
        problems.append(("warning", f"{int(outside.sum())} closes outside high-low range"))
    return problems


def update_prices_for_symbol(symbol, end, full_refresh, logger):  # [#6] incremental
    path = RAW_DIR / f"{symbol}.csv"
    old = None
    if path.exists() and not full_refresh:
        old = clean_prices(pd.read_csv(path))
        if old.empty:
            old = None

    if old is not None:
        last = old["date"].max().date()
        if last >= end:
            return {"symbol": symbol, "status": "up_to_date", "last_date": last, "new_rows": 0}
        start = max(START_DATE, last - timedelta(days=OVERLAP_DAYS))
    else:
        start = START_DATE

    raw = fetch_prices(symbol, start, end, logger)
    if raw is None or len(raw) == 0:
        status = "no_data" if old is None else "no_new_rows"
        return {"symbol": symbol, "status": status, "start": start, "new_rows": 0}

    new = clean_prices(raw)
    combined = new if old is None else pd.concat([old, new], ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    problems = validate_prices(combined)
    critical = [msg for sev, msg in problems if sev == "critical"]
    if critical:
        return {"symbol": symbol, "status": "invalid", "start": start, "new_rows": 0, "note": "; ".join(critical)}

    atomic_write_csv(combined, path, date_format="%Y-%m-%d")
    return {
        "symbol": symbol,
        "status": "updated",
        "start": start,
        "last_date": combined["date"].max().date(),
        "new_rows": len(combined) - (0 if old is None else len(old)),
        "note": "; ".join(msg for _, msg in problems),
    }


def update_all_prices(plan, full_refresh, logger):  # [#6] [split]
    report = []
    for i, (symbol, end) in enumerate(sorted(plan.items()), start=1):
        try:
            r = update_prices_for_symbol(symbol, end, full_refresh, logger)
        except Exception as e:
            r = {"symbol": symbol, "status": "failed", "new_rows": 0, "note": str(e)[:300]}
        logger.info(f"[{i}/{len(plan)}] {symbol}: {r['status']} (+{r.get('new_rows', 0)} rows) {r.get('note', '')}")
        report.append(r)
        if r["status"] != "up_to_date":
            time.sleep(PRICE_REQUEST_PAUSE)

    rep = pd.DataFrame(report)
    for c in ("start", "last_date", "note"):
        if c not in rep.columns:
            rep[c] = ""
    atomic_write_csv(rep, PRICE_RUN_REPORT)
    failed = rep[rep["status"].isin(["failed", "invalid"])][["symbol", "status", "note"]]
    atomic_write_csv(failed, FAILED_FILE)
    return rep
```

## Changes in Step 4.5

```python
# =====================================================================
# FILE: scripts/update_prices.py
# Step 4.5: Corporate Actions & Price Adjustment
# =====================================================================
ADJUST_DIVIDENDS = False   # False = price-adjusted; True = total-return-adjusted (see issues.md 1.5)

# Manual fixes: (SYMBOL, "YYYY-MM-DD" ex-date) -> (price_factor, share_factor).   [fix]
# split/bonus: both factors equal; demerger/dividend: share_factor = 1.0. Example:
# MANUAL_CA_FACTORS = {("TMPV", "2025-10-14"): (0.6054, 1.0)}
MANUAL_CA_FACTORS = {}

PRICE_COLS = ["openprice", "highprice", "lowprice", "lastprice", "closeprice", "averageprice"]
QTY_COLS = ["totaltradedquantity", "deliverableqty"]
RUPEE = r"(?:rs|re)\.?\s*"
CA_REFETCH_DAYS = 60  # [#6]


def update_corporate_actions(end, full_refresh, logger):  # [#6] incremental, replaces fetch_corporate_actions
    existing = None
    if CA_FILE.exists() and not full_refresh:
        existing = pd.read_csv(CA_FILE)
        existing["exdate"] = pd.to_datetime(existing["exdate"], errors="coerce")
        last_seen = min(existing["exdate"].max().date(), end)
        start = max(START_DATE, last_seen - timedelta(days=CA_REFETCH_DAYS))
    else:
        start = START_DATE

    frames = []
    for y in range(start.year, end.year + 1):
        f = max(start, date(y, 1, 1))
        t = min(end, date(y, 12, 31))
        if f > t:
            continue
        ca = with_retry(
            capital_market.corporate_actions_for_equity,
            from_date=f.strftime("%d-%m-%Y"),
            to_date=t.strftime("%d-%m-%Y"),
            what=f"corporate actions {y}",
            logger=logger,
        )
        if ca is not None and len(ca):
            frames.append(ca)
        time.sleep(1)

    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["symbol", "subject", "exdate"])
    new.columns = new.columns.str.lower().str.strip()
    # On the first run, confirm the column names:
    # print(new.columns.tolist())  -> expect 'symbol', 'subject', 'exdate', ...
    new["exdate"] = pd.to_datetime(new["exdate"], format="mixed", dayfirst=True, errors="coerce")
    parts = [p for p in (existing, new) if p is not None and not p.empty]
    ca = pd.concat(parts, ignore_index=True) if parts else new
    ca["symbol"] = ca["symbol"].astype(str).str.strip().replace(SYMBOL_RENAMES)  # [#5]
    ca = ca.dropna(subset=["exdate"]).drop_duplicates(subset=["symbol", "exdate", "subject"])
    ca = ca.sort_values(["exdate", "symbol"]).reset_index(drop=True)
    atomic_write_csv(ca, CA_FILE, date_format="%Y-%m-%d")
    logger.info(f"Corporate actions: {len(ca)} rows ({len(new)} fetched from {start} to {end})")
    return ca


def parse_actions(subject):
    """Turn NSE 'subject' text into a list of (kind, value)."""
    s = str(subject).lower()
    out = []

    # Split, e.g. "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
    m = re.search(r"from\s*" + RUPEE + r"([\d.]+).*?to\s*" + RUPEE + r"([\d.]+)", s)
    if ("split" in s or "sub-division" in s or "sub division" in s) and m:
        out.append(("split", float(m.group(2)) / float(m.group(1))))

    # Bonus a:b, e.g. "Bonus 1:1"
    for a, b in re.findall(r"bonus\s*(\d+)\s*:\s*(\d+)", s):
        out.append(("bonus", int(b) / (int(a) + int(b))))

    if "demerger" in s or "scheme of arrangement" in s:
        out.append(("demerger", None))

    if "rights" in s:
        out.append(("rights", None))

    # Dividend, e.g. "Interim Dividend - Rs 5.50 Per Share / Special Dividend - Rs 2 Per Share"
    divs = re.findall(r"dividend[^/]*?" + RUPEE + r"([\d.]+)", s)
    if divs:
        out.append(("dividend", sum(float(x) for x in divs)))

    return out


def build_events(symbol, px, ca, logger=None):
    """Return a list of (exdate, price_factor, share_factor, kind) for one symbol."""
    px = px.sort_values("date").reset_index(drop=True)
    events = []
    for _, r in ca[ca["symbol"] == symbol].iterrows():
        ex = r["exdate"]
        before = px[px["date"] < ex]
        on_or_after = px[px["date"] >= ex]
        if before.empty or on_or_after.empty:
            continue
        prev_close = before["closeprice"].iloc[-1]

        for kind, val in parse_actions(r["subject"]):
            if kind in ("split", "bonus"):
                events.append((ex, val, val, kind))
            elif kind == "demerger":
                f = on_or_after["openprice"].iloc[0] / prev_close
                events.append((ex, f, 1.0, kind))
            elif kind == "dividend" and ADJUST_DIVIDENDS and val:
                events.append((ex, 1 - val / prev_close, 1.0, kind))
            elif kind == "rights" and logger:  # [#6]
                logger.warning(f"[review] {symbol} rights issue {ex.date()} not adjusted: {r['subject']}")

    for (sym, d), (f_price, f_share) in MANUAL_CA_FACTORS.items():  # [fix]
        if sym == symbol:
            d = pd.Timestamp(d)
            events = [e for e in events if e[0] != d]
            events.append((d, f_price, f_share, "manual"))
    return events


def apply_adjustments(px, events):
    """Add adj_* columns. Raw NSE columns are left untouched."""
    px = px.sort_values("date").reset_index(drop=True)
    price_cum = np.ones(len(px))
    share_cum = np.ones(len(px))
    for ex, f_price, f_share, _ in events:
        mask = (px["date"] < ex).to_numpy()
        price_cum[mask] *= f_price
        share_cum[mask] *= f_share
    for c in PRICE_COLS:
        px[f"adj_{c}"] = px[c] * price_cum
    for c in QTY_COLS:
        px[f"adj_{c}"] = px[c] / share_cum
    px["adj_factor"] = price_cum
    return px


def adjust_all_prices(ca, logger):  # [#6] logger + atomic writes
    log = []
    files = sorted(RAW_DIR.glob("*.csv"))
    for f in files:
        symbol = f.stem
        px = pd.read_csv(f, parse_dates=["date"])
        px = px.drop(columns=[c for c in px.columns if c.startswith("adj_")])
        events = build_events(symbol, px, ca, logger)
        px = apply_adjustments(px, events)
        atomic_write_csv(px, f, date_format="%Y-%m-%d")
        for ex, fp, fs, kind in events:
            log.append({"symbol": symbol, "exdate": ex.date(), "kind": kind,
                        "price_factor": round(fp, 6), "share_factor": round(fs, 6)})
    atomic_write_csv(pd.DataFrame(log, columns=["symbol", "exdate", "kind", "price_factor", "share_factor"]),
                     ADJ_LOG_FILE)
    logger.info(f"Adjusted {len(files)} files, {len(log)} events applied")
```

## Changes in Step 5

```python
# =====================================================================
# FILE: scripts/update_prices.py
# Step 5 (prices): file header + main()
# =====================================================================
import argparse
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from nselib import capital_market

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import *  # noqa: E402,F401,F403

# >>> paste the "Changes in Step 2" block here
# >>> paste the "Changes in Step 4" block here
# >>> paste the "Changes in Step 4.5" block here
# >>> paste the "Changes in Step 6" block here


def main(argv=None):
    parser = argparse.ArgumentParser(description="Incremental NSE price update (Nifty 100, point-in-time universe)")
    parser.add_argument("--full-refresh", action="store_true", help="re-download all history (raw folder is backed up first)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)

    logger = setup_logging("update_prices")
    ensure_dirs()
    end = date.fromisoformat(args.end_date) if args.end_date else date.today()
    logger.info(f"Price update to {end} | full_refresh={args.full_refresh} | project={PROJECT_ROOT}")

    if args.full_refresh:
        logger.info(f"Backup of raw prices: {backup_folder(RAW_DIR)}")

    membership = update_membership(end, logger)
    plan = fetch_plan(membership, end)
    price_report = update_all_prices(plan, args.full_refresh, logger)

    try:
        ca = update_corporate_actions(end, args.full_refresh, logger)
    except Exception as e:
        if not CA_FILE.exists():
            raise
        logger.error(f"Corporate-action update failed ({e}); using saved {CA_FILE.name}")
        ca = pd.read_csv(CA_FILE, parse_dates=["exdate"])
    adjust_all_prices(ca, logger)

    ok = run_quality_checks(membership, price_report, logger)
    logger.info("PRICE UPDATE " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

```python
# =====================================================================
# FILE: scripts/update_news.py
# Step 5 (news): file header + main()
# =====================================================================
import argparse
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import *  # noqa: E402,F401,F403

# >>> paste the "Changes in Step 1" block here
# >>> paste the "Changes in Step 3" block here

MAX_NEWS_FAILED_SHARE = 0.20


def main(argv=None):
    parser = argparse.ArgumentParser(description="Incremental news + FinBERT sentiment update")
    parser.add_argument("--symbols", nargs="*", default=None, help="limit to these symbols")
    parser.add_argument("--backfill-start", default=None, help="YYYY-MM-DD: one-off historical backfill from this date")
    parser.add_argument("--backfill-window-days", type=int, default=30)
    args = parser.parse_args(argv)

    logger = setup_logging("update_news")
    ensure_dirs()
    today = date.today()

    membership = load_membership()
    if membership is None:
        logger.error("No membership file: run scripts/update_prices.py first")
        return 1
    current = membership[membership["end_date"].isna()].drop_duplicates("symbol")
    if args.symbols:
        current = current[current["symbol"].isin(args.symbols)]
    logger.info(f"News update for {len(current)} symbols | backfill_start={args.backfill_start}")

    report = []
    for i, r in enumerate(current.itertuples(index=False), start=1):
        try:
            if args.backfill_start:
                res = backfill_news_for_symbol(r.symbol, r.company_name, date.fromisoformat(args.backfill_start),
                                               today, args.backfill_window_days, today, logger)
            else:
                res = update_news_for_symbol(r.symbol, r.company_name, today, logger)
        except Exception as e:
            res = {"ticker": r.symbol, "status": "failed", "note": str(e)[:300]}
        logger.info(f"[{i}/{len(current)}] {r.symbol}: {res['status']} "
                    f"+{res.get('new_headlines', 0)} new, {res.get('scored', 0)} scored")
        report.append(res)
        time.sleep(NEWS_REQUEST_PAUSE)

    rep = pd.DataFrame(report)
    atomic_write_csv(rep, NEWS_RUN_REPORT)
    cov = update_news_coverage()
    logger.info(f"News coverage written for {len(cov)} tickers")

    failed_share = (rep["status"] == "failed").mean() if len(rep) else 0.0
    ok = failed_share <= MAX_NEWS_FAILED_SHARE
    logger.info("NEWS UPDATE " + ("PASSED" if ok else f"FAILED ({failed_share:.0%} symbols failed)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

```python
# =====================================================================
# FILE: notebooks/01_domain_setup_data_collection.ipynb
# Step 5 (notebook): replace ALL cells of notebook 01 with these 3 cells
# =====================================================================

# ---- Cell 1
import runpy
from pathlib import Path

PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "requirements.txt").is_file() and (p / "src" / "ml_alpha").is_dir()
)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ---- Cell 2: prices (first run after the upgrade: prices["main"](["--full-refresh"]))
prices = runpy.run_path(str(SCRIPTS_DIR / "update_prices.py"), run_name="notebook")
exit_code = prices["main"]([])
print("Price update exit code:", exit_code)

# ---- Cell 3: news (one-off history backfill: news["main"](["--backfill-start", "2020-01-01"]))
news = runpy.run_path(str(SCRIPTS_DIR / "update_news.py"), run_name="notebook")
exit_code = news["main"]([])
print("News update exit code:", exit_code)
```

```bat
:: =====================================================================
:: FILE: scripts/run_daily_update.bat
:: =====================================================================
@echo off
cd /d "%~dp0.."
python scripts\update_prices.py
if errorlevel 1 exit /b 1
python scripts\update_news.py
exit /b %errorlevel%
```

```bat
:: Windows Task Scheduler: run once in a Command Prompt to register the weekday 19:15 job
schtasks /Create /TN "ml_alpha_daily_update" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 19:15 /TR "\"C:\Users\jayan\Study Materials\ml_alpha_portfolio\ml-alpha-portfolio-model\scripts\run_daily_update.bat\""
```

```yaml
# =====================================================================
# FILE: .github/workflows/data_update.yml   (only if you use GitHub Actions with a self-hosted runner)
# =====================================================================
name: data-update

on:
  schedule:
    - cron: "45 13 * * 1-5"   # 19:15 IST, Monday-Friday
  workflow_dispatch:

jobs:
  prices:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
        with:
          clean: false
      - run: python -m pip install -r requirements.txt
      - run: python scripts/update_prices.py

  news:
    needs: prices
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
        with:
          clean: false
      - run: python scripts/update_news.py
```

## Changes in Step 6

```python
# =====================================================================
# FILE: scripts/update_prices.py
# Step 6: Validate data and decide pass / fail for the pipeline
# =====================================================================
KNOWN_EVENTS = [
    ("BAJFINANCE", "2025-06-16"), ("HDFCBANK", "2025-08-26"), ("RELIANCE", "2024-10-28"),
    ("TMPV", "2025-10-14"), ("SIEMENS", "2025-04-07"), ("EICHERMOT", "2020-08-24"),
    ("NESTLEIND", "2024-01-05"), ("KOTAKBANK", "2026-01-14"), ("VBL", "2024-09-12"),
    ("ITC", "2025-01-06"), ("RELIANCE", "2023-07-20"),
]
KNOWN_EVENT_TOLERANCE = 0.15
LARGE_MOVE = 0.20
MAX_FAILED_SHARE = 0.05  # [#6]
STALE_DAYS = 5  # [#6]
LARGE_MOVES_FILE = DATA_DIR / "large_moves_review.csv"


def check_known_events():
    rows = []
    for sym, d in KNOWN_EVENTS:
        fp = RAW_DIR / f"{sym}.csv"
        if not fp.exists():
            continue
        px = pd.read_csv(fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        idx = px.index[px["date"] == pd.Timestamp(d)]
        if len(idx) == 0 or idx[0] == 0:
            rows.append(("known_event", "warning", f"{sym} {d}: date not found"))
            continue
        i = idx[0]
        adj_r = px.loc[i, "adj_closeprice"] / px.loc[i - 1, "adj_closeprice"] - 1
        status = "ok" if abs(adj_r) < KNOWN_EVENT_TOLERANCE else "critical"
        rows.append(("known_event", status, f"{sym} {d}: adjusted 1-day return {adj_r:+.1%}"))
    return rows


def write_large_moves():
    flags = []
    for fp in sorted(RAW_DIR.glob("*.csv")):
        px = pd.read_csv(fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        r = px["adj_closeprice"].pct_change()
        gap = px["date"].diff().dt.days
        for i in px.index[r.abs() > LARGE_MOVE]:
            flags.append({"symbol": fp.stem, "date": px.loc[i, "date"].date(),
                          "adj_ret": round(r[i], 4), "gap_days": int(gap[i])})
    flags = pd.DataFrame(flags, columns=["symbol", "date", "adj_ret", "gap_days"])
    atomic_write_csv(flags, LARGE_MOVES_FILE)
    return [("large_moves", "warning" if len(flags) else "ok",
             f"{len(flags)} moves > {LARGE_MOVE:.0%} listed in {LARGE_MOVES_FILE.name}")]


def check_freshness(membership):  # [#6]
    current = sorted(membership.loc[membership["end_date"].isna(), "symbol"].unique())
    last_dates = {}
    for sym in current:
        fp = RAW_DIR / f"{sym}.csv"
        if fp.exists():
            last_dates[sym] = pd.read_csv(fp, usecols=["date"], parse_dates=["date"])["date"].max()
    rows = []
    missing = [s for s in current if s not in last_dates]
    rows.append(("missing_files", "critical" if missing else "ok", f"current members without a price file: {missing}"))
    if last_dates:
        newest = max(last_dates.values())
        stale = sorted(s for s, d in last_dates.items() if (newest - d).days > STALE_DAYS)
        rows.append(("stale_symbols", "warning" if stale else "ok",
                     f"newest date {newest.date()}; stale (> {STALE_DAYS} days behind): {stale}"))
    return rows


def check_run(price_report, membership):  # [#5] [#6]
    rows = []
    current = set(membership.loc[membership["end_date"].isna(), "symbol"])
    bad = price_report[price_report["status"].isin(["failed", "invalid"])]
    bad_current = bad[bad["symbol"].isin(current)]
    bad_removed = bad[~bad["symbol"].isin(current)]
    share = len(bad_current) / len(current) if current else 1.0
    rows.append(("failed_current_members",
                 "critical" if share > MAX_FAILED_SHARE else ("warning" if len(bad_current) else "ok"),
                 f"{len(bad_current)}/{len(current)} failed or invalid: {bad_current['symbol'].tolist()}"))
    rows.append(("failed_removed_members", "warning" if len(bad_removed) else "ok",
                 f"{len(bad_removed)} former members without data: {bad_removed['symbol'].tolist()}"))
    n_current = int(membership["end_date"].isna().sum())
    lo, hi = EXPECTED_INDEX_SIZE
    rows.append(("index_size", "ok" if lo <= n_current <= hi else "critical", f"{n_current} current members"))
    seeded = (membership["source"] == "seed_current_list").all()
    rows.append(("membership_history", "warning" if seeded else "ok",
                 "membership seeded from today's list only (survivorship bias)" if seeded
                 else "membership history available"))
    return rows


def run_quality_checks(membership, price_report, logger):  # [#6]
    rows = check_known_events() + write_large_moves() + check_freshness(membership) + check_run(price_report, membership)
    report = pd.DataFrame(rows, columns=["check", "status", "detail"])
    report.insert(0, "run_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    atomic_write_csv(report, QUALITY_REPORT)
    for _, r in report.iterrows():
        log = logger.error if r["status"] == "critical" else (logger.warning if r["status"] == "warning" else logger.info)
        log(f"[{r['check']}] {r['status'].upper()}: {r['detail']}")
    return not (report["status"] == "critical").any()
```

## Changes outside notebook 01

### requirements.txt

```text
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4
matplotlib==3.8.2
scikit-learn==1.3.2
nselib
requests
feedparser
transformers>=4.30
torch
tqdm
lightgbm
seaborn
plotly
sentence-transformers
openai
python-dotenv
nb_black
```

### Notebook 02

```python
# Original line numbers. Edit each cell from the bottom up.

# ---- NB02 · Step 1: Setup paths & logging · insert after line 8
NEWS_COVERAGE_FILE = PROJECT_ROOT / "data" / "news_coverage.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- NB02 · Step 1: Setup paths & logging · replace line 5
PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "requirements.txt").is_file() and (p / "src" / "ml_alpha").is_dir()
)

# ---- NB02 · Step 3: Process Each Ticker · insert after line 118  (8-space indent, inside the for-loop)
        cov_start = news_coverage.get(symbol, pd.NaT)
        df["has_news_coverage"] = (df["Date"] >= cov_start).astype(int) if pd.notna(cov_start) else 0
        df.loc[df["has_news_coverage"] == 0, final_news_features] = np.nan

# ---- NB02 · Step 3: Process Each Ticker · insert before line 26  (no indent, before the for-loop)
news_coverage = (
    pd.read_csv(NEWS_COVERAGE_FILE, parse_dates=["coverage_start"]).set_index("ticker")["coverage_start"]
    if NEWS_COVERAGE_FILE.exists()
    else pd.Series(dtype="datetime64[ns]")
)

# ---- NB02 · Step 6: Filter Stocks With Minimum 1000 Rows · replace line 5
MIN_DAYS = 63

# ---- NB02 · cell starting 'feature_files = list(FEATURE_DIR.glob("*.csv"))' + 'summary = []' · replace line 18
summary_df.to_csv(RESULTS_DIR / "phase2_metadata.csv", index=False)
```

### Notebook 03

```python
# Original line numbers. Edit each cell from the bottom up.

# ---- NB03 · CONFIGURATION (FIXED PATH HANDLING) · replace line 8
PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "requirements.txt").is_file() and (p / "src" / "ml_alpha").is_dir()
)
MEMBERSHIP_FILE = PROJECT_ROOT / "data" / "nifty100_membership.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- NB03 · STEP 0: LOAD PANEL · insert after line 25
membership = pd.read_csv(MEMBERSHIP_FILE, parse_dates=["start_date", "end_date"])
panel["in_index"] = 0
for m in membership.itertuples(index=False):
    end = m.end_date if pd.notna(m.end_date) else pd.Timestamp.max
    mask = (panel["ticker"] == m.symbol) & (panel["date"] >= m.start_date) & (panel["date"] < end)
    panel.loc[mask, "in_index"] = 1
print("Rows inside the index on their date:", int(panel["in_index"].sum()), "of", len(panel))

# ---- NB03 · STEP 1: Correlation heatmap ... · replace line 14
exclude_cols = ["date", "ticker", "future_return_21d", "symbol", "series", "in_index"]

# ---- NB03 · STEP 1: Correlation heatmap ... · insert after line 11
panel = panel[panel["in_index"] == 1].reset_index(drop=True)

# ---- NB03 · STEP 2: FEATURE LIST · replace lines 7-14
USE_NEWS_FEATURES = False

TECH_FEATURES = [
    "momentum_1m", "momentum_3m", "momentum_6m",
    "vol_30d", "vol_90d", "volume_zscore", "volume_momentum", "volume_change",
    "rsi_14", "macd_hist", "ema_cross",
    "mom_vol_interaction", "high_vol_regime", "momentum_1m_lag5",
    "ret_skew_10", "ret_kurt_10", "ret_skew_30", "ret_kurt_30",
]
NEWS_FEATURES = [
    "news_sent", "news_count", "sent_3d", "sent_7d", "sent_momentum", "sent_surprise", "news_intensity",
    "news_event", "days_since_news", "news_decay_weight", "sent_weighted", "surprise_weighted",
    "is_recent_news", "sent_mom_interaction", "has_news_coverage",
]
FEATURES = TECH_FEATURES + (NEWS_FEATURES if USE_NEWS_FEATURES else [])

# ---- NB03 · SAVE RESULTS (cell after STEP 11: FINAL RANKING) · replace line 5
top_stocks.to_csv(RESULTS_DIR / f"phase3_top10_next_month_{latest_date:%Y%m%d}.csv", index=False)
```

### Notebook 04

```python
# ---- NB04 · STEP 0: Setup Path · replace line 12
PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "requirements.txt").is_file() and (p / "src" / "ml_alpha").is_dir()
)

# ---- NB04 · STEP 4: PORTFOLIO CONSTRUCTION · replace line 108
portfolio_log_df.to_csv(PROJECT_ROOT / "results" / "portfolio_decisions.csv", index=False)
```

### Notebook 05

```python
# ---- NB05 · cell with 'PROJECT_ROOT = Path().resolve().parent' (paths) · replace line 6
PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents]
    if (p / "requirements.txt").is_file() and (p / "src" / "ml_alpha").is_dir()
)

# ---- NB05 · cell starting 'results = [{' · replace line 7
explain_df.to_csv(PROJECT_ROOT / "results" / "latest_portfolio_explanations.csv", index=False)
```

### .gitignore

```text
data/
data_backup/
logs/
notebooks/data/
*.csv.tmp
.env
```
