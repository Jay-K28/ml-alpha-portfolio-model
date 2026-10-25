# =====================================================================
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

# =====================================================================
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

# =====================================================================
# Step 4.5: Corporate Actions & Price Adjustment
# =====================================================================
ADJUST_DIVIDENDS = False   # False = price-adjusted; True = total-return-adjusted (see issues.md 1.5)

# fixes: (SYMBOL, "YYYY-MM-DD" ex-date) -> (price_factor, share_factor).   [fix]
# split/bonus: both factors equal; demerger/dividend: share_factor = 1.0. 
MANUAL_CA_FACTORS = {("TVSMOTOR", "2025-08-25"): (1.0, 1.0)}

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
    
# =====================================================================
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

# =====================================================================
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


if __name__ == "__main__":
    sys.exit(main())
