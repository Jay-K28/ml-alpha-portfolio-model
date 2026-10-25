# =====================================================================
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