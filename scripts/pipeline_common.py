# =====================================================================
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
