# =====================================================================
# Step 0: Project Setup (shared by update_prices.py and update_news.py)
# =====================================================================
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
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
RAW_DIR = DATA_DIR / "raw"                    # per-stock price files (DERIVED from bhavcopies, rebuilt every run)
NEWS_DIR = DATA_DIR / "news_raw"
INTERIM_DIR = DATA_DIR / "interim"            # caches + audit tables, rebuildable from data/source
REFERENCE_DIR = DATA_DIR / "reference"        # hand-maintained tables (sector map, official membership)
BACKUP_DIR = PROJECT_ROOT / "data_backup"  # [#4]
LOG_DIR = PROJECT_ROOT / "logs"  # [#6]

MEMBERSHIP_FILE = DATA_DIR / "nifty100_membership.csv"  # [#5] point-in-time, read by Phase 3
MEMBERSHIP_BACKFILL_FILE = DATA_DIR / "nifty100_membership_backfill.csv"  # [#5] legacy, no longer written
OFFICIAL_MEMBERSHIP_FILE = REFERENCE_DIR / "nifty100_membership_official.csv"  # optional, hand-curated
REBALANCE_FILE = INTERIM_DIR / "universe_rebalances.csv"
CA_FILE = DATA_DIR / "corporate_actions.csv"  # [#1] combined copy of the per-year NSE files (for reading)
ADJ_LOG_FILE = DATA_DIR / "adjustment_log.csv"  # [#1]
NEWS_COVERAGE_FILE = DATA_DIR / "news_coverage.csv"  # [#3]
PRICE_RUN_REPORT = DATA_DIR / "price_update_report.csv"  # [#6]
NEWS_RUN_REPORT = DATA_DIR / "news_update_report.csv"  # [#6]
QUALITY_REPORT = DATA_DIR / "data_quality_report.csv"  # [#6]
FAILED_FILE = DATA_DIR / "failed_tickers.csv"
MANIFEST_FILE = DATA_DIR / "run_manifest.json"


# ---------------------------------------------------------------------
# Run settings: defaults here, overridable by environment variables (CI/CD)
# ---------------------------------------------------------------------
def _env_str(name, default):
    v = os.environ.get(name, "").strip()
    return v if v else default


def _env_date(name, default):
    v = os.environ.get(name, "").strip()
    return date.fromisoformat(v) if v else default


@dataclass(frozen=True)
class PipelineSettings:
    price_start: date            # first bhavcopy day used (needs ~1 year before universe_start for warm-up)
    universe_start: date         # first day the point-in-time universe is defined
    universe_size: int           # 100 = Nifty-100-equivalent
    exit_rank: int               # existing members stay while ranked <= exit_rank (index-style buffer)
    candidate_rank: int          # ranked names considered per file (size + spare for non-traded names)
    ranking_lag_days: int        # ranking as-of date -> effective date (mimics index reconstitution delay)
    min_trading_share: float     # share of sessions traded in the 3 months before the ranking date
    universe_mode: str           # "mcap_rank" (rule-based, default) or "official" (hand-curated file)
    adjust_dividends: bool       # True = total-return adjusted prices (matches TRI benchmarks)
    source_dir: Path             # NSE downloads (never deleted by code); must be inside the project
    extra_source_dirs: tuple     # read-only folders with the same layout (e.g. the Fama-French project's data/raw)

    @classmethod
    def from_env(cls):
        source_dir = Path(_env_str("ML_ALPHA_SOURCE_DIR", str(DATA_DIR / "source"))).resolve()
        if source_dir != PROJECT_ROOT and PROJECT_ROOT not in source_dir.parents:
            raise RuntimeError(f"ML_ALPHA_SOURCE_DIR must be inside {PROJECT_ROOT}; "
                               "put other folders in ML_ALPHA_EXTRA_SOURCE_DIRS (read-only)")
        extra = tuple(Path(p).resolve() for p in _env_str("ML_ALPHA_EXTRA_SOURCE_DIRS", "").split(os.pathsep) if p)
        s = cls(
            price_start=_env_date("ML_ALPHA_PRICE_START", date(2015, 1, 1)),
            universe_start=_env_date("ML_ALPHA_UNIVERSE_START", date(2016, 5, 2)),
            universe_size=int(_env_str("ML_ALPHA_UNIVERSE_SIZE", "100")),
            exit_rank=int(_env_str("ML_ALPHA_EXIT_RANK", "110")),
            candidate_rank=int(_env_str("ML_ALPHA_CANDIDATE_RANK", "250")),
            ranking_lag_days=int(_env_str("ML_ALPHA_RANKING_LAG_DAYS", "30")),
            min_trading_share=float(_env_str("ML_ALPHA_MIN_TRADING_SHARE", "0.9")),
            universe_mode=_env_str("ML_ALPHA_UNIVERSE_MODE", "mcap_rank"),
            adjust_dividends=_env_str("ML_ALPHA_ADJUST_DIVIDENDS", "1") == "1",
            source_dir=source_dir,
            extra_source_dirs=extra,
        )
        if s.universe_mode not in {"mcap_rank", "official"}:
            raise ValueError(f"ML_ALPHA_UNIVERSE_MODE must be mcap_rank or official, got {s.universe_mode}")
        if s.exit_rank < s.universe_size or s.candidate_rank < s.exit_rank:
            raise ValueError("need universe_size <= exit_rank <= candidate_rank")
        if (s.universe_start - s.price_start).days < 300:
            raise ValueError("price_start must be ~1 year before universe_start (trading filter + feature warm-up)")
        return s

    def source_dirs(self, sub):
        """Writable folder first, then read-only extra folders, for one sub-folder (e.g. 'bhavcopy')."""
        return [self.source_dir / sub] + [d / sub for d in self.extra_source_dirs]

    def to_dict(self):
        return {k: (str(v) if isinstance(v, (Path, date)) else [str(x) for x in v] if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}


SETTINGS = PipelineSettings.from_env()
START_DATE = SETTINGS.price_start  # kept for backward compatibility
SOURCE_DIR = SETTINGS.source_dir

MEMBERSHIP_COLUMNS = ["symbol", "company_name", "industry", "start_date", "end_date", "source",
                      "entry_rank", "ranking_as_of", "exit_reason"]  # [#5]

# Undated fallback only: used when NSE's symbolchange.csv does not list the old symbol.
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
    for d in (DATA_DIR, RAW_DIR, NEWS_DIR, INTERIM_DIR, REFERENCE_DIR, BACKUP_DIR, LOG_DIR,
              SOURCE_DIR / "bhavcopy", SOURCE_DIR / "corporate_actions", SOURCE_DIR / "universe" / "nse_mcap"):
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


def atomic_write_bytes(data, path):
    path = assert_inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(obj, path):
    atomic_write_bytes(json.dumps(obj, indent=2, default=str).encode("utf-8"), path)


def load_membership():  # [#5]
    if not MEMBERSHIP_FILE.exists():
        return None
    m = pd.read_csv(MEMBERSHIP_FILE)
    for c in MEMBERSHIP_COLUMNS:
        if c not in m.columns:
            m[c] = pd.NA
    m["symbol"] = m["symbol"].astype(str).str.strip()
    m["start_date"] = pd.to_datetime(m["start_date"])
    m["end_date"] = pd.to_datetime(m["end_date"])
    m["source"] = m["source"].fillna("unknown")
    return m[MEMBERSHIP_COLUMNS]