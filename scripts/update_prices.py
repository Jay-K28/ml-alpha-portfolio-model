# =====================================================================
# FILE: scripts/update_prices.py
# Phase 1 prices: point-in-time Nifty-100-equivalent universe, NSE bhavcopy prices,
# corporate actions (total-return adjusted), merger continuation, quality gate.
# =====================================================================
import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import *  # noqa: E402,F401,F403

# =====================================================================
# Step 1: NSE downloads (one place, so CI can switch them off with --offline)
# =====================================================================
class NSEDownloader:
    """Downloads one NSE archive file through nselib's browser-like session. Never raises."""

    MIN_BYTES = 1000

    def __init__(self, logger, enabled=True, pause=0.3):
        self.logger, self.enabled, self.pause = logger, enabled, pause
        self._fetch, self.last_status = None, None
        if enabled:
            try:
                from nselib.libutil import nse_urlfetch
                self._fetch = nse_urlfetch
            except ImportError:
                logger.warning("nselib not installed: nothing will be downloaded (cached files are used)")
                self.enabled = False

    def fetch(self, url, dest):
        """True = saved. False = failed. self.last_status keeps the HTTP code (404 = file does not exist)."""
        self.last_status = None
        if not self.enabled:
            return False
        try:
            r = self._fetch(url)
            self.last_status = r.status_code
            if r.status_code == 200 and len(r.content) > self.MIN_BYTES:
                atomic_write_bytes(r.content, dest)
                time.sleep(self.pause)
                return True
        except Exception as e:
            self.logger.debug(f"download failed {Path(dest).name}: {type(e).__name__} {e}")
        return False


def first_existing(dirs, name):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def clean_cols(cols):
    """Fix BOM/quote garbage in NSE headers and lower-case them."""
    return (pd.Index(cols).astype(str)
            .str.replace("﻿", "", regex=False).str.replace("ï»¿", "", regex=False)
            .str.replace('"', "", regex=False).str.strip().str.lower())


def parse_dates(s):
    return pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")

# =====================================================================
# Step 2: Symbol changes -> permanent stock ID (latest symbol), date-aware
# =====================================================================
class SymbolResolver:
    """resolve(symbol, date) follows NSE renames that happened AFTER `date` and returns the latest symbol.
    Date-aware, so a symbol later reused by another company is not mixed up."""

    URL = "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"
    # Renames missing from NSE's list (effective date = first trading day under the new symbol)
    MANUAL_RENAMES = [
        ("PHILIPCARB", "PCBL", "2022-01-13"),
        ("BURGERKING", "RBA", "2022-02-17"),
        ("WABCOINDIA", "ZFCVINDIA", "2022-04-01"),
        ("RUCHI", "PATANJALI", "2022-07-13"),
    ]

    def __init__(self, changes, logger=None):
        self.changes = changes            # old -> sorted [(effective date or NaT for undated, new)]

    @classmethod
    def load(cls, settings, downloader, end_date, logger):
        dirs = settings.source_dirs("universe")
        path = first_existing(dirs, "symbolchange.csv")
        own = dirs[0] / "symbolchange.csv"
        if path is None or date.fromtimestamp(path.stat().st_mtime) < end_date - timedelta(days=7):
            if downloader.fetch(cls.URL, own):
                path = own
        changes = {}
        if path is not None:
            sc = pd.read_csv(path, encoding="latin-1")
            sc.columns = clean_cols(sc.columns)
            names = {"old symbol": "old", "new symbol": "new", "effective date": "eff_date"}
            if not set(names) <= set(sc.columns):          # header changed: fall back to position
                names = {sc.columns[1]: "old", sc.columns[2]: "new", sc.columns[3]: "eff_date"}
            sc = sc.rename(columns=names)
            sc["old"] = sc["old"].astype(str).str.strip().str.upper()
            sc["new"] = sc["new"].astype(str).str.strip().str.upper()
            sc["eff_date"] = parse_dates(sc["eff_date"])
            if sc["eff_date"].notna().mean() < 0.95:
                raise ValueError(f"symbolchange.csv dates not parsed - check format in {path}")
            for old, new, eff in zip(sc["old"], sc["new"], sc["eff_date"]):
                if pd.notna(eff):
                    changes.setdefault(old, []).append((eff, new))
            logger.info(f"Symbol changes: {len(sc)} from {path}")
        else:
            logger.warning("symbolchange.csv not available: only manual renames are used")
        for old, new, eff in cls.MANUAL_RENAMES:
            changes.setdefault(old, []).append((pd.Timestamp(eff), new))
        for old, new in SYMBOL_RENAMES.items():          # undated fallback
            if old not in changes:
                changes[old] = [(pd.NaT, new)]
                logger.info(f"Undated rename used (not in NSE list): {old} -> {new}")
        for old in changes:
            changes[old] = sorted(changes[old], key=lambda x: (pd.isna(x[0]), x[0] if pd.notna(x[0]) else 0))
        return cls(changes)

    def resolve(self, symbol, when):
        sym, t = str(symbol).strip().upper(), pd.Timestamp(when)
        for _ in range(10):
            nxt = [(e, n) for e, n in self.changes.get(sym, []) if pd.isna(e) or e > t]
            if not nxt:
                return sym
            e, sym = nxt[0]
            if pd.notna(e):
                t = e
        return sym

    def resolve_frame(self, df, sym_col, date_col):
        out = df[sym_col].astype(str).str.strip().str.upper()
        hit = out.isin(self.changes.keys())
        if hit.any():
            pairs = df.loc[hit, [sym_col, date_col]].drop_duplicates()
            lookup = {(s, t): self.resolve(s, t) for s, t in zip(pairs[sym_col], pairs[date_col])}
            out.loc[hit] = [lookup[(s, t)] for s, t in zip(df.loc[hit, sym_col], df.loc[hit, date_col])]
        return out

    def final(self, sym):
        """Last symbol of a rename chain, ignoring dates (pre-filter only)."""
        for _ in range(10):
            if sym not in self.changes:
                return sym
            sym = self.changes[sym][-1][1]
        return sym

    def aliases(self, sids):
        """Every symbol that may have been used by these stocks (superset, for filtering bhavcopies)."""
        sids = set(sids)
        return sids | {old for old in self.changes if self.final(old) in sids}

    
# =====================================================================
# Step 3: NSE market-cap ranking files (universe candidates)
# =====================================================================
NSE_FILES = "https://nsearchives.nseindia.com/web/sites/default/files/inline-files/"
# (as-of date, type, URL). Add a row whenever NSE publishes a new file: a data change, not a code change.
# type: "top" = ranked list without market cap, "point" = market cap on the date,
#       "average" = average market cap over the 6 months ending on the date
MCAP_RANKINGS = [
    ("2016-03-31", "top", NSE_FILES + "Top500_BRR_3103016_2.xls"),
    ("2017-03-31", "top", NSE_FILES + "TOP500_BRR_31032017_2.xlsx"),
    ("2018-03-31", "top", NSE_FILES + "TOP500_BRR_31032018_2.xlsx"),
    ("2019-03-31", "top", NSE_FILES + "Top1000_31032019_1.xlsx"),
    ("2019-12-31", "point", NSE_FILES + "MCAP31122019_0.xlsx"),
    ("2020-03-31", "point", NSE_FILES + "MCAP_31032020_2.xlsx"),
    ("2020-12-31", "point", NSE_FILES + "MCAP31122020_1.xlsx"),
    ("2021-03-31", "point", NSE_FILES + "MCAP31032021_2.xlsx"),
    ("2021-12-31", "point", NSE_FILES + "MCAP31122021_1.xlsx"),
    ("2022-03-31", "point", NSE_FILES + "MCAP31032022_1.xlsx"),
    ("2022-12-31", "point", NSE_FILES + "MCAP31122022_3.xlsx"),
    ("2023-03-31", "point", NSE_FILES + "MCAP31032023_0.xlsx"),
    ("2023-12-31", "point", NSE_FILES + "MCAP31122023_0.xlsx"),
    ("2024-03-28", "point", NSE_FILES + "MCAP28032024.xlsx"),
    ("2024-12-31", "average", NSE_FILES + "Average%20MCAP_July2024ToDecember%202024%20%281%29.xlsx"),
    ("2025-12-31", "average", "https://nsearchives.nseindia.com//web/mediaattachment/2026-01/"
                              "Average_MCAP_July2025ToDecember2025_20260102201101.xlsx"),
    # TODO (data): add the January-June average files for 2024, 2025 and 2026 from NSE's page
    # "Market capitalisation of all listed companies" -> semi-annual universe updates instead of annual.
]


class RankingStore:
    def __init__(self, settings, downloader, logger, rankings=MCAP_RANKINGS):
        self.dirs = settings.source_dirs("universe/nse_mcap")
        self.downloader, self.logger, self.rankings = downloader, logger, rankings

    @staticmethod
    def file_name(url):
        return url.split("/")[-1].replace("%20", " ").replace("%28", "(").replace("%29", ")")

    @staticmethod
    def read_ranking(path):
        """One NSE ranking file -> DataFrame(rank, symbol, mcap_cr). Finds the header row itself."""
        raw = pd.read_excel(path, header=None, sheet_name=0)
        header_row = next(i for i in range(min(len(raw), 20))
                          if raw.iloc[i].astype(str).str.strip().str.lower().eq("symbol").any())
        df = raw.iloc[header_row + 1:].copy()
        df.columns = clean_cols(raw.iloc[header_row])
        rank_col = df.columns[0]
        mcap_cols = [c for c in df.columns if "capitali" in c or "mcap" in c]
        out = pd.DataFrame({"rank": pd.to_numeric(df[rank_col], errors="coerce"),
                            "symbol": df["symbol"].astype(str).str.strip().str.upper()})
        if mcap_cols:
            raw_mcap = pd.to_numeric(df[mcap_cols[0]], errors="coerce")
            top = raw_mcap.max()
            # units vary (lakh vs rupees): the largest Indian company has been Rs 50k crore - 1 crore crore
            scale = next((s for s in [1.0, 0.01, 1e-4, 1e-7] if 5e4 <= top * s <= 1e7),
                         0.01 if "lakh" in mcap_cols[0] else 1.0)
            out["mcap_cr"] = raw_mcap * scale
        else:
            out["mcap_cr"] = np.nan
        return out.dropna(subset=["rank"]).query("symbol != 'NAN'").reset_index(drop=True)

    def load(self, resolver):
        frames = []
        for as_of, kind, url in self.rankings:
            name = self.file_name(url)
            path = first_existing(self.dirs, name)
            if path is None and self.downloader.fetch(url, self.dirs[0] / name):
                path = self.dirs[0] / name
            if path is None:
                self.logger.warning(f"Ranking file MISSING: {name} -> download it in a browser into {self.dirs[0]}")
                continue
            r = self.read_ranking(path)
            r["as_of"], r["kind"], r["file"] = pd.Timestamp(as_of), kind, name
            r["sid"] = resolver.resolve_frame(r, "symbol", "as_of")
            frames.append(r)
        if not frames:
            raise FileNotFoundError(f"No NSE market-cap ranking files in {self.dirs}")
        rk = pd.concat(frames, ignore_index=True)
        self.logger.info(f"Ranking files: {rk['as_of'].nunique()} "
                         f"({rk['as_of'].min().date()} -> {rk['as_of'].max().date()})")
        return rk

    
# =====================================================================
# Step 4: NSE daily bhavcopies (whole market, incl. later-delisted stocks)
# =====================================================================
class BhavcopyStore:
    UDIFF_START = date(2024, 7, 8)          # NSE changed the bhavcopy format on this date
    KEEP_SERIES = ("EQ", "BE", "BZ")        # BE/BZ = trade-to-trade (often distressed): must be kept
    OLD_MAP = {"SYMBOL": "symbol", "SERIES": "series", "ISIN": "isin", "OPEN": "open", "HIGH": "high",
               "LOW": "low", "CLOSE": "close", "LAST": "last", "PREVCLOSE": "prevclose",
               "TOTTRDQTY": "volume", "TOTTRDVAL": "turnover", "TOTALTRADES": "trades"}
    NEW_MAP = {"TckrSymb": "symbol", "SctySrs": "series", "ISIN": "isin", "OpnPric": "open", "HghPric": "high",
               "LwPric": "low", "ClsPric": "close", "LastPric": "last", "PrvsClsgPric": "prevclose",
               "TtlTradgVol": "volume", "TtlTrfVal": "turnover", "TtlNbOfTxsExctd": "trades",
               "FinInstrmNm": "name"}
    NUM = ["open", "high", "low", "close", "last", "prevclose", "volume", "turnover", "trades"]
    NO_FILE_LOG = "no_file_days.txt"        # weekdays confirmed without a file (holidays) -> not retried

    def __init__(self, settings, downloader, logger):
        self.dirs = settings.source_dirs("bhavcopy")
        self.downloader, self.logger = downloader, logger
        self.cache = INTERIM_DIR / "bhav_ohlcv.pkl"

    @classmethod
    def url(cls, d):
        if d < cls.UDIFF_START:
            mon = d.strftime("%b").upper()
            return (f"https://nsearchives.nseindia.com/content/historical/EQUITIES/"
                    f"{d.year}/{mon}/cm{d:%d}{mon}{d.year}bhav.csv.zip")
        return f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"

    def files(self, start, end):
        """{date: path} for every day with a file in any source folder (own folder wins)."""
        out = {}
        for d in reversed(self.dirs):
            for p in Path(d).glob("????-??-??.csv.zip"):
                day = date.fromisoformat(p.name[:10])
                if start <= day <= end:
                    out[day] = p
        return dict(sorted(out.items()))

    def download_missing(self, start, end):
        have = self.files(start, end)
        log = self.dirs[0] / self.NO_FILE_LOG
        known_none = set(log.read_text().split()) if log.exists() else set()
        todo = [d.date() for d in pd.bdate_range(start, end)
                if d.date() not in have and d.strftime("%Y-%m-%d") not in known_none]
        got, none = 0, []
        for d in todo:
            if self.downloader.fetch(self.url(d), self.dirs[0] / f"{d:%Y-%m-%d}.csv.zip"):
                got += 1
            elif self.downloader.last_status == 404 and (date.today() - d).days > 5:
                none.append(f"{d:%Y-%m-%d}")       # NSE says no file (holiday): do not retry
            # any other failure (blocked, timeout) is retried on the next run, never marked as a holiday
        if none:
            atomic_write_bytes(("\n".join(sorted(known_none | set(none))) + "\n").encode(), log)
        self.logger.info(f"Bhavcopies: {len(have) + got} files | downloaded now: {got} | "
                         f"weekdays without file (holidays): {len(known_none) + len(none)}")

    @classmethod
    def read(cls, path, symbols=None):
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()
        df = df.rename(columns=cls.NEW_MAP if "TckrSymb" in df.columns else cls.OLD_MAP)
        df = df[[c for c in ["symbol", "series", "isin", "name", *cls.NUM] if c in df.columns]].copy()
        df["series"] = df["series"].astype(str).str.strip().str.upper()
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df = df[df["series"].isin(cls.KEEP_SERIES)]
        if symbols is not None:
            df = df[df["symbol"].isin(symbols)]
        for c in cls.NUM:
            df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False), errors="coerce") if c in df else np.nan
        if "name" not in df:
            df["name"] = np.nan
        df["date"] = pd.Timestamp(Path(path).name[:10])
        df["date"] = df["date"].astype("datetime64[ns]")      # one time unit everywhere (pandas 2.x)
        return df

    def load(self, symbols, start, end, rebuild=False):
        """OHLCV rows for `symbols` (all their old names included). Cached; re-reads files only when needed."""
        files = self.files(start, end)
        symbols = set(symbols)
        bhav, cached_syms, cached_days = pd.DataFrame(), set(), set()
        if self.cache.exists() and not rebuild:
            bhav = pd.read_pickle(self.cache)
            cached_syms = set(bhav.attrs.get("symbols", []))
            cached_days = set(bhav.attrs.get("days", []))
        if not symbols <= cached_syms:
            self.logger.info(f"Bhavcopy cache: {len(symbols - cached_syms)} new symbols -> re-reading all files")
            bhav, cached_days = pd.DataFrame(), set()
        all_syms = symbols | cached_syms
        new_days = [d for d in files if d.isoformat() not in cached_days]
        if new_days:
            parts = [self.read(files[d], all_syms) for d in new_days]
            bhav = pd.concat([bhav, *parts], ignore_index=True)
            for c in ["symbol", "series", "isin", "name"]:
                bhav[c] = bhav[c].astype("category")
            bhav.attrs["symbols"] = sorted(all_syms)
            bhav.attrs["days"] = sorted(cached_days | {d.isoformat() for d in new_days})
            bhav.to_pickle(assert_inside_project(self.cache))
            self.logger.info(f"Bhavcopy cache: read {len(new_days)} files, {len(bhav):,} rows")
        sessions = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in files)).astype("datetime64[ns]")
        bhav = bhav[(bhav["date"] >= pd.Timestamp(start)) & (bhav["date"] <= pd.Timestamp(end))].copy()
        bhav["date"] = bhav["date"].astype("datetime64[ns]")
        return bhav, sessions

    
# =====================================================================
# Step 5: Point-in-time universe (top-N by market cap, index-style buffer)
# =====================================================================
class UniverseBuilder:
    """Rule: at each rebalance (first session >= ranking as-of date + lag) the universe is the top
    `universe_size` stocks of that ranking file that traded on >= min_trading_share of sessions in the
    3 months up to the as-of date and traded in the 5 sessions before the rebalance. Existing members
    stay while ranked <= exit_rank. Uses only information available on the rebalance date."""

    def __init__(self, settings, logger):
        self.s, self.logger = settings, logger

    def schedule(self, rankings, sessions, end):
        files = rankings[["as_of", "file"]].drop_duplicates("as_of").sort_values("as_of")
        rows = []
        for as_of, f in zip(files["as_of"], files["file"]):
            target = as_of + pd.Timedelta(days=self.s.ranking_lag_days)
            i = sessions.searchsorted(target)
            if i < len(sessions):
                rows.append({"effective": sessions[i], "as_of": as_of, "file": f})
        sch = pd.DataFrame(rows)
        start = sessions[sessions.searchsorted(pd.Timestamp(self.s.universe_start))]
        before = sch[sch["effective"] <= start]
        if before.empty:
            raise ValueError(f"No ranking file effective on/before universe_start {self.s.universe_start}")
        first = before.iloc[[-1]].assign(effective=start)
        sch = pd.concat([first, sch[sch["effective"] > start]], ignore_index=True)
        return sch[sch["effective"] <= pd.Timestamp(end)].reset_index(drop=True)

    def build(self, rankings, presence, sessions, last_trade, names, end):
        """presence: DataFrame(sid, date) unique rows of days each stock traded."""
        sch = self.schedule(rankings, sessions, end)
        traded = {sid: set(g) for sid, g in presence.groupby("sid")["date"]}
        prev, rows, reb = set(), [], []
        for eff, as_of, f in zip(sch["effective"], sch["as_of"], sch["file"]):
            win = set(sessions[(sessions > as_of - pd.DateOffset(months=3)) & (sessions <= as_of)])
            recent = set(sessions[sessions < eff][-5:])
            cand = (rankings[(rankings["as_of"] == as_of) & (rankings["rank"] <= self.s.candidate_rank)]
                    .sort_values("rank").drop_duplicates("sid"))

            def eligible(sid):
                d = traded.get(sid)
                if not d or not win:
                    return False
                return len(d & win) / len(win) >= self.s.min_trading_share and bool(d & recent)

            cand = cand[[eligible(s) for s in cand["sid"]]]
            cand = cand.reset_index(drop=True)
            cand["elig_rank"] = np.arange(1, len(cand) + 1)       # rank among eligible names
            keep = cand[cand["sid"].isin(prev) & (cand["elig_rank"] <= self.s.exit_rank)]
            keep = keep.head(self.s.universe_size)
            fill = cand[~cand["sid"].isin(keep["sid"])].head(self.s.universe_size - len(keep))
            members = pd.concat([keep, fill]).sort_values("elig_rank")
            if len(members) < self.s.universe_size:
                self.logger.warning(f"{eff.date()}: only {len(members)} eligible members")
            cur = set(members["sid"])
            reb.append({"effective": eff, "ranking_as_of": as_of, "ranking_file": f, "members": len(cur),
                        "added": sorted(cur - prev), "removed": sorted(prev - cur)})
            rows.append(members.assign(effective=eff, ranking_as_of=as_of)[
                ["effective", "sid", "elig_rank", "ranking_as_of"]])
            prev = cur
        snap = pd.concat(rows, ignore_index=True)
        return self._intervals(snap, sch, last_trade, names), pd.DataFrame(reb)

    def _intervals(self, snap, sch, last_trade, names):
        dates = list(sch["effective"])
        out = []
        for sid, g in snap.groupby("sid"):
            g = g.set_index("effective")
            start = None
            for d in dates:
                if d in g.index and start is None:
                    start = d                                    # joins at this rebalance
                elif d not in g.index and start is not None:     # left at this rebalance (end is exclusive)
                    out.append((sid, start, d, g.loc[start]))
                    start = None
            if start is not None:                                # still a member
                out.append((sid, start, pd.NaT, g.loc[start]))
        m = pd.DataFrame([{"symbol": sid, "start_date": st, "end_date": en, "entry_rank": int(e["elig_rank"]),
                           "ranking_as_of": e["ranking_as_of"],
                           "exit_reason": "rebalance" if pd.notna(en) else "open"} for sid, st, en, e in out])
        # A member that stops trading leaves the universe the day after its last trade
        lt = m["symbol"].map(last_trade)
        last_session = max(last_trade.values())
        stopped = lt.notna() & (lt < last_session) & (m["end_date"].isna() | (lt + pd.Timedelta(days=1) < m["end_date"]))
        stopped &= lt >= m["start_date"]
        m.loc[stopped, "end_date"] = lt[stopped] + pd.Timedelta(days=1)
        m.loc[stopped, "exit_reason"] = "stopped_trading"
        m = m[~(lt < m["start_date"])]                       # never traded while a member (rare)
        m["company_name"] = m["symbol"].map(names)
        m["industry"] = pd.NA
        m["source"] = "mcap_rank"
        return m.sort_values(["symbol", "start_date"]).reset_index(drop=True)[MEMBERSHIP_COLUMNS]

    def load_official(self, resolver):
        """Hand-curated intervals (symbol,start_date,end_date[,source]) from NSE index-change notices."""
        if not OFFICIAL_MEMBERSHIP_FILE.exists():
            raise FileNotFoundError(f"universe_mode=official needs {OFFICIAL_MEMBERSHIP_FILE}")
        m = pd.read_csv(OFFICIAL_MEMBERSHIP_FILE, parse_dates=["start_date", "end_date"])
        m["symbol"] = resolver.resolve_frame(m, "symbol", "start_date")
        for c in MEMBERSHIP_COLUMNS:
            if c not in m:
                m[c] = pd.NA
        m["source"] = m["source"].fillna("official")
        return m[MEMBERSHIP_COLUMNS], pd.DataFrame()

    
# =====================================================================
# Step 6: Corporate actions: NSE file, parsers (self-tested), event validation
# =====================================================================
class CorporateActionStore:
    def __init__(self, settings, downloader, logger):
        self.dirs = settings.source_dirs("corporate_actions")
        self.downloader, self.logger = downloader, logger

    def update(self, start, end):
        for yr in range(start.year, end.year + 1):
            name = f"corporate_actions_{yr}.csv"
            path = first_existing(self.dirs, name)
            # past years never change; the current year is re-downloaded until a copy made after `end` exists
            if path is not None and (yr < end.year or date.fromtimestamp(path.stat().st_mtime) > end):
                continue
            if not self.downloader.enabled:
                continue
            try:
                from nselib import capital_market
                ca = with_retry(capital_market.corporate_actions_for_equity, from_date=f"01-01-{yr}",
                                to_date=f"31-12-{yr}", what=f"corporate actions {yr}", logger=self.logger)
                atomic_write_csv(ca, self.dirs[0] / name)
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Corporate actions {yr} not downloaded: {type(e).__name__} {e}")

    def load(self, resolver, start, end):
        frames = []
        for yr in range(start.year, end.year + 1):
            p = first_existing(self.dirs, f"corporate_actions_{yr}.csv")
            if p is None:
                self.logger.warning(f"No corporate-action file for {yr}")
                continue
            frames.append(pd.read_csv(p, dtype=str))
        ca = pd.concat(frames, ignore_index=True)
        ca.columns = clean_cols(ca.columns)
        ca["ex_date"] = parse_dates(ca["exdate"])
        ca = ca.dropna(subset=["ex_date"])
        ca["symbol"] = ca["symbol"].astype(str).str.strip().str.upper()
        ca["sid"] = resolver.resolve_frame(ca, "symbol", "ex_date")
        if "faceval" not in ca:
            ca["faceval"] = np.nan
        atomic_write_csv(ca, CA_FILE, date_format="%Y-%m-%d")
        return ca


class CorporateActionParser:
    """NSE 'subject' text -> numbers. Self-tests run on every pipeline run (CI gate)."""

    @staticmethod
    def dividend_per_share(subject, face_value):
        parts = [p for p in str(subject).lower().split("/") if re.search(r"\bdiv(idend)?\b", p)]
        if not parts:
            return 0.0
        text = " / ".join(parts)
        amounts = re.findall(r"(?:\brs|\bre|\binr|₹)\.?\s*-?\s*(\d+(?:\.\d+)?)", text)
        total = sum(float(a) for a in amounts)
        if not amounts:                                      # "Interim Dividend - 14.50 Per Share" (no Rs)
            total = sum(float(a) for a in re.findall(r"(\d+(?:\.\d+)?)\s*(?:/-)?\s*per\s*(?:equity\s*)?share", text))
        if total == 0:                                       # old style: % of face value
            pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
            fv = pd.to_numeric(face_value, errors="coerce")
            if pcts and pd.notna(fv):
                total = sum(float(p) for p in pcts) / 100 * fv
        return total

    @staticmethod
    def share_events(subject):
        """-> [(type, share_mult)], share_mult = new share count / old share count."""
        s = str(subject).lower()
        events = []
        for x, y in re.findall(r"bonus\s*(?:issue)?\s*(\d+)\s*:\s*(\d+)", s):
            if "debenture" in s or "preference" in s:
                continue
            x, y = float(x), float(y)
            if y > 0:
                events.append(("bonus", (x + y) / y))
        if "split" in s or "sub-div" in s or "consolidat" in s:
            m = re.search(r"from\s*r[se]\.?\s*(\d+(?:\.\d+)?).*?to\s*r[se]\.?\s*(\d+(?:\.\d+)?)", s)
            if m:
                old_fv, new_fv = float(m.group(1)), float(m.group(2))
                if new_fv > 0 and old_fv != new_fv:
                    events.append(("split", old_fv / new_fv))
        return events

    @staticmethod
    def rights(subject, face_value):
        """-> (new_shares, held_shares, issue_price) or None. Issue price = face value + premium."""
        s = str(subject).lower()
        if "right" not in s or any(w in s for w in ["debenture", "ncd", "warrant", "preference"]):
            return None
        ratio = re.search(r"rights?\s*(?:issue)?[^0-9:]{0,40}?(\d+)\s*:\s*(\d+)", s)
        if ratio is None:
            return None
        new_sh, held_sh = float(ratio.group(1)), float(ratio.group(2))
        fv = pd.to_numeric(face_value, errors="coerce")
        if re.search(r"@\s*par\b", s) and pd.notna(fv):
            price = float(fv)
        else:
            p = re.search(r"@\s*(premium\s*(?:of\s*)?)?r[se]\.?\s*(\d+(?:\.\d+)?)", s)
            if p is None:
                return None
            price = float(p.group(2))
            if p.group(1):
                if pd.isna(fv):
                    return None
                price += float(fv)
        if held_sh <= 0 or new_sh <= 0:
            return None
        return new_sh, held_sh, price

    @staticmethod
    def is_demerger(subject):
        return bool(re.search(r"demerg", str(subject), flags=re.IGNORECASE))

    @classmethod
    def self_test(cls):
        tests = [
            (cls.dividend_per_share("Interim Dividend - Rs 5 Per Share", 10), 5.0),
            (cls.dividend_per_share("Dividend - 50%", 10), 5.0),
            (cls.dividend_per_share("Annual General Meeting/Dividend - Rs 10 Per Share/Face Value Split "
                                    "(Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", 10), 10.0),
            (cls.dividend_per_share("Interim Dividend - 14.50 Per Share", 10), 14.5),
            (cls.dividend_per_share("Int Div- 2 Per Share (Purpose Revised)", 2), 2.0),
            (cls.dividend_per_share("Annual General Meeting/ Dividend - Re 0.0 Per Share", 10), 0.0),
            (cls.dividend_per_share("Dividend - Rs 2 Per Share/Rights 1:5 @ Premium Rs 40/-", 10), 2.0),
            (cls.dividend_per_share("Face Value Split (Sub-Division) - From Rs 10/- To Re 1/-", 10), 0.0),
            (cls.share_events("Bonus 1:2"), [("bonus", 1.5)]),
            (cls.share_events("Bonus 4:1"), [("bonus", 5.0)]),
            (cls.share_events(" Bonus 1:5/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- "
                              "Per Share"), [("bonus", 1.2), ("split", 5.0)]),
            (cls.share_events("Annual General Meeting/Dividend Rs.3/- Per Share/Scheme Of Arrangement - "
                              "Bonus Debentures 1:1 (Purpose Revised)"), []),
            (cls.share_events("Face Value Consolidation From Re 1/- Per Share To Rs 10/- Per Share"),
             [("split", 0.1)]),
            (cls.rights("Rights 8:15 @ Premium Rs 40/-", 10), (8.0, 15.0, 50.0)),
            (cls.rights(" Rights 1:14 @ Premium Rs 530/-", 5), (1.0, 14.0, 535.0)),
            (cls.rights("Rights 2:7 @ Rs 205/-", 10), (2.0, 7.0, 205.0)),
            (cls.rights("Rights 1:6 @ Par", 10), (1.0, 6.0, 10.0)),
            (cls.rights("Rights Issue Of Non Convertible Debentures", 10), None),
            (cls.is_demerger("Scheme of Arrangement - Demerger"), True),
        ]
        failures = []
        for got, expected in tests:
            g = [(t, round(f, 4)) for t, f in got] if isinstance(got, list) else got
            if g != expected:
                failures.append(f"got {g}, expected {expected}")
        return failures


# Manual tables (DATA, not code). Symbol = NSE symbol used ON THAT DATE (latest symbol also works).
MANUAL_SHARE_EVENTS = [      # missing from the NSE file; factor implied by the price drop (from the FF project)
    ("MOTHERSUMI", "2015-07-23", 1.5, "bonus"),
    ("SOLARINDS", "2016-07-13", 5.0, "split"),
    ("JSWSTEEL", "2017-01-04", 10.0, "split"),
    ("AJANTPHARM", "2022-06-22", 1.5, "bonus"),
]
FIXED_SHARE_FACTORS = []     # (symbol, ex_date, correct share_mult) when NSE's factor is wrong
DEMERGER_DAYS = [            # NSE labels demergers reliably only from ~2019 (list from the FF project)
    ("ADANIENT", "2015-06-03"), 
    ("JSL", "2015-11-19"),
    ("CROMPGREAV", "2016-03-15"), 
    ("KPIT", "2019-01-24"), 
    ("IIFL", "2019-06-14"),
    ("CENTURYTEX", "2019-10-11"), 
    ("TATACHEM", "2020-03-04"), 
    ("GMRINFRA", "2022-01-11"),
    ("MOTHERSUMI", "2022-01-14"), 
    ("NMDC", "2022-10-27"), 
    ("RELIANCE", "2023-07-20"),
    ("ITC", "2025-01-06"), 
    ("SIEMENS", "2025-04-07"), 
    ("ABFRL", "2025-05-22"),
    ("TATAMOTORS", "2025-10-14"), 
    ("HINDUNILVR", "2025-12-05"), 
    ("VEDL", "2026-04-30"),
]
MANUAL_DIVIDENDS = [         # NSE rows without an amount: (symbol, ex_date, Rs per share, source)
    ("VEDL", "2022-03-09", 13.0, "3rd interim FY22, board 2-Mar-2022"),
    ("SRF", "2017-08-16", 6.0, "interim FY18, board 9-Aug-2017"),
]
REAL_MOVES = [("YESBANK", "2020-03-06")]   # big moves checked and confirmed as genuine
MERGERS = [                  # target merged INTO a listed acquirer: (target, acquirer, acquirer shares per target share)
    ("CAIRN", "VEDL", 1.0, 57.7),  # + 4 x Rs10 7.5% redeemable pref shares + Rs 17.70 dividend (record date 27-Apr-2017)
    ("SBT", "SBIN", 2.2), 
    ("SBBJ", "SBIN", 2.8), 
    ("MYSOREBANK", "SBIN", 2.2),
    ("ABIRLANUVO", "GRASIM", 1.5), 
    ("CAPF", "IDFCBANK", 13.9),
    ("VIJAYABANK", "BANKBARODA", 0.402), 
    ("DENABANK", "BANKBARODA", 0.110),
    ("BHARATFIN", "INDUSINDBK", 0.639), 
    ("GRUH", "BANDHANBNK", 0.568),
    ("SYNDIBANK", "CANBK", 0.158), 
    ("ANDHRABANK", "UNIONBANK", 0.324), 
    ("CORPBANK", "UNIONBANK", 0.330),
    ("ALBK", "INDIANB", 0.115), 
    ("ORIENTBANK", "PNB", 1.150), 
    ("UNITEDBNK", "PNB", 0.121),
    ("GSKCONS", "HINDUNILVR", 4.39), 
    ("TATASTLBSL", "TATASTEEL", 1 / 15), 
    ("MINDTREE", "LTI", 0.73),
    ("SHRIRAMCIT", "SRTRANSFIN", 1.55), 
    ("INOXLEISUR", "PVR", 0.3), 
    ("EQUITAS", "EQUITASBNK", 2.31),
    ("JSLHISAR", "JSL", 1.95), 
    ("HDFC", "HDFCBANK", 42 / 25), 
    ("TATASTLLP", "TATASTEEL", 67 / 10),
    ("UJJIVAN", "UJJIVANSFB", 11.665), 
    ("TCNSBRANDS", "ABFRL", 2.2), 
    ("TATACOFFEE", "TATACONSUM", 3 / 10),
    ("IDFC", "IDFCFIRSTB", 1.55), 
    ("ISEC", "ICICIBANK", 0.67),
]
DELISTINGS = [               # other exits of universe members: (symbol, terminal value per share in Rs, note)
    # e.g. ("DHFL", 0.0, "IBC resolution 2021: equity extinguished") - only if it was a member when it stopped
]


@dataclass
class EventTables:
    share: pd.DataFrame       # sid, ex_date, trade_date, share_mult, status, source
    dividends: pd.DataFrame   # sid, ex_date, trade_date, div_ps
    rights: pd.DataFrame      # sid, ex_date, trade_date, ret_factor, use
    demergers: pd.DataFrame   # sid, ex_date, trade_date, source
    log: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)


class CorporateActionEngine:
    """Builds validated event tables for our stocks from the NSE file + manual tables."""

    STANDARD_FACTORS = [1.25, 4 / 3, 1.5, 2, 2.5, 3, 4, 5, 10, 20, 0.5, 0.2, 0.1]
    P = CorporateActionParser

    def __init__(self, resolver, logger):
        self.resolver, self.logger = resolver, logger

    def _manual(self, rows, cols):
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df.assign(sid=pd.Series(dtype=str))
        df["ex_date"] = pd.to_datetime(df["ex_date"])
        df["sid"] = self.resolver.resolve_frame(df, "symbol", "ex_date")
        return df

    @staticmethod
    def to_trading_day(table, daily, max_days=30):
        """trade_date = first trading day on/after ex_date for the same stock (+ that day's prices)."""
        cols = ["sid", "date", "close", "yesterday_close", "gap_days", "price_jump", "open"]
        if table.empty:
            return table.assign(**{c: pd.Series(dtype=float) for c in cols if c != "sid"}).rename(
                columns={"date": "trade_date"})
        left = table[table["sid"].isin(set(daily["sid"]))].sort_values("ex_date").reset_index(drop=True)
        left["ex_date"] = left["ex_date"].astype("datetime64[ns]")
        right = daily[cols].sort_values("date")
        m = pd.merge_asof(left, right, left_on="ex_date", right_on="date", by="sid",
                          direction="forward", tolerance=pd.Timedelta(days=max_days))
        return m.rename(columns={"date": "trade_date"})

    def build(self, ca, daily):
        P, log = self.P, []
        ours = set(daily["sid"])
        ca = ca[ca["sid"].isin(ours)].copy()

        # --- dividends (one amount per stock and ex-date; repeated rows counted once) ---
        ca["div_ps"] = [P.dividend_per_share(s, fv) for s, fv in zip(ca["subject"], ca["faceval"])]
        no_amount = ca[ca["subject"].str.contains(r"\bdiv(?:idend)?\b", case=False, na=False) & (ca["div_ps"] == 0)]
        md = self._manual([r[:3] for r in MANUAL_DIVIDENDS], ["symbol", "ex_date", "div_ps"])
        md = md[md["sid"].isin(ours)]
        no_amount = no_amount[~no_amount.set_index(["sid", "ex_date"]).index.isin(md.set_index(["sid", "ex_date"]).index)]
        div = (pd.concat([ca.loc[ca["div_ps"] > 0, ["sid", "ex_date", "div_ps"]], md[["sid", "ex_date", "div_ps"]]])
               .drop_duplicates(["sid", "ex_date", "div_ps"])
               .groupby(["sid", "ex_date"], as_index=False)["div_ps"].sum())
        # --- splits / bonuses ---
        rows = [{"sid": sid, "ex_date": ex, "event_type": t, "share_mult": m}
                for sid, ex, subj in zip(ca["sid"], ca["ex_date"], ca["subject"]) for t, m in P.share_events(subj)]
        ev = pd.DataFrame(rows, columns=["sid", "ex_date", "event_type", "share_mult"])
        ev = ev.drop_duplicates(["sid", "ex_date", "event_type", "share_mult"])
        share = (ev.groupby(["sid", "ex_date"], as_index=False)
                 .agg(event_type=("event_type", lambda s: "+".join(sorted(s))), share_mult=("share_mult", "prod")))
        share["source"] = "nse_file"
        for sym, dt, mult in FIXED_SHARE_FACTORS:
            hit = (share["sid"] == self.resolver.resolve(sym, dt)) & (share["ex_date"] == pd.Timestamp(dt))
            share.loc[hit, ["share_mult", "source"]] = [mult, "nse_file, factor fixed"]
            log.append({"sid": sym, "ex_date": dt, "kind": "fixed_factor", "detail": f"rows matched {hit.sum()}"})
        man = self._manual(MANUAL_SHARE_EVENTS, ["symbol", "ex_date", "share_mult", "event_type"])
        man = man[man["sid"].isin(ours)].assign(source="manual (price-implied)")
        share = (pd.concat([share, man.drop(columns="symbol")], ignore_index=True)
                 .drop_duplicates(["sid", "ex_date"], keep="first"))

        # --- rights ---
        rows = []
        for sid, ex, subj, fv in zip(ca["sid"], ca["ex_date"], ca["subject"], ca["faceval"]):
            r = P.rights(subj, fv)
            if r is not None:
                rows.append({"sid": sid, "ex_date": ex, "rights_new": r[0], "rights_held": r[1],
                             "issue_price": r[2], "subject": subj})
        rights = pd.DataFrame(rows, columns=["sid", "ex_date", "rights_new", "rights_held", "issue_price",
                                             "subject"]).drop_duplicates(["sid", "ex_date"])

        # --- demergers (NSE file + manual list) ---
        dm_file = ca[ca["subject"].map(P.is_demerger)][["sid", "ex_date"]].drop_duplicates().assign(source="nse_file")
        dm_man = self._manual(DEMERGER_DAYS, ["symbol", "ex_date"]).assign(source="manual")
        dm_man = dm_man[dm_man["sid"].isin(ours)]
        dem = pd.concat([dm_man[["sid", "ex_date", "source"]], dm_file], ignore_index=True)

        # --- map to trading days ---
        share_m = self.to_trading_day(share, daily)
        div_m = self.to_trading_day(div, daily, max_days=7)
        rights_m = self.to_trading_day(rights, daily)
        dem_all = self.to_trading_day(dem, daily)
        dem_m = (dem_all.dropna(subset=["trade_date"]).sort_values("source")
                 .drop_duplicates(["sid", "trade_date"], keep="first"))
        first_px = daily.groupby("sid")["date"].min()
        last_px = daily.groupby("sid")["date"].max()
        in_range = (dem_all["ex_date"] > dem_all["sid"].map(first_px)) & (dem_all["ex_date"] <= dem_all["sid"].map(last_px))
        unmapped_manual = dem_all[(dem_all["source"] == "manual") & dem_all["trade_date"].isna() & in_range]

        # --- CHECK 1: every split/bonus must match the price drop (NSE prevclose is NOT adjusted) ---
        share_m = share_m.dropna(subset=["trade_date"]).copy()
        ratio = share_m["price_jump"] / share_m["share_mult"]
        share_m["status"] = np.where(share_m["gap_days"] > 5, "after trading gap",
                                     np.where(ratio.between(0.8, 1.25), "confirmed", "FAILED"))
        for i in share_m.index[share_m["status"] == "FAILED"]:
            e = share_m.loc[i]
            win = daily[(daily["sid"] == e["sid"]) & daily["date"].between(e["ex_date"] - pd.Timedelta(days=15),
                                                                          e["ex_date"] + pd.Timedelta(days=15))]
            hit = win[(win["price_jump"] / e["share_mult"]).between(0.8, 1.25) & (win["gap_days"] <= 5)]
            if len(hit):
                h = hit.loc[(hit["date"] - e["ex_date"]).abs().idxmin()]
                share_m.loc[i, ["trade_date", "price_jump", "gap_days"]] = [h["date"], h["price_jump"], h["gap_days"]]
                share_m.loc[i, "status"] = "moved to matching drop"
            else:
                share_m.loc[i, "status"] = "DROPPED (no matching drop)"
            log.append({"sid": e["sid"], "ex_date": e["ex_date"].date(), "kind": "split/bonus check",
                        "detail": f"{share_m.loc[i, 'status']} (mult {e['share_mult']:.4g})"})
        share_m = share_m[share_m["status"] != "DROPPED (no matching drop)"]

        # --- CHECK 2: every big one-day move (fall > 37.5% or rise > 60%) must be explained ---
        real = self.to_trading_day(self._manual(REAL_MOVES, ["symbol", "ex_date"]), daily)
        explained = set()
        for t in (share_m, dem_m, rights_m, real):
            explained |= set(zip(t["sid"], t["trade_date"]))
        big = daily[((daily["price_jump"] > 1.6) | (daily["price_jump"] < 0.625)) & (daily["gap_days"] <= 5)]
        big = big[[k not in explained for k in zip(big["sid"], big["date"])]]
        added = []
        for _, u in big.iterrows():
            f = min(self.STANDARD_FACTORS, key=lambda x: abs(u["price_jump"] / x - 1))
            if abs(u["price_jump"] / f - 1) <= 0.03:
                added.append({"sid": u["sid"], "ex_date": u["date"], "trade_date": u["date"], "share_mult": f,
                              "event_type": "auto", "source": "auto (price-implied)", "status": "auto-added",
                              "price_jump": u["price_jump"], "gap_days": u["gap_days"]})
                log.append({"sid": u["sid"], "ex_date": u["date"].date(), "kind": "auto-added split/bonus",
                            "detail": f"price jump {u['price_jump']:.3f} -> factor {f:.4g}"})
        if added:
            share_m = pd.concat([share_m, pd.DataFrame(added)], ignore_index=True)
        unexplained = big[[(s, d) not in {(a["sid"], a["trade_date"]) for a in added}
                           for s, d in zip(big["sid"], big["date"])]]

        # --- CHECK 3: rights -> CRSP factor = cum price / theoretical ex-rights price ---
        r = rights_m.dropna(subset=["trade_date"]).copy()
        if len(r):
            cum = r["yesterday_close"]
            terp = (r["rights_held"] * cum + r["rights_new"] * r["issue_price"]) / (r["rights_held"] + r["rights_new"])
            r["ret_factor"] = np.where(r["issue_price"] < cum, cum / terp, 1.0)
            confirms = (r["price_jump"] / r["ret_factor"]).between(0.8, 1.25) & (r["gap_days"] <= 5)
            r["use"] = (r["ret_factor"] <= 1.5) | confirms
        else:
            r = r.assign(ret_factor=pd.Series(dtype=float), use=pd.Series(dtype=bool))
        for _, x in r[~r["use"]].iterrows():
            log.append({"sid": x["sid"], "ex_date": x["ex_date"].date(), "kind": "rights NOT applied",
                        "detail": f"factor {x['ret_factor']:.3f} not confirmed by price"})

        checks = {
            "split_bonus_confirmed": int((share_m["status"] == "confirmed").sum()),
            "split_bonus_moved": int((share_m["status"] == "moved to matching drop").sum()),
            "split_bonus_dropped": int(sum("DROPPED" in str(x.get("detail", "")) for x in log)),
            "split_bonus_auto_added": len(added),
            "big_moves_unexplained": int(len(unexplained)),
            "rights_applied": int(r["use"].sum()) if len(r) else 0,
            "rights_not_applied": int((~r["use"]).sum()) if len(r) else 0,
            "dividends_mapped": int(div_m["trade_date"].notna().sum()),
            "dividend_rows_without_amount": int(len(no_amount)),
            "demergers_mapped": int(len(dem_m)),
            "manual_demergers_unmapped": int(len(unmapped_manual)),
        }
        for _, x in unexplained.iterrows():
            log.append({"sid": x["sid"], "ex_date": x["date"].date(), "kind": "big move kept as real",
                        "detail": f"yesterday/today close {x['price_jump']:.3f}"})
        for _, x in no_amount.iterrows():
            log.append({"sid": x["sid"], "ex_date": x["ex_date"].date(), "kind": "dividend without amount",
                        "detail": str(x["subject"])[:80]})
        return EventTables(share=share_m, dividends=div_m.dropna(subset=["trade_date"]),
                           rights=r[r["use"]] if len(r) else r, demergers=dem_m, log=log, checks=checks)

    
# =====================================================================
# Step 7: Daily price table, total-return adjustment, merger continuation
# =====================================================================
class PriceTableBuilder:
    """Bhavcopy rows -> one row per stock per day under its permanent ID (sid)."""

    SERIES_RANK = {"EQ": 0, "BE": 1, "BZ": 2}

    def __init__(self, resolver):
        self.resolver = resolver

    def build(self, bhav, sids):
        wanted = self.resolver.aliases(sids)
        d = bhav[bhav["symbol"].astype(str).isin(wanted)].copy()
        for c in ["symbol", "series", "isin", "name"]:
            d[c] = d[c].astype(str)
        d["sid"] = self.resolver.resolve_frame(d, "symbol", "date")
        d = d[d["sid"].isin(set(sids))]
        d["_r"] = d["series"].map(self.SERIES_RANK)
        d = d.sort_values(["sid", "date", "_r"]).drop_duplicates(["sid", "date"], keep="first").drop(columns="_r")
        d = d.dropna(subset=["close"])
        d = d[d["close"] > 0].reset_index(drop=True)
        d["yesterday_close"] = d.groupby("sid")["close"].shift(1)
        d["gap_days"] = d.groupby("sid")["date"].diff().dt.days
        d["price_jump"] = d["yesterday_close"] / d["close"]          # > 1 = price fell
        return d


class Adjuster:
    """Daily total-return relative per stock, then backward adjustment to today's units.
    R_t = (close_t * share_mult_t + dividend_t) * rights_factor_t / close_{t-1}
    Demerger ex-day: R_t = close_t / open_t (NSE's special pre-open price prices the residual company)."""

    PRICE_COLS = ["openprice", "highprice", "lowprice", "lastprice", "closeprice", "averageprice"]

    def __init__(self, adjust_dividends=True):
        self.adjust_dividends = adjust_dividends

    @staticmethod
    def _per_day(table, value_col, how="prod", default=1.0):
        if table.empty:
            return pd.Series(dtype=float)
        return table.groupby(["sid", "trade_date"])[value_col].agg(how)

    def apply(self, daily, ev):
        d = daily.copy()
        key = pd.MultiIndex.from_arrays([d["sid"], d["date"]])
        mult = pd.Series(self._per_day(ev.share, "share_mult").reindex(key).to_numpy(), index=d.index).fillna(1.0)
        div = pd.Series(self._per_day(ev.dividends, "div_ps", "sum").reindex(key).to_numpy(), index=d.index).fillna(0.0)
        rf = pd.Series(self._per_day(ev.rights, "ret_factor").reindex(key).to_numpy(), index=d.index).fillna(1.0)
        dem = pd.Series(key.isin(pd.MultiIndex.from_arrays([ev.demergers["sid"], ev.demergers["trade_date"]])),
                        index=d.index) if len(ev.demergers) else pd.Series(False, index=d.index)

        prev = d["yesterday_close"]
        r_price = (d["close"] * mult) * rf / prev
        r_tr = (d["close"] * mult + div) * rf / prev
        dem_rel = (d["close"] / d["open"]).where(d["open"] > 0, 1.0)
        dem_rel = dem_rel.where(d["open"] < prev, 1.0)   # no drop at the open -> nothing transferred
        r_price = r_price.where(~dem, dem_rel)
        r_tr = r_tr.where(~dem, dem_rel)
        first = prev.isna()
        r_price[first], r_tr[first] = 1.0, 1.0
        d["share_mult"], d["div_ps"], d["rights_factor"], d["is_demerger_day"] = mult, div, rf, dem.astype(int)
        d["ret_price"], d["ret_tr"] = r_price - 1, r_tr - 1

        rel = r_tr if self.adjust_dividends else r_price
        g = d.groupby("sid")
        idx = np.exp(np.log(rel).groupby(d["sid"]).cumsum())        # total-return index (1 at first day)
        last_idx, last_close = idx.groupby(d["sid"]).transform("last"), g["close"].transform("last")
        d["adj_closeprice_tmp"] = last_close * idx / last_idx
        d["adj_factor"] = d["adj_closeprice_tmp"] / d["close"]
        idx_p = np.exp(np.log(r_price).groupby(d["sid"]).cumsum())
        d["adj_factor_price"] = (last_close * idx_p / idx_p.groupby(d["sid"]).transform("last")) / d["close"]
        # share factor: shares today per share held on day t (split/bonus only) -> volumes in today's units
        later_mult = np.exp(np.log(mult)[::-1].groupby(d["sid"][::-1]).cumsum())[::-1] / mult
        d["share_factor"] = later_mult
        return d.drop(columns=["adj_closeprice_tmp"])


class ExitContinuation:
    """A member that stops trading must not silently drop out of forward returns.
    Merger: holders now own `ratio` acquirer shares -> synthetic rows follow the acquirer's total return.
    Other delisting (DELISTINGS table): synthetic rows hold the terminal value (0 = wiped out).
    Synthetic rows (series SYN, is_synthetic=1) cover `days` sessions; the stock is out of the universe then."""

    def __init__(self, resolver, logger, days=30, max_swap=0.25):
        self.resolver, self.logger, self.days, self.max_swap = resolver, logger, days, max_swap

    def acquirer_sids(self, sids):
        out = {}
        for row in MERGERS:
            tgt, acq, ratio = row[:3]
            cash = float(row[3]) if len(row) > 3 else 0.0
            t = self.resolver.final(tgt)
            if t in sids:
                out[t] = (self.resolver.final(acq), ratio, cash)
        return out
    
    @staticmethod
    def _rows(sid, dates, values):
        v = np.asarray(values, dtype=float)
        return pd.DataFrame({"sid": sid, "date": dates, "symbol": sid, "series": "SYN", "close": v, "open": v,
                             "high": v, "low": v, "last": v, "volume": 0.0, "turnover": 0.0, "trades": 0.0,
                             "adj_factor": 1.0, "adj_factor_price": 1.0, "share_factor": 1.0, "is_synthetic": 1})

    def build(self, adj, all_prices, sessions):
        last_session = sessions.max()
        rows, log = [], []
        
        for tgt, (acq, ratio, cash) in self.acquirer_sids(set(adj["sid"])).items():
            t = adj[adj["sid"] == tgt]
            a = all_prices[all_prices["sid"] == acq].sort_values("date")
            t_last = t["date"].max()
            if t_last >= last_session or a.empty:
                continue
            a = a[a["date"] > t_last].head(self.days)
            if a.empty:
                continue
            last_close = t.loc[t["date"] == t_last, "close"].iloc[0]
            first_val = ratio * a["close"].iloc[0]
            swap = (first_val + cash) / last_close - 1
            if abs(swap) > self.max_swap:
                log.append({"sid": tgt, "ex_date": t_last.date(), "kind": "merger SKIPPED",
                            "detail": f"swap return {swap:+.1%} beyond +/-{self.max_swap:.0%}: check ratio {ratio:.4g}"})
                continue
            val = first_val * a["adj_closeprice"] / a["adj_closeprice"].iloc[0] + cash   # acquirer TR + cash part
            rows.append(self._rows(tgt, a["date"].values, val.values))
            log.append({"sid": tgt, "ex_date": t_last.date(), "kind": "merger continuation",
                        "detail": f"-> {acq} x{ratio:.4g} + Rs {cash:g}, swap return {swap:+.1%}, {len(val)} synthetic rows"})
        
        for sym, value, note in DELISTINGS:
            tgt = self.resolver.final(sym)
            t = adj[adj["sid"] == tgt]
            if t.empty or t["date"].max() >= last_session:
                continue
            t_last = t["date"].max()
            dates = sessions[sessions > t_last][: self.days]
            value = max(float(value), 0.01)     # Phase 2 drops close <= 0; Rs 0.01 = economically -100%
            rows.append(self._rows(tgt, dates, np.full(len(dates), value)))
            ret = value / t.loc[t["date"] == t_last, "close"].iloc[0] - 1
            log.append({"sid": tgt, "ex_date": t_last.date(), "kind": "delisting terminal value",
                        "detail": f"Rs {value} per share ({ret:+.1%}): {note}"})
        return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), log


class RawFileWriter:
    """Writes data/raw/<SID>.csv in the column layout Phase 2 already reads."""

    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def to_raw_layout(g):
        f, sh = g["adj_factor"], g["share_factor"]
        avg = (g["turnover"] / g["volume"]).where(g["volume"] > 0, g["close"])
        out = pd.DataFrame({
            "symbol": g["sid"], "symbol_traded": g["symbol"], "series": g["series"],
            "date": g["date"], "prevclose": g["prevclose"] if "prevclose" in g else np.nan,
            "openprice": g["open"], "highprice": g["high"], "lowprice": g["low"], "lastprice": g["last"],
            "closeprice": g["close"], "averageprice": avg, "totaltradedquantity": g["volume"],
            "turnoverinrs": g["turnover"], "no.oftrades": g["trades"],
        })
        for c in ["openprice", "highprice", "lowprice", "lastprice", "closeprice", "averageprice"]:
            out[f"adj_{c}"] = out[c] * f
        out["adj_totaltradedquantity"] = out["totaltradedquantity"] * sh
        out["adj_factor"] = f
        out["adj_factor_price"] = g["adj_factor_price"]
        out["share_factor"] = sh
        for c in ["ret_price", "ret_tr", "div_ps", "is_demerger_day", "gap_days"]:
            out[c] = g[c] if c in g else np.nan
        out["is_synthetic"] = g.get("is_synthetic", 0)
        return out

    def write(self, frame, sids):
        frame = frame.copy()
        frame["is_synthetic"] = frame.get("is_synthetic", 0)
        frame["is_synthetic"] = frame["is_synthetic"].fillna(0).astype(int)
        written = []
        for sid, g in frame.groupby("sid"):
            if sid not in sids:
                continue
            out = self.to_raw_layout(g.sort_values("date"))
            atomic_write_csv(out, RAW_DIR / f"{sid}.csv", date_format="%Y-%m-%d", float_format="%.10g")
            written.append(sid)
        # files of stocks no longer in the universe -> moved to a backup, never deleted
        stale = [p for p in RAW_DIR.glob("*.csv") if p.stem not in set(written)]
        if stale:
            dest = BACKUP_DIR / f"raw_not_in_universe_{datetime.now():%Y%m%d_%H%M%S}"
            dest.mkdir(parents=True, exist_ok=True)
            for p in stale:
                p.replace(dest / p.name)
            self.logger.info(f"Moved {len(stale)} raw files not in the universe to {dest}")
        return written

    
    
# =====================================================================
# Step 8: Quality gate (critical = pipeline fails, warning = review)
# =====================================================================
KNOWN_EVENTS = [
    ("BAJFINANCE", "2025-06-16"), ("HDFCBANK", "2025-08-26"), ("RELIANCE", "2024-10-28"),
    ("TMPV", "2025-10-14"), ("EICHERMOT", "2020-08-24"), ("NESTLEIND", "2024-01-05"),
    ("KOTAKBANK", "2026-01-14"), ("VBL", "2024-09-12"), ("ITC", "2025-01-06"), ("RELIANCE", "2023-07-20"),
    ("JSWSTEEL", "2017-01-04"), ("TATASTEEL", "2022-07-28"),
]
KNOWN_EVENT_TOLERANCE = 0.12
LARGE_MOVE = 0.20
MAX_FAILED_SHARE = 0.05  # [#6]
STALE_DAYS = 5  # [#6]
MIN_OFFICIAL_OVERLAP = 0.85
LARGE_MOVES_FILE = DATA_DIR / "large_moves_review.csv"
BENCHMARK_FILE = DATA_DIR / "benchmarks" / "nifty100_ew_tri.csv"   # NIFTY100 EQUAL WEIGHT TRI (niftyindices.com)
UNIVERSE_RETURNS_FILE = INTERIM_DIR / "universe_ew_returns.csv"     # PIT equal-weight universe (reused in Phase 3)
TRACKING_LIMITS = {"corr_critical": 0.90, "corr_warning": 0.97, "te_warning": 0.04}
NIFTY100_URL = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"


class QualityGate:
    def __init__(self, settings, logger):
        self.s, self.logger, self.rows = settings, logger, []

    def add(self, check, status, detail):
        self.rows.append((check, status, detail))

    def bhav_coverage(self, sessions, end):
        per_year = pd.Series(1, index=sessions).groupby(sessions.year).sum()
        full_years = [y for y in per_year.index if self.s.price_start.year < y < end.year]
        thin = {int(y): int(per_year[y]) for y in full_years if per_year[y] < 240}   # NSE: ~245-252 sessions/yr
        self.add("bhavcopy_coverage", "warning" if thin else "ok",
                 f"years with < 240 bhavcopy files (missing days): {thin}" if thin else
                 f"{len(sessions)} sessions, every full year has >= 240 files")
        lag = (pd.Timestamp(end) - sessions.max()).days
        self.add("data_freshness", "warning" if lag > 4 else "ok",
                 f"last bhavcopy {sessions.max().date()} vs end date {end} ({lag} days)")

    def parser_tests(self):
        fails = CorporateActionParser.self_test()
        self.add("parser_self_test", "critical" if fails else "ok", "; ".join(fails) or "all parser tests passed")

    def events(self, checks):
        self.add("manual_demergers_unmapped", "critical" if checks["manual_demergers_unmapped"] else "ok",
                 f"{checks['manual_demergers_unmapped']} manual demergers without a trading day (must be 0)")
        self.add("split_bonus_validation", "warning" if checks["split_bonus_dropped"] + checks["split_bonus_auto_added"]
                 else "ok", f"confirmed {checks['split_bonus_confirmed']}, moved {checks['split_bonus_moved']}, "
                 f"dropped {checks['split_bonus_dropped']}, auto-added {checks['split_bonus_auto_added']} "
                 f"(see {ADJ_LOG_FILE.name})")
        self.add("big_moves_unexplained", "warning" if checks["big_moves_unexplained"] else "ok",
                 f"{checks['big_moves_unexplained']} one-day moves >37.5% down / >60% up kept as real")
        self.add("rights", "warning" if checks["rights_not_applied"] else "ok",
                 f"{checks['rights_applied']} rights factors applied, {checks['rights_not_applied']} not applied")
        self.add("dividends", "warning" if checks["dividend_rows_without_amount"] else "ok",
                 f"{checks['dividends_mapped']} dividends applied ({'total-return' if self.s.adjust_dividends else 'NOT applied: price-only'}); "
                 f"{checks['dividend_rows_without_amount']} dividend rows without an amount")

    def known_events(self, adj):
        for sym, d in KNOWN_EVENTS:
            g = adj[adj["sid"] == sym]
            row = g[g["date"] == pd.Timestamp(d)]
            if row.empty:
                continue
            r = row["ret_tr"].iloc[0]
            self.add("known_event", "ok" if abs(r) < KNOWN_EVENT_TOLERANCE else "critical",
                     f"{sym} {d}: adjusted 1-day return {r:+.1%}")

    def large_moves(self, adj):
        big = adj[(adj["ret_tr"].abs() > LARGE_MOVE) & (adj.get("is_synthetic", 0) == 0)]
        out = big[["sid", "date", "ret_tr", "gap_days", "is_demerger_day"]].rename(
            columns={"sid": "symbol", "ret_tr": "adj_ret"})
        atomic_write_csv(out, LARGE_MOVES_FILE, date_format="%Y-%m-%d")
        self.add("large_moves", "warning" if len(out) else "ok",
                 f"{len(out)} moves > {LARGE_MOVE:.0%} listed in {LARGE_MOVES_FILE.name}")

    def universe(self, m, reb, adj, last_session):
        n = self.s.universe_size
        if len(reb):
            small = reb[reb["members"] < 0.95 * n]
            self.add("universe_size", "critical" if len(small) else "ok",
                     f"{len(reb)} rebalances, members min {reb['members'].min()} / max {reb['members'].max()}")
            lag_ok = reb["effective"] >= reb["ranking_as_of"] + pd.Timedelta(days=self.s.ranking_lag_days)
            self.add("point_in_time", "ok" if lag_ok.all() else "critical",
                     "every rebalance uses a ranking file dated >= lag days before it")
        closed = m["end_date"].notna().sum()
        self.add("survivorship", "ok" if closed > 0 else "critical",
                 f"{m['symbol'].nunique()} stocks ever in the universe, {closed} closed intervals "
                 f"({(m['exit_reason'] == 'stopped_trading').sum()} stopped trading while a member)")
        open_now = m[m["end_date"].isna()]
        self.add("current_members", "ok" if 0.95 * n <= len(open_now) <= 1.05 * n else "critical",
                 f"{len(open_now)} current members")
        have = set(adj["sid"])
        missing = sorted(set(m["symbol"]) - have)
        self.add("missing_prices", "critical" if missing else "ok", f"members without prices: {missing[:20]}")
        last = adj[adj["is_synthetic"] == 0].groupby("sid")["date"].max()
        stale = sorted(s for s in open_now["symbol"] if s in last and (last_session - last[s]).days > STALE_DAYS)
        self.add("stale_current_members", "warning" if stale else "ok", f"current members > {STALE_DAYS} days stale: {stale}")
        stopped = m[m["exit_reason"] == "stopped_trading"]["symbol"]
        syn = set(adj.loc[adj["is_synthetic"] == 1, "sid"])
        delisted_known = {s for s, *_ in DELISTINGS}
        unexplained = sorted(set(stopped) - syn - delisted_known)
        self.add("exits_without_terminal_value", "warning" if unexplained else "ok",
                 f"members that stopped trading with no merger/delisting entry (forward returns will be NaN): {unexplained}")

    def official_overlap(self, m, resolver, downloader, end):
        try:
            cur = pd.read_csv(NIFTY100_URL) if downloader.enabled else None
        except Exception as e:
            cur = None
            self.logger.warning(f"Could not read today's Nifty 100 list: {e}")
        if cur is None:
            self.add("official_overlap", "warning", "today's NSE Nifty 100 list not available (offline)")
            return
        cur.columns = clean_cols(cur.columns)
        cur["when"] = pd.Timestamp(end)
        off = set(resolver.resolve_frame(cur, "symbol", "when"))
        ours = set(m.loc[m["end_date"].isna(), "symbol"])
        ov = len(off & ours) / max(len(off), 1)
        self.add("official_overlap", "ok" if ov >= MIN_OFFICIAL_OVERLAP else "warning",
                 f"{ov:.0%} of today's Nifty 100 is in our current universe | only official: "
                 f"{sorted(off - ours)[:15]} | only ours: {sorted(ours - off)[:15]}")
    
    @staticmethod
    def universe_ew_returns(m, full):
        """Daily equal-weight total return of the PIT universe: a stock counts on day t if it was a member
        at the close of t-1 (synthetic merger rows included: that value is what holders really got)."""
        px = (full.assign(p=full["close"] * full["adj_factor"])
              .pivot_table(index="date", columns="sid", values="p", aggfunc="last").sort_index())
        ret = px.pct_change(fill_method=None)
        held = pd.DataFrame(False, index=ret.index, columns=ret.columns)
        prev_day = pd.Series(ret.index, index=ret.index).shift(1)
        for row in m.itertuples():
            if row.symbol in held.columns:
                end = row.end_date if pd.notna(row.end_date) else pd.Timestamp.max
                held.loc[(prev_day >= row.start_date) & (prev_day < end), row.symbol] = True
        ew = ret.where(held).mean(axis=1)
        return ew[held.any(axis=1)].rename("universe_ew")

    @staticmethod
    def read_index_file(path):
        """niftyindices.com export: a date column + one value column (total-return index level)."""
        b = pd.read_csv(path)
        b.columns = clean_cols(b.columns)
        dcol = next(c for c in b.columns if "date" in c)
        vcol = next(c for c in b.columns if c != dcol and pd.to_numeric(
            b[c].astype(str).str.replace(",", "", regex=False), errors="coerce").notna().mean() > 0.9)
        out = pd.Series(pd.to_numeric(b[vcol].astype(str).str.replace(",", "", regex=False), errors="coerce").values,
                        index=parse_dates(b[dcol])).dropna().sort_index()
        return out[~out.index.duplicated(keep="last")]

    def benchmark_tracking(self, m, full):
        """Data check, not a performance number: the PIT universe must move like the official index."""
        ew = self.universe_ew_returns(m, full)
        out = ew.to_frame()
        if not BENCHMARK_FILE.exists():
            atomic_write_csv(out.reset_index(names="date"), UNIVERSE_RETURNS_FILE, date_format="%Y-%m-%d")
            self.add("benchmark_tracking", "warning", f"{BENCHMARK_FILE.name} not found: save the NIFTY100 "
                     f"EQUAL WEIGHT TRI there (Date, value) to validate the universe")
            return
        tri = self.read_index_file(BENCHMARK_FILE)
        out["benchmark"] = tri.pct_change().reindex(out.index)
        atomic_write_csv(out.reset_index(names="date"), UNIVERSE_RETURNS_FILE, date_format="%Y-%m-%d")
        both = out.dropna()
        if len(both) < 250:
            self.add("benchmark_tracking", "warning", f"only {len(both)} overlapping days with {BENCHMARK_FILE.name}")
            return
        corr = both["universe_ew"].corr(both["benchmark"])
        te = (both["universe_ew"] - both["benchmark"]).std() * np.sqrt(252)
        yrs = len(both) / 252
        cagr = (1 + both).prod() ** (1 / yrs) - 1
        lim = TRACKING_LIMITS
        status = ("critical" if corr < lim["corr_critical"] else
                  "warning" if corr < lim["corr_warning"] or te > lim["te_warning"] else "ok")
        self.add("benchmark_tracking", status,
                 f"{both.index.min().date()} -> {both.index.max().date()}: daily corr {corr:.3f}, tracking error "
                 f"{te:.2%}/yr, CAGR universe {cagr['universe_ew']:.2%} vs index {cagr['benchmark']:.2%}")
    
    def write(self):
        rep = pd.DataFrame(self.rows, columns=["check", "status", "detail"])
        rep.insert(0, "run_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        atomic_write_csv(rep, QUALITY_REPORT)
        for _, r in rep.iterrows():
            log = self.logger.error if r["status"] == "critical" else (
                self.logger.warning if r["status"] == "warning" else self.logger.info)
            log(f"[{r['check']}] {r['status'].upper()}: {r['detail']}")
        return not (rep["status"] == "critical").any()

    
    
# =====================================================================
# Step 9: Orchestration
# =====================================================================
class PricePipeline:
    def __init__(self, settings, end, logger, offline=False, rebuild_cache=False):
        self.s, self.end, self.logger = settings, end, logger
        self.rebuild = rebuild_cache
        self.dl = NSEDownloader(logger, enabled=not offline)

    def run(self):
        s, end, log = self.s, self.end, self.logger
        resolver = SymbolResolver.load(s, self.dl, end, log)
        bhav_store = BhavcopyStore(s, self.dl, log)
        bhav_store.download_missing(s.price_start, end)

        # 1) candidates from the ranking files (+ merger acquirers) -> one bhavcopy read
        rankings = RankingStore(s, self.dl, log).load(resolver) if s.universe_mode == "mcap_rank" else None
        cand = set(rankings.loc[rankings["rank"] <= s.candidate_rank, "sid"]) if rankings is not None else set()
        builder = UniverseBuilder(s, log)
        
        if s.universe_mode == "official":
            membership, reb = builder.load_official(resolver)
            cand = set(membership["symbol"])
        acq = {resolver.final(m[1]) for m in MERGERS}
        bhav, sessions = bhav_store.load(resolver.aliases(cand | acq), s.price_start, end, rebuild=self.rebuild)
        
        if len(sessions) == 0:
            raise RuntimeError("No bhavcopy files: run online once, or set ML_ALPHA_EXTRA_SOURCE_DIRS")
        
        last_session = sessions.max()
        ptb = PriceTableBuilder(resolver)
        daily_all = ptb.build(bhav, cand | acq)
        names = (daily_all.dropna(subset=["name"]).query("name != 'nan'")
                 .sort_values("date").groupby("sid")["name"].last().to_dict())

        # 2) point-in-time universe
        if s.universe_mode == "mcap_rank":
            presence = daily_all[["sid", "date"]]
            last_trade = daily_all.groupby("sid")["date"].max().to_dict()
            membership, reb = builder.build(rankings, presence, sessions, last_trade, names, end)
        membership["company_name"] = membership["company_name"].fillna(membership["symbol"].map(names))
        atomic_write_csv(membership, MEMBERSHIP_FILE, date_format="%Y-%m-%d")
        
        if len(reb):
            atomic_write_csv(reb.assign(added=reb["added"].map(" ".join), removed=reb["removed"].map(" ".join)),
                             REBALANCE_FILE, date_format="%Y-%m-%d")
        members = set(membership["symbol"])
        log.info(f"Universe: {len(members)} stocks ever, {membership['end_date'].isna().sum()} current")

        # 3) corporate actions -> validated events -> adjusted prices (members + acquirers)
        ca_store = CorporateActionStore(s, self.dl, log)
        ca_store.update(s.price_start, end)
        ca = ca_store.load(resolver, s.price_start, end)
        daily = daily_all[daily_all["sid"].isin(members | acq)].reset_index(drop=True)
        engine = CorporateActionEngine(resolver, log)
        ev = engine.build(ca, daily)
        adj = Adjuster(s.adjust_dividends).apply(daily, ev)
        adj["adj_closeprice"] = adj["close"] * adj["adj_factor"]
        adj["is_synthetic"] = 0
        syn, mlog = ExitContinuation(resolver, log).build(adj[adj["sid"].isin(members)], adj, sessions)
        full = pd.concat([adj, syn], ignore_index=True) if len(syn) else adj
        written = RawFileWriter(log).write(full[full["sid"].isin(members)], members)

        # 4) logs + quality gate + manifest
        ev_log = ev.log + mlog
        share_rows = [{"sid": r.sid, "ex_date": r.ex_date.date(), "kind": f"split/bonus applied ({r.status})",
                       "detail": f"x{r.share_mult:.4g} on {pd.Timestamp(r.trade_date).date()}"}
                      for r in ev.share.itertuples()]
        atomic_write_csv(pd.DataFrame(ev_log + share_rows, columns=["sid", "ex_date", "kind", "detail"]),
                         ADJ_LOG_FILE)
        report = pd.DataFrame({"symbol": written})
        atomic_write_csv(report, PRICE_RUN_REPORT)
        gate = QualityGate(s, log)
        gate.bhav_coverage(sessions, end)
        gate.parser_tests()
        gate.events(ev.checks)
        gate.known_events(full)
        gate.large_moves(full[full["sid"].isin(members)])
        gate.universe(membership, reb, full, last_session)
        gate.official_overlap(membership, resolver, self.dl, end)
        gate.benchmark_tracking(membership, full)
        ok = gate.write()
        self.manifest(membership, reb, sessions, ev.checks, ok, len(written))
        return ok

    def manifest(self, m, reb, sessions, checks, ok, n_files):
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            commit = ""
        atomic_write_json({
            "run_at": datetime.now().isoformat(timespec="seconds"), "end_date": str(self.end),
            "last_session": str(sessions.max().date()), "first_session": str(sessions.min().date()),
            "n_sessions": int(len(sessions)), "git_commit": commit, "settings": self.s.to_dict(),
            "universe": {"stocks_ever": int(m["symbol"].nunique()), "current": int(m["end_date"].isna().sum()),
                         "rebalances": int(len(reb))},
            "raw_files_written": n_files, "corporate_action_checks": checks, "passed": bool(ok),
            "pandas": pd.__version__, "numpy": np.__version__,
        }, MANIFEST_FILE)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 1 prices: PIT Nifty 100 universe from NSE bhavcopies")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD data cutoff (default: today / ML_ALPHA_END_DATE)")
    parser.add_argument("--offline", action="store_true", help="no downloads: use cached NSE files only (CI)")
    parser.add_argument("--rebuild-cache", action="store_true", help="re-read every bhavcopy file")
    parser.add_argument("--full-refresh", action="store_true", help="alias of --rebuild-cache (kept for old scripts)")
    args = parser.parse_args(argv)

    logger = setup_logging("update_prices")
    ensure_dirs()
    end_txt = args.end_date or os.environ.get("ML_ALPHA_END_DATE", "").strip()
    end = date.fromisoformat(end_txt) if end_txt else date.today()
    logger.info(f"Price update to {end} | settings={SETTINGS.to_dict()} | project={PROJECT_ROOT}")
    if not (RAW_DIR / "_source_bhavcopy.txt").exists() and any(RAW_DIR.glob("*.csv")):
        logger.info(f"First bhavcopy run: old raw files backed up to {backup_folder(RAW_DIR)}")
    ok = PricePipeline(SETTINGS, end, logger, offline=args.offline,
                       rebuild_cache=args.rebuild_cache or args.full_refresh).run()
    atomic_write_bytes(b"data/raw is rebuilt from NSE bhavcopies by scripts/update_prices.py\n",
                       RAW_DIR / "_source_bhavcopy.txt")
    logger.info("PRICE UPDATE " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
