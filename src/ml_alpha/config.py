# ============================================================
# ML ALPHA PORTFOLIO MODEL - CONFIGURATION
# ============================================================

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"


# Results directories
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PORTFOLIO_DIR = RESULTS_DIR / "portfolio"
EXPLANATIONS_DIR = RESULTS_DIR / "explanations"


# ============================================================
# PORTFOLIO PARAMETERS
# ============================================================

TOP_N = 10

MAX_WEIGHT = 0.10

TRANSACTION_COST = 0.001


# ============================================================
# PREDICTION HORIZON
# ============================================================

# 21 trading days ≈ 1 month
PREDICTION_HORIZON = 21


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

TRAIN_DAYS = 84

PURGE_DAYS = 21

TEST_DAYS = 21

STEP_SIZE = 21


# ============================================================
# FAMA-FRENCH 5 FACTOR PARAMETERS
# ============================================================

# Annual portfolio formation month
FORMATION_MONTH = 7

# Size breakpoint
SIZE_BREAKPOINT = "median"

# BM breakpoints
BM_LOW = 0.30
BM_HIGH = 0.70


# ============================================================
# RAG PARAMETERS
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RAG_TOP_K = 5


# ============================================================
# MODEL PARAMETERS
# ============================================================

MODEL_TYPE = "lightgbm"