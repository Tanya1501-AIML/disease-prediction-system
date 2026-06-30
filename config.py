# ─────────────────────────────────────────────────────────────
# config.py — Central Configuration File
# All paths, constants, and settings live here.
# Every other file imports from this file — never hardcode paths.
# ─────────────────────────────────────────────────────────────

import os

# ── Base directory ────────────────────────────────────────────
# __file__ = the path to this config.py file itself
# os.path.dirname gives us the folder it lives in
# This makes all paths work correctly regardless of which
# machine or OS the project is running on.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ────────────────────────────────────────────────
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")

# Raw dataset file paths (we download these in Phase 2)
HEART_DATA_PATH    = os.path.join(RAW_DATA_DIR, "heart.csv")
DIABETES_DATA_PATH = os.path.join(RAW_DATA_DIR, "diabetes.csv")
CANCER_DATA_PATH   = os.path.join(RAW_DATA_DIR, "breast_cancer.csv")

# ── Model save paths ──────────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")

HEART_MODEL_PATH    = os.path.join(MODELS_DIR, "heart_model.pkl")
DIABETES_MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_model.pkl")
CANCER_MODEL_PATH   = os.path.join(MODELS_DIR, "cancer_model.pkl")

# ── Reports / output paths ────────────────────────────────────
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ── ML Settings ───────────────────────────────────────────────
# Random state ensures results are reproducible.
# If you train the model twice with the same RANDOM_STATE,
# you get identical results — important for debugging and comparison.
RANDOM_STATE = 42

# Fraction of data used for testing (20% test, 80% train)
TEST_SIZE = 0.2

# Number of cross-validation folds
CV_FOLDS = 5

# ── Disease labels ────────────────────────────────────────────
# These are used in the Streamlit app for display
DISEASES = {
    "heart":    "Heart Disease",
    "diabetes": "Diabetes",
    "cancer":   "Breast Cancer",
}

# ── Target column names in each dataset ───────────────────────
TARGET_COLUMNS = {
    "heart":    "target",
    "diabetes": "Outcome",
    "cancer":   "target",   # will be added when loading sklearn dataset
}

# ─────────────────────────────────────────────────────────────
# HOW TO USE THIS FILE IN OTHER SCRIPTS:
#
#   from config import HEART_DATA_PATH, RANDOM_STATE
#
# That's it. No hardcoded paths anywhere else in the project.
# ─────────────────────────────────────────────────────────────
