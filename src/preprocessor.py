# ─────────────────────────────────────────────────────────────
# src/preprocessor.py
# Cleans raw data and prepares it for model training.
# Handles: missing values, feature scaling, train-test split.
# ─────────────────────────────────────────────────────────────

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HEART_DATA_PATH, DIABETES_DATA_PATH, CANCER_DATA_PATH,
    PROCESSED_DIR, MODELS_DIR,
    RANDOM_STATE, TEST_SIZE
)


# ── Shared utility: scale + split ───────────────────────────────
def scale_and_split(X, y, scaler_save_path=None):
    """
    Splits data into train/test sets, then scales features.

    WHY WE SPLIT BEFORE SCALING:
    The scaler learns mean/std ONLY from training data.
    If we scaled before splitting, information from the test set
    would "leak" into training — this is called DATA LEAKAGE
    and gives falsely optimistic results. Real-world data won't
    have this luxury, so we must avoid it during development too.

    Parameters:
        X : features (DataFrame)
        y : target (Series)
        scaler_save_path : where to save the fitted scaler (for later use in the app)

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    # Step 1: Split FIRST (before scaling) — prevents data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y   # ensures train/test have same class balance as original data
    )

    # Step 2: Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # learn + apply
    X_test_scaled = scaler.transform(X_test)          # only apply (no learning)

    # Step 3: Save the scaler — we'll need the EXACT same scaler
    # later when the Streamlit app receives a new patient's data
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"  Scaler saved → {scaler_save_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test


# ── Preprocessing: Heart Disease ────────────────────────────────
def preprocess_heart_data():
    """
    Cleans the heart disease dataset.

    Issues to fix:
    - 'ca' and 'thal' columns have a few missing values (NaN)
      We fill these using the MEDIAN — median is preferred over mean
      because it's not affected by outliers (robust statistic).
    """
    print("Preprocessing Heart Disease data...")
    df = pd.read_csv(HEART_DATA_PATH)

    # Separate features (X) from target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Fill missing values with median of each column
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )

    print(f"  Missing values before: {X.isnull().sum().sum()}")
    print(f"  Missing values after:  {X_imputed.isnull().sum().sum()}")

    scaler_path = os.path.join(MODELS_DIR, "heart_scaler.pkl")
    X_train, X_test, y_train, y_test = scale_and_split(X_imputed, y, scaler_path)

    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test, X_imputed.columns.tolist()


# ── Preprocessing: Diabetes ─────────────────────────────────────
def preprocess_diabetes_data():
    """
    Cleans the diabetes dataset.

    Issue to fix:
    - Glucose, BloodPressure, SkinThickness, Insulin, BMI have
      ZERO values that are biologically impossible.
      These zeros are actually MISSING values in disguise.
      We replace them with NaN first, then fill with median.
    """
    print("Preprocessing Diabetes data...")
    df = pd.read_csv(DIABETES_DATA_PATH)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Columns where zero is medically impossible
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Step 1: Replace impossible zeros with NaN (so we can treat them as missing)
    X_clean = X.copy()
    for col in zero_invalid_cols:
        X_clean[col] = X_clean[col].replace(0, np.nan)

    zeros_found = X[zero_invalid_cols].eq(0).sum().sum()
    print(f"  Impossible zero values found: {zeros_found}")

    # Step 2: Fill missing values with median
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_clean),
        columns=X_clean.columns
    )
    print(f"  Missing values after imputation: {X_imputed.isnull().sum().sum()}")

    scaler_path = os.path.join(MODELS_DIR, "diabetes_scaler.pkl")
    X_train, X_test, y_train, y_test = scale_and_split(X_imputed, y, scaler_path)

    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test, X_imputed.columns.tolist()


# ── Preprocessing: Breast Cancer ────────────────────────────────
def preprocess_cancer_data():
    """
    Cleans the breast cancer dataset.

    This dataset is already clean (no missing values, no impossible
    zeros) — sklearn's built-in datasets are pre-cleaned.
    We only need to scale and split it.
    """
    print("Preprocessing Breast Cancer data...")
    df = pd.read_csv(CANCER_DATA_PATH)

    X = df.drop('target', axis=1)
    y = df['target']

    print(f"  Missing values: {X.isnull().sum().sum()} (should be 0)")

    scaler_path = os.path.join(MODELS_DIR, "cancer_scaler.pkl")
    X_train, X_test, y_train, y_test = scale_and_split(X, y, scaler_path)

    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ── Master function — preprocess everything at once ────────────
def preprocess_all():
    """
    Runs preprocessing for all 3 diseases and returns a dictionary
    with everything needed for model training.
    """
    print("=" * 50)
    print("  PREPROCESSING ALL DATASETS")
    print("=" * 50 + "\n")

    results = {
        'heart': preprocess_heart_data(),
        'diabetes': preprocess_diabetes_data(),
        'cancer': preprocess_cancer_data(),
    }

    print("=" * 50)
    print("  ALL DATASETS PREPROCESSED SUCCESSFULLY!")
    print("=" * 50)

    return results


# ── Run this file directly to test preprocessing ───────────────
if __name__ == "__main__":
    results = preprocess_all()

    print("\n── PREPROCESSING SUMMARY ──")
    for name, (X_train, X_test, y_train, y_test, feature_names) in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Features used: {len(feature_names)}")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Testing samples: {X_test.shape[0]}")
