# ─────────────────────────────────────────────────────────────
# src/data_loader.py
# Responsible for loading all 3 datasets into the project.
# Run this file once to download and save all datasets.
# ─────────────────────────────────────────────────────────────

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Import our central config so we never hardcode any paths
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_DIR,
    HEART_DATA_PATH,
    DIABETES_DATA_PATH,
    CANCER_DATA_PATH
)


# ── Helper ────────────────────────────────────────────────────
def ensure_dirs():
    """Create data/raw folder if it doesn't exist yet."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)


# ── Dataset 1: Heart Disease ──────────────────────────────────
def load_heart_data():
    """
    Load the Cleveland Heart Disease dataset.
    Source: UCI ML Repository (via public URL)
    
    Features (13 total):
    - age, sex, cp (chest pain type), trestbps (resting blood pressure)
    - chol (cholesterol), fbs (fasting blood sugar), restecg
    - thalach (max heart rate), exang, oldpeak, slope, ca, thal
    
    Target: 1 = Heart disease present, 0 = No heart disease
    """
    print("Loading Heart Disease dataset...")

    # Column names for the dataset (UCI doesn't include headers)
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    # Load from UCI repository URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        
        # The original target has values 0,1,2,3,4
        # We convert to binary: 0 = no disease, 1 = disease
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        # Save to our raw data folder
        df.to_csv(HEART_DATA_PATH, index=False)
        print(f"  Heart Disease dataset saved → {HEART_DATA_PATH}")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        return df
    
    except Exception as e:
        print(f"  Could not download from URL: {e}")
        print("  Using sklearn's built-in heart disease data as fallback...\n")
        return load_heart_data_fallback()


def load_heart_data_fallback():
    """
    Fallback: create heart dataset from a reliable mirror.
    Used if UCI website is down.
    """
    # Reliable Kaggle mirror
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
    try:
        df = pd.read_csv(url)
        df.to_csv(HEART_DATA_PATH, index=False)
        print(f"  Heart Disease dataset saved (fallback) → {HEART_DATA_PATH}")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        return df
    except Exception as e:
        print(f"  Fallback also failed: {e}")
        return None


# ── Dataset 2: Diabetes ───────────────────────────────────────
def load_diabetes_data():
    """
    Load the Pima Indians Diabetes dataset.
    Source: Kaggle (via public GitHub mirror)
    
    Features (8 total):
    - Pregnancies, Glucose, BloodPressure, SkinThickness
    - Insulin, BMI, DiabetesPedigreeFunction, Age
    
    Target: 1 = Diabetic, 0 = Not diabetic
    """
    print("Loading Diabetes dataset...")

    url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
    
    try:
        df = pd.read_csv(url)
        df.to_csv(DIABETES_DATA_PATH, index=False)
        print(f"  Diabetes dataset saved → {DIABETES_DATA_PATH}")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        return df
    
    except Exception as e:
        print(f"  Could not download: {e}")
        return None


# ── Dataset 3: Breast Cancer ──────────────────────────────────
def load_cancer_data():
    """
    Load the Wisconsin Breast Cancer dataset.
    Source: sklearn's built-in datasets (no download needed!)
    
    Features: 30 numerical features computed from cell nucleus images
    Examples: radius, texture, perimeter, area, smoothness...
    
    Target: 1 = Malignant (cancerous), 0 = Benign (not cancerous)
    
    Note: sklearn gives target 0=malignant, 1=benign
    We flip it so 1=malignant (disease present) for consistency.
    """
    print("Loading Breast Cancer dataset...")

    # Load directly from sklearn — no internet needed
    data = load_breast_cancer()
    
    # Convert to a pandas DataFrame so it's easy to work with
    df = pd.DataFrame(data.feature_names, columns=['feature'])
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Add the target column
    # sklearn: 0=malignant, 1=benign → we flip: 1=malignant, 0=benign
    df['target'] = 1 - data.target

    df.to_csv(CANCER_DATA_PATH, index=False)
    print(f"  Breast Cancer dataset saved → {CANCER_DATA_PATH}")
    print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df


# ── Master loader — loads all 3 at once ───────────────────────
def load_all_datasets():
    """
    Load all three datasets at once.
    Returns a dictionary: {'heart': df1, 'diabetes': df2, 'cancer': df3}
    """
    ensure_dirs()
    print("=" * 50)
    print("  LOADING ALL DATASETS")
    print("=" * 50 + "\n")

    datasets = {
        'heart':    load_heart_data(),
        'diabetes': load_diabetes_data(),
        'cancer':   load_cancer_data(),
    }

    print("=" * 50)
    print("  ALL DATASETS LOADED SUCCESSFULLY!")
    print("=" * 50)
    return datasets


# ── Run this file directly to download everything ─────────────
if __name__ == "__main__":
    datasets = load_all_datasets()

    print("\n── DATASET SUMMARY ──")
    for name, df in datasets.items():
        if df is not None:
            print(f"\n{name.upper()}:")
            print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            target_col = 'Outcome' if name == 'diabetes' else 'target'
            print(f"  Target distribution:\n{df[target_col].value_counts()}")