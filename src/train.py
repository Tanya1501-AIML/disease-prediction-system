# ─────────────────────────────────────────────────────────────
# src/train.py
# Trains multiple ML algorithms for each disease and compares
# their performance to select the best model.
# ─────────────────────────────────────────────────────────────

import os
import sys
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, RANDOM_STATE
from src.preprocessor import preprocess_all


# ── Define the models we want to try ────────────────────────────
def get_models():
    """
    Returns a dictionary of model name -> model object.
    Each model is initialized with sensible default parameters.
    We'll fine-tune the BEST one later in Milestone 5 (Hyperparameter Tuning).
    """
    return {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000   # increased so the model has enough iterations to converge
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100   # number of decision trees in the forest
        ),
        'SVM': SVC(
            random_state=RANDOM_STATE,
            probability=True   # needed to get confidence scores later
        ),
        'XGBoost': XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric='logloss'
        ),
    }


# ── Train and evaluate all models for ONE disease ───────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, disease_name):
    """
    Trains all 4 models on the given data, evaluates each,
    and returns a results table + the best trained model.
    """
    print(f"\n{'='*55}")
    print(f"  TRAINING MODELS FOR: {disease_name.upper()}")
    print(f"{'='*55}")

    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Train the model on training data
        model.fit(X_train, y_train)

        # Predict on the UNSEEN test data
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]   # probability of class 1

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4),
        })

        trained_models[name] = model
        print(f"    Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    # Convert results into a clean comparison table
    results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False).reset_index(drop=True)

    print(f"\n  ── RESULTS TABLE ({disease_name.upper()}) ──")
    print(results_df.to_string(index=False))

    # Pick the best model based on F1-Score
    # WHY F1-SCORE and not just accuracy? In medical data, F1 balances
    # Precision (avoiding false alarms) and Recall (not missing real cases) —
    # both matter a lot when predicting diseases.
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]

    print(f"\n  BEST MODEL: {best_model_name}")

    return results_df, best_model, best_model_name


# ── Save a trained model to disk ─────────────────────────────────
def save_model(model, disease_key):
    """
    Saves the trained model using joblib so it can be
    loaded later without retraining (used by the Streamlit app).
    """
    model_path = os.path.join(MODELS_DIR, f"{disease_key}_model.pkl")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"  Model saved → {model_path}")


# ── Master training function ─────────────────────────────────────
def train_all():
    """
    Runs the full training pipeline for all 3 diseases:
    preprocess -> train multiple models -> pick best -> save it.
    """
    print("=" * 55)
    print("  STARTING FULL MODEL TRAINING PIPELINE")
    print("=" * 55)

    # Get preprocessed data for all diseases
    preprocessed = preprocess_all()

    all_results = {}

    disease_keys = {
        'heart': 'heart',
        'diabetes': 'diabetes',
        'cancer': 'cancer',
    }

    for disease_key, display_name in disease_keys.items():
        X_train, X_test, y_train, y_test, feature_names = preprocessed[disease_key]

        results_df, best_model, best_model_name = train_and_evaluate(
            X_train, X_test, y_train, y_test, display_name
        )

        # Save the best model for this disease
        save_model(best_model, disease_key)

        all_results[disease_key] = {
            'results_table': results_df,
            'best_model_name': best_model_name,
            'feature_names': feature_names,
        }

    # ── Final summary across all diseases ──
    print(f"\n{'='*55}")
    print("  FINAL SUMMARY — BEST MODEL PER DISEASE")
    print(f"{'='*55}")
    for disease_key, info in all_results.items():
        best_row = info['results_table'].iloc[0]
        print(f"\n{disease_key.upper()}:")
        print(f"  Best Model: {info['best_model_name']}")
        print(f"  Accuracy: {best_row['Accuracy']} | F1-Score: {best_row['F1-Score']} | ROC-AUC: {best_row['ROC-AUC']}")

    return all_results


# ── Run this file directly to train everything ──────────────────
if __name__ == "__main__":
    results = train_all()
    print("\n\nTraining complete! All best models saved in the 'models' folder.")
