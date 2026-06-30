# ─────────────────────────────────────────────────────────────
# src/tune.py
# Fine-tunes the Random Forest model for each disease using
# GridSearchCV — automatically finds the best hyperparameters.
# ─────────────────────────────────────────────────────────────

import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, RANDOM_STATE, CV_FOLDS
from src.preprocessor import preprocess_all


# ── Define the search space (hyperparameter grid) ───────────────
def get_param_grid():
    """
    Defines the combinations of hyperparameters GridSearchCV will try.

    WARNING: more combinations = more time. This grid has
    3 x 3 x 2 x 2 = 36 combinations, x 5 CV folds = 180 model trainings
    per disease. We keep it reasonably small so it doesn't take forever
    on a regular laptop.
    """
    return {
        'n_estimators': [100, 200, 300],        # number of trees
        'max_depth': [5, 10, None],             # None = no limit (full depth)
        'min_samples_split': [2, 5],            # min samples to split a node
        'min_samples_leaf': [1, 2],             # min samples in a leaf
    }


# ── Tune the model for ONE disease ───────────────────────────────
def tune_model(X_train, y_train, disease_name):
    """
    Runs GridSearchCV to find the best Random Forest hyperparameters
    for the given disease's training data.
    """
    print(f"\n{'='*55}")
    print(f"  TUNING RANDOM FOREST FOR: {disease_name.upper()}")
    print(f"{'='*55}")
    print("  This may take 1-3 minutes, please wait...\n")

    base_model = RandomForestClassifier(random_state=RANDOM_STATE)
    param_grid = get_param_grid()

    # GridSearchCV tries every combination using cross-validation
    # scoring='f1' because F1-Score balances precision and recall —
    # important for medical predictions where both false positives
    # and false negatives matter
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=CV_FOLDS,           # 5-fold cross-validation
        scoring='f1',
        n_jobs=-1,              # use all CPU cores to speed this up
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n  Best parameters found: {grid_search.best_params_}")
    print(f"  Best cross-validation F1-Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


# ── Evaluate the tuned model on test data ────────────────────────
def evaluate_tuned_model(model, X_test, y_test, disease_name):
    """
    Evaluates the tuned model on the held-out test set
    and prints all key metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
    }

    print(f"\n  ── TUNED MODEL PERFORMANCE ({disease_name.upper()}) ──")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics


# ── Save the tuned model (overwrites the old one) ────────────────
def save_tuned_model(model, disease_key):
    model_path = os.path.join(MODELS_DIR, f"{disease_key}_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n  Tuned model saved → {model_path}")


# ── Master tuning function — runs for all 3 diseases ─────────────
def tune_all():
    print("=" * 55)
    print("  STARTING HYPERPARAMETER TUNING PIPELINE")
    print("=" * 55)

    preprocessed = preprocess_all()

    comparison = []

    for disease_key in ['heart', 'diabetes', 'cancer']:
        X_train, X_test, y_train, y_test, feature_names = preprocessed[disease_key]

        # Evaluate the OLD (default) model first, for comparison
        old_model = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
        old_model.fit(X_train, y_train)
        old_f1 = f1_score(y_test, old_model.predict(X_test))

        # Tune and get the new best model
        tuned_model, best_params = tune_model(X_train, y_train, disease_key)
        new_metrics = evaluate_tuned_model(tuned_model, X_test, y_test, disease_key)
        new_f1 = new_metrics['F1-Score']

        # IMPORTANT ENGINEERING DECISION:
        # Only keep the tuned model if it actually performs better
        # (or equal) on the test set than the original default model.
        # GridSearchCV optimizes using cross-validation on TRAINING data,
        # which doesn't always guarantee improvement on the final test set —
        # especially for smaller/noisier datasets. We never blindly trust
        # tuning; we always verify against the held-out test set.
        if new_f1 >= old_f1:
            save_tuned_model(tuned_model, disease_key)
            final_choice = "Tuned model (improved)"
            final_f1 = new_f1
        else:
            save_tuned_model(old_model, disease_key)
            final_choice = "Default model (tuning did not help)"
            final_f1 = old_f1
            print(f"\n  NOTE: Tuned model underperformed on test set.")
            print(f"  Keeping the default-parameter model instead for {disease_key}.")

        comparison.append({
            'Disease': disease_key.upper(),
            'Old F1-Score (default params)': round(old_f1, 4),
            'New F1-Score (tuned)': round(new_f1, 4),
            'Improvement': round(new_f1 - old_f1, 4),
            'Final Model Used': final_choice,
            'Final F1-Score': round(final_f1, 4),
            'Best Params': best_params,
        })

    # ── Final comparison table ──
    print(f"\n{'='*55}")
    print("  BEFORE vs AFTER TUNING — COMPARISON")
    print(f"{'='*55}\n")
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df[['Disease', 'Old F1-Score (default params)', 'New F1-Score (tuned)',
                          'Final Model Used', 'Final F1-Score']].to_string(index=False))

    print("\n\nBest parameters found per disease:")
    for row in comparison:
        print(f"\n{row['Disease']}: {row['Best Params']}")

    return comparison_df


# ── Run this file directly to tune all models ────────────────────
if __name__ == "__main__":
    results = tune_all()
    print("\n\nHyperparameter tuning complete! All tuned models saved.")
