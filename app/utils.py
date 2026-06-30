# ─────────────────────────────────────────────────────────────
# app/utils.py
# Shared utility functions used by all prediction pages.
# Keeps the app modular — avoids repeating the same code
# in every page file (DRY principle).
# ─────────────────────────────────────────────────────────────

import os
import sys
import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HEART_MODEL_PATH, DIABETES_MODEL_PATH, CANCER_MODEL_PATH,
    MODELS_DIR
)


def load_model_and_scaler(disease_key):
    """
    Loads the trained model and its matching scaler for a given disease.

    Parameters:
        disease_key: 'heart', 'diabetes', or 'cancer'

    Returns:
        (model, scaler) tuple
    """
    model_path = os.path.join(MODELS_DIR, f"{disease_key}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{disease_key}_scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def make_prediction(model, scaler, input_values):
    """
    Takes raw user input, scales it the SAME way training data was scaled,
    then returns the model's prediction and confidence score.

    WHY WE MUST USE THE SAME SCALER:
    The model was trained on scaled data (mean=0, std=1). If we feed it
    raw unscaled values, predictions will be completely wrong — the model
    has never seen numbers in that original range during training.

    Parameters:
        model: trained classifier
        scaler: the SAME StandardScaler used during training
        input_values: list of raw feature values entered by the user

    Returns:
        prediction (0 or 1), confidence (float between 0 and 1)
    """
    # Convert to 2D array — sklearn expects shape (1, num_features) for a single sample
    input_array = np.array(input_values).reshape(1, -1)

    # Scale using the SAME scaler fitted during training
    input_scaled = scaler.transform(input_array)

    # Get prediction (0 or 1)
    prediction = model.predict(input_scaled)[0]

    # Get confidence score (probability of the predicted class)
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = probabilities[prediction]

    return prediction, confidence


def get_risk_level(confidence):
    """
    Converts a raw confidence score into a simple human-readable label.
    Used for friendly UI display.
    """
    if confidence >= 0.85:
        return "Very High Confidence"
    elif confidence >= 0.70:
        return "High Confidence"
    elif confidence >= 0.55:
        return "Moderate Confidence"
    else:
        return "Low Confidence"
