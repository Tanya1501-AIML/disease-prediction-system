# 🏥 Disease Prediction System

A production-style **Multi-Disease Prediction System** built with Python and Machine Learning that predicts the likelihood of **Heart Disease**, **Diabetes**, and **Breast Cancer** from patient medical data.

Deployed as an interactive **Streamlit web application**.

---

## 🔗 Live Demo
> [Click here to try the app](#) ← *(link added after deployment)*

---

## 📌 Project Overview

| Disease | Dataset | Algorithm | Accuracy |
|---|---|---|---|
| Heart Disease | UCI Cleveland Heart Disease | Random Forest / XGBoost | TBD |
| Diabetes | Pima Indians Diabetes (Kaggle) | Random Forest / XGBoost | TBD |
| Breast Cancer | UCI Wisconsin Breast Cancer | SVM / Random Forest | TBD |

> Accuracy figures updated after model training phase.

---

## 🗂️ Project Structure

```
disease_prediction/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned & preprocessed data
├── notebooks/            # EDA Jupyter notebooks
├── src/
│   ├── data_loader.py    # Dataset loading utilities
│   ├── preprocessor.py   # Feature engineering & scaling
│   ├── train.py          # Model training pipeline
│   ├── evaluate.py       # Evaluation metrics & plots
│   └── predict.py        # Prediction interface
├── models/               # Saved trained models (.pkl)
├── app/
│   └── app.py            # Streamlit web application
├── reports/              # Evaluation plots & reports
├── config.py             # Central configuration
└── requirements.txt      # Project dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/disease-prediction.git
cd disease-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the web application
```bash
streamlit run app/app.py
```

---

## 🧠 ML Pipeline

```
Raw Data → EDA → Preprocessing → Model Training → Evaluation → Hyperparameter Tuning → Deployment
```

**Algorithms used:** Logistic Regression, Random Forest, Support Vector Machine (SVM), XGBoost, K-Nearest Neighbors

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Techniques:** Feature scaling (StandardScaler), Train-Test Split (80/20), Cross-Validation (5-fold), GridSearchCV

---

## 📊 Results

> *(Evaluation plots and confusion matrices will be added after model training)*

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML Libraries:** scikit-learn, XGBoost
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Streamlit
- **Model Persistence:** Joblib

---

## 👩‍💻 Author

**Sumi**
MBA (AI/ML) — Manipal University Jaipur

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
