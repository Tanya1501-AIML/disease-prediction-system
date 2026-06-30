# ─────────────────────────────────────────────────────────────
# app/Home.py
# Main landing page of the Multi-Disease Prediction System.
# Run with: streamlit run app/Home.py
# ─────────────────────────────────────────────────────────────

import streamlit as st

# Page configuration — must be the FIRST streamlit command in the file
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# ── Custom CSS — clinical / lab-report visual identity ─────────
# WHY: Streamlit's default theme looks the same on every student
# project. Custom CSS makes the app instantly recognizable and
# professional — a real differentiator for a portfolio piece.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }

    .main {
        background-color: #F7F5F1;
    }

    /* Hero section */
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        color: #2D7D6E;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: 'Lora', serif;
        font-size: 3.1rem;
        font-weight: 700;
        color: #0B1F2A;
        line-height: 1.1;
        margin-bottom: 0.8rem;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #5A6470;
        max-width: 620px;
        line-height: 1.55;
        margin-bottom: 0;
    }
    .hero-pipeline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #8B95A1;
        margin-top: 1.4rem;
        letter-spacing: 0.02em;
    }

    /* Divider */
    .thin-rule {
        border: none;
        border-top: 1px solid #DDD8CD;
        margin: 2.4rem 0 2rem 0;
    }

    /* Chart cards */
    .chart-card {
        background: #0B1F2A;
        border-radius: 4px;
        padding: 1.7rem 1.6rem 1.5rem 1.6rem;
        height: 100%;
        border-left: 3px solid #2D7D6E;
    }
    .chart-card.alert {
        border-left: 3px solid #C1502E;
    }
    .chart-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #7E8B96;
        margin-bottom: 0.3rem;
    }
    .chart-name {
        font-family: 'Lora', serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #F7F5F1;
        margin-bottom: 0.9rem;
    }
    .chart-desc {
        font-size: 0.87rem;
        color: #A8B0B8;
        line-height: 1.5;
        margin-bottom: 1.2rem;
        min-height: 65px;
    }
    .chart-vital {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #5FBFA9;
        line-height: 1;
    }
    .chart-vital-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #6B7680;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.35rem;
    }

    /* Section heading */
    .section-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.16em;
        color: #2D7D6E;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .section-title {
        font-family: 'Lora', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #0B1F2A;
        margin-bottom: 1.4rem;
    }

    /* Tech stack pills */
    .pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #0B1F2A;
        background: #EAE6DB;
        border: 1px solid #DDD8CD;
        padding: 0.32rem 0.85rem;
        border-radius: 3px;
        margin: 0 0.5rem 0.5rem 0;
    }

    /* Footer */
    .footer-text {
        font-size: 0.85rem;
        color: #8B95A1;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown('<div class="hero-eyebrow">Machine Learning · Clinical Risk Modeling</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Multi-Disease<br>Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">An end-to-end Machine Learning pipeline that screens for '
    'Heart Disease, Diabetes, and Breast Cancer from patient clinical data — '
    'trained, tuned, and validated on real medical datasets.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-pipeline">DATA &nbsp;→&nbsp; EDA &nbsp;→&nbsp; PREPROCESSING &nbsp;→&nbsp; '
    'MODEL TRAINING &nbsp;→&nbsp; TUNING &nbsp;→&nbsp; DEPLOYMENT</div>',
    unsafe_allow_html=True
)

st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)

# ── Section: prediction modules ─────────────────────────────────
st.markdown('<div class="section-eyebrow">Select a module</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Choose a prediction tool from the sidebar</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-label">Module 01 · Cardiology</div>
        <div class="chart-name">❤️ Heart Disease</div>
        <div class="chart-desc">Predicts heart disease likelihood from 13 clinical
        features — cholesterol, resting blood pressure, chest pain type, and max
        heart rate among them.</div>
        <div class="chart-vital">88.5%</div>
        <div class="chart-vital-label">Test Accuracy · Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="chart-card alert">
        <div class="chart-label">Module 02 · Endocrinology</div>
        <div class="chart-name">🩸 Diabetes</div>
        <div class="chart-desc">Predicts diabetes risk from 8 health indicators
        including glucose level, BMI, insulin, and family history (pedigree
        function).</div>
        <div class="chart-vital">77.9%</div>
        <div class="chart-vital-label">Test Accuracy · Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-label">Module 03 · Oncology</div>
        <div class="chart-name">🎗️ Breast Cancer</div>
        <div class="chart-desc">Predicts malignancy from 30 cell nucleus
        measurements — radius, texture, concavity, and symmetry computed from
        biopsy imaging.</div>
        <div class="chart-vital">97.4%</div>
        <div class="chart-vital-label">Test Accuracy · Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)

# ── Tech stack ────────────────────────────────────────────────
st.markdown('<div class="section-eyebrow">Built with</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tech Stack</div>', unsafe_allow_html=True)

st.markdown("""
<span class="pill">Python</span>
<span class="pill">scikit-learn</span>
<span class="pill">XGBoost</span>
<span class="pill">Pandas / NumPy</span>
<span class="pill">Streamlit</span>
<span class="pill">Matplotlib / Seaborn</span>
<span class="pill">Joblib</span>
""", unsafe_allow_html=True)

st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────
st.warning(
    "⚠️ **Disclaimer:** This tool is built for educational and portfolio "
    "purposes only. It is NOT a substitute for professional medical advice, "
    "diagnosis, or treatment. Always consult a qualified healthcare provider."
)

# ── Footer ────────────────────────────────────────────────────
st.markdown('<div class="footer-text">Made by <strong>Tanya Sharma</strong></div>', unsafe_allow_html=True)

