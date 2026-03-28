"""
Page 1 — Prediction
User inputs a review → predict star rating, sentiment, and category.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Prediction", page_icon="⭐", layout="wide")
st.title("⭐ Review Prediction")
st.markdown("Enter an insurance review to predict its **star rating**, **sentiment**, and **category**.")

MODELS_DIR = Path("models")

@st.cache_resource
def load_models():
    models = {}
    tasks = ['sentiment', 'star_rating', 'category']
    for task in tasks:
        tfidf_path = MODELS_DIR / f'tfidf_{task}.pkl'
        logreg_path = MODELS_DIR / f'tfidf_logreg_{task}.pkl'
        if tfidf_path.exists() and logreg_path.exists():
            with open(tfidf_path, 'rb') as f:
                tfidf = pickle.load(f)
            with open(logreg_path, 'rb') as f:
                logreg = pickle.load(f)
            models[task] = (tfidf, logreg)
    return models

models = load_models()

if not models:
    st.error("Models not found. Please run Notebook 04 (`04_supervised_tfidf_classical.ipynb`) first.")
    st.stop()

# Input
text_input = st.text_area(
    "Insurance Review (in English)",
    placeholder="e.g. The customer service was excellent and my claim was processed quickly.",
    height=150
)

col1, col2 = st.columns([1, 4])
predict_btn = col1.button("🔮 Predict", type="primary")
clear_btn = col2.button("Clear")

if predict_btn and text_input.strip():
    st.markdown("---")
    st.subheader("Prediction Results")

    results_col1, results_col2, results_col3 = st.columns(3)

    task_labels = {
        'sentiment': ('😊 Sentiment', {'positive': '🟢 Positive', 'neutral': '🟡 Neutral', 'negative': '🔴 Negative'}),
        'star_rating': ('⭐ Star Rating', {}),
        'category': ('📁 Category', {}),
    }

    for (task, (label, emoji_map)), col in zip(task_labels.items(), [results_col1, results_col2, results_col3]):
        if task in models:
            tfidf, logreg = models[task]
            X = tfidf.transform([text_input])
            pred = logreg.predict(X)[0]
            proba = logreg.predict_proba(X)[0] if hasattr(logreg, 'predict_proba') else None

            display_pred = emoji_map.get(str(pred), str(pred))
            col.metric(label, display_pred)

            if proba is not None:
                conf_df = pd.DataFrame({
                    'Class': logreg.classes_,
                    'Confidence': proba.round(3)
                }).sort_values('Confidence', ascending=False)
                col.dataframe(conf_df, use_container_width=True, hide_index=True)
        else:
            col.warning(f"No model for {task}")
elif predict_btn:
    st.warning("Please enter a review text.")

# Example reviews
st.markdown("---")
st.subheader("Try Example Reviews")
examples = [
    "The insurance company was absolutely terrible. My claim took 6 months and was eventually denied without explanation.",
    "Pretty average experience. Nothing stood out as particularly good or bad.",
    "Excellent service! My claim was approved within 3 days and everyone I spoke to was very helpful.",
    "The premium is way too expensive for the coverage provided. Looking to switch.",
    "Enrolling in the plan was very confusing and the website kept crashing.",
]

for i, ex in enumerate(examples):
    if st.button(f"Example {i+1}", key=f"ex_{i}"):
        st.session_state['example_text'] = ex

if 'example_text' in st.session_state:
    st.text_area("Selected example:", value=st.session_state['example_text'], height=100, disabled=True)
