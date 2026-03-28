"""
Page 3 — Explanation
Text input → prediction + LIME word highlighting.
"""
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Explanation", page_icon="💡", layout="wide")
st.title("💡 Prediction Explanation")
st.markdown("Enter a review to get a prediction **with word-level LIME explanations**.")

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

@st.cache_resource
def load_model():
    tfidf_path = MODELS_DIR / 'tfidf_sentiment.pkl'
    logreg_path = MODELS_DIR / 'tfidf_logreg_sentiment.pkl'
    if tfidf_path.exists() and logreg_path.exists():
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        with open(logreg_path, 'rb') as f:
            logreg = pickle.load(f)
        return tfidf, logreg
    return None, None

tfidf, logreg = load_model()

if tfidf is None:
    st.error("Models not found. Please run Notebook 04 first.")
    st.stop()

text_input = st.text_area(
    "Insurance Review (English)",
    placeholder="Type or paste a review here...",
    height=150
)

if st.button("🔮 Explain Prediction", type="primary") and text_input.strip():
    from lime.lime_text import LimeTextExplainer

    class_names = logreg.classes_.tolist()

    def predict_proba(texts):
        X = tfidf.transform(texts)
        return logreg.predict_proba(X)

    pred = logreg.predict(tfidf.transform([text_input]))[0]
    proba = logreg.predict_proba(tfidf.transform([text_input]))[0]

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Sentiment", pred)

    sentiment_colors = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}
    for c, (cls, p) in zip([col1, col2, col3], zip(class_names, proba)):
        c.metric(f"{sentiment_colors.get(cls, '')} {cls.capitalize()}", f"{p:.1%}")

    st.subheader("LIME Word-Level Explanation")

    with st.spinner("Computing LIME explanation..."):
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(text_input, predict_proba, num_features=15, num_samples=500)

    # Display as colored text
    exp_list = exp.as_list()

    col_pos, col_neg = st.columns(2)

    positive_words = [(w, s) for w, s in exp_list if s > 0]
    negative_words = [(w, s) for w, s in exp_list if s < 0]

    col_pos.markdown("**Words supporting prediction:**")
    for word, score in sorted(positive_words, key=lambda x: -x[1]):
        col_pos.markdown(f"🟢 `{word}` → {score:+.4f}")

    col_neg.markdown("**Words opposing prediction:**")
    for word, score in sorted(negative_words, key=lambda x: x[1]):
        col_neg.markdown(f"🔴 `{word}` → {score:+.4f}")

    # Highlight text
    st.subheader("Highlighted Review")
    word_weights = dict(exp_list)

    words_in_text = text_input.split()
    highlighted_parts = []
    for word in words_in_text:
        clean_word = word.lower().strip('.,!?;:')
        if clean_word in word_weights:
            weight = word_weights[clean_word]
            if weight > 0:
                highlighted_parts.append(f'<span style="background-color: #c8f7c5; padding: 2px 4px; border-radius: 3px;">{word}</span>')
            else:
                highlighted_parts.append(f'<span style="background-color: #f7c8c8; padding: 2px 4px; border-radius: 3px;">{word}</span>')
        else:
            highlighted_parts.append(word)

    st.markdown(' '.join(highlighted_parts), unsafe_allow_html=True)

    # Save HTML
    exp.save_to_file(str(OUTPUTS_DIR / 'lime_explanation_latest.html'))

    with open(OUTPUTS_DIR / 'lime_explanation_latest.html', 'r') as f:
        html_content = f.read()
    st.download_button("Download Full LIME Report", html_content, "lime_explanation.html", "text/html")
