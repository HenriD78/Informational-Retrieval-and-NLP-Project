"""
NLP Project 2 — Insurance Reviews Analysis
Multi-page Streamlit application entry point.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Insurance Reviews NLP",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    path = Path("outputs/reviews_clean.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def main():
    st.title("🔍 Insurance Reviews NLP Dashboard")
    st.markdown("""
    Welcome to the **Insurance Reviews NLP Dashboard** — an interactive tool to explore, analyze,
    and understand French insurance customer reviews (translated to English).

    ### Available Pages

    | Page | Description |
    |------|-------------|
    | **1 — Prediction** | Enter a review to predict star rating, sentiment, and category |
    | **2 — Summary** | Aggregate statistics and summaries by insurer |
    | **3 — Explanation** | Prediction with LIME/SHAP word-level explanations |
    | **4 — Information Retrieval** | Search reviews by keyword and filters |
    | **5 — RAG** | Ask questions answered using relevant reviews |
    | **6 — QA** | Extractive question answering over reviews |

    ### Dataset Overview
    """)

    df = load_data()
    if len(df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{len(df):,}")
        col2.metric("Insurance Companies", df['assureur'].nunique() if 'assureur' in df.columns else "N/A")
        col3.metric("Product Types", df['produit'].nunique() if 'produit' in df.columns else "N/A")
        col4.metric("Avg Rating", f"{df['note'].mean():.2f} ⭐" if 'note' in df.columns else "N/A")

        st.subheader("Sample Reviews")
        st.dataframe(df[['text', 'sentiment', 'star_rating', 'category']].head(10), use_container_width=True)
    else:
        st.warning("Data not found. Please run Notebook 01 first to generate `outputs/reviews_clean.csv`.")

    st.sidebar.success("Select a page above to get started.")

if __name__ == "__main__":
    main()
