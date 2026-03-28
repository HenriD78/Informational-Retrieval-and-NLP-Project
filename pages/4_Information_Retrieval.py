"""
Page 4 — Information Retrieval
Search bar + filters (insurer, rating, category).
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Information Retrieval", page_icon="🔍", layout="wide")
st.title("🔍 Information Retrieval")
st.markdown("Search reviews by keyword with optional filters.")

@st.cache_data
def load_data():
    path = Path("outputs/reviews_clean.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

df = load_data()

if len(df) == 0:
    st.error("Data not found. Run Notebook 01 first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

query = st.text_input("🔍 Search Query", placeholder="e.g. claim denied, excellent service, too expensive...")

col1, col2, col3 = st.columns(3)

insurers = sorted(df['assureur'].dropna().unique().tolist()) if 'assureur' in df.columns else []
selected_insurer = col1.selectbox("Insurer", ["All"] + insurers)

ratings = sorted(df['star_rating'].dropna().unique().tolist()) if 'star_rating' in df.columns else []
selected_ratings = col2.multiselect("Star Rating", ratings, default=ratings)

categories = sorted(df['category'].dropna().unique().tolist()) if 'category' in df.columns else []
selected_category = col3.selectbox("Category", ["All"] + categories)

sentiments = ['All', 'positive', 'neutral', 'negative']
selected_sentiment = col1.selectbox("Sentiment", sentiments)

max_results = st.sidebar.slider("Max Results", 10, 200, 50)

search_btn = st.button("🔍 Search", type="primary")

if search_btn:
    results = df.copy()

    # Apply filters
    if selected_insurer != "All" and 'assureur' in results.columns:
        results = results[results['assureur'] == selected_insurer]

    if selected_ratings and 'star_rating' in results.columns:
        results = results[results['star_rating'].isin(selected_ratings)]

    if selected_category != "All" and 'category' in results.columns:
        results = results[results['category'] == selected_category]

    if selected_sentiment != "All" and 'sentiment' in results.columns:
        results = results[results['sentiment'] == selected_sentiment]

    # Keyword search
    if query.strip():
        mask = results['text'].str.contains(query, case=False, na=False)
        results = results[mask]

    results = results.head(max_results)

    st.markdown(f"**Found {len(results)} reviews**")

    if len(results) > 0:
        display_cols = [c for c in ['text', 'sentiment', 'star_rating', 'category', 'assureur', 'produit'] if c in results.columns]
        st.dataframe(results[display_cols].reset_index(drop=True), use_container_width=True)

        # Download
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv, "search_results.csv", "text/csv")

        # Stats on results
        if len(results) > 1:
            st.subheader("Results Statistics")
            scol1, scol2 = st.columns(2)
            if 'sentiment' in results.columns:
                scol1.write("Sentiment distribution:")
                scol1.bar_chart(results['sentiment'].value_counts())
            if 'star_rating' in results.columns:
                scol2.write("Star rating distribution:")
                scol2.bar_chart(results['star_rating'].value_counts().sort_index())
    else:
        st.warning("No reviews found matching your criteria.")

# Semantic search (if FAISS index available)
st.markdown("---")
st.subheader("🧠 Semantic Search (FAISS + Sentence Transformers)")

FAISS_PATH = Path("outputs/faiss_index.bin")
EMBEDDINGS_PATH = Path("outputs/sentence_embeddings.npy")

if FAISS_PATH.exists() and EMBEDDINGS_PATH.exists():
    semantic_query = st.text_input("Semantic Search Query", placeholder="e.g. insurance company refused to pay...")
    k_results = st.slider("Number of results", 3, 20, 5)

    if st.button("🧠 Semantic Search") and semantic_query.strip():
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        @st.cache_resource
        def load_faiss():
            index = faiss.read_index(str(FAISS_PATH))
            embeddings = np.load(str(EMBEDDINGS_PATH))
            return index, embeddings

        @st.cache_resource
        def load_st_model():
            return SentenceTransformer('all-MiniLM-L6-v2')

        with st.spinner("Searching..."):
            index, _ = load_faiss()
            st_model = load_st_model()
            q_emb = st_model.encode([semantic_query]).astype(np.float32)
            distances, indices = index.search(q_emb, k_results)

        st.markdown(f"**Top {k_results} semantic matches:**")
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(df):
                row = df.iloc[idx]
                with st.expander(f"#{rank+1} — Distance: {dist:.4f} | {row.get('sentiment','?')} | ⭐{row.get('star_rating','?')}"):
                    st.write(row.get('text', ''))
else:
    st.info("FAISS index not found. Run Notebook 03 to enable semantic search.")
