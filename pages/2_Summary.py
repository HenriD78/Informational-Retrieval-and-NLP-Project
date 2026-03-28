"""
Page 2 — Summary
Aggregate summaries by insurer.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Summary", page_icon="📊", layout="wide")
st.title("📊 Insurer Summary")
st.markdown("Select an insurer to see aggregate statistics and a summary of reviews.")

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
all_insurers = sorted(df['assureur'].dropna().unique().tolist()) if 'assureur' in df.columns else []
selected_insurer = st.sidebar.selectbox("Select Insurer", ["All"] + all_insurers)

if selected_insurer != "All":
    view_df = df[df['assureur'] == selected_insurer].copy()
else:
    view_df = df.copy()

# Top-level metrics
st.subheader(f"Overview: {selected_insurer}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", len(view_df))
col2.metric("Avg Rating", f"{view_df['note'].mean():.2f} ⭐" if 'note' in view_df.columns else "N/A")
col3.metric("% Positive", f"{(view_df['sentiment']=='positive').mean()*100:.1f}%" if 'sentiment' in view_df.columns else "N/A")
col4.metric("% Negative", f"{(view_df['sentiment']=='negative').mean()*100:.1f}%" if 'sentiment' in view_df.columns else "N/A")

# Charts
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

if 'sentiment' in view_df.columns:
    view_df['sentiment'].value_counts().plot(kind='bar', ax=axes[0],
        color=['green', 'gray', 'red'], title='Sentiment Distribution')
    axes[0].tick_params(rotation=0)

if 'star_rating' in view_df.columns:
    view_df['star_rating'].value_counts().sort_index().plot(kind='bar', ax=axes[1],
        color='steelblue', title='Star Rating Distribution')
    axes[1].tick_params(rotation=0)

if 'category' in view_df.columns:
    view_df['category'].value_counts().plot(kind='barh', ax=axes[2],
        color='purple', title='Category Distribution')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# NLP Summary: most frequent words per sentiment
st.subheader("Most Frequent Words by Sentiment")

from collections import Counter
import re

def get_top_words(texts, n=15):
    words = []
    for t in texts:
        words.extend(re.findall(r'\b[a-z]{3,}\b', str(t).lower()))
    stop = {'the','and','was','for','with','this','that','they','have','has','are','our','not','but','you',
            'your','all','very','been','from','their','will','its','were','had','would','more','just','also'}
    words = [w for w in words if w not in stop]
    return Counter(words).most_common(n)

if 'sentiment' in view_df.columns:
    cols = st.columns(3)
    for col, sent in zip(cols, ['positive', 'neutral', 'negative']):
        sent_texts = view_df[view_df['sentiment'] == sent]['text'].tolist()
        if sent_texts:
            top = get_top_words(sent_texts)
            words_df = pd.DataFrame(top, columns=['Word', 'Count'])
            color = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}[sent]
            col.markdown(f"**{color} {sent.capitalize()} Reviews**")
            col.dataframe(words_df, use_container_width=True, hide_index=True)

# Sample reviews
st.subheader("Sample Reviews")
sample = view_df[['text', 'sentiment', 'star_rating', 'category']].sample(
    min(10, len(view_df)), random_state=42
)
st.dataframe(sample, use_container_width=True, hide_index=True)
