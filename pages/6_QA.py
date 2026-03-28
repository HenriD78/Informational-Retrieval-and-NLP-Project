"""
Page 6 — Extractive QA
Extractive question answering over reviews using DistilBERT-SQuAD.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="QA", page_icon="❓", layout="wide")
st.title("❓ Question Answering")
st.markdown("Ask a question about the insurance reviews. The system will extract direct answers from relevant reviews.")

OUTPUTS_DIR = Path("outputs")

@st.cache_data
def load_data():
    path = OUTPUTS_DIR / 'reviews_clean.csv'
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_qa_model():
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    model.eval()
    return tokenizer, model

def extractive_qa(question, context, tokenizer, model):
    import torch
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0]
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    # Find best answer span (skip [CLS] and question tokens)
    input_ids = inputs["input_ids"][0]
    sep_idx = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()
    start_logits[:sep_idx + 1] = -1e10
    end_logits[:sep_idx + 1] = -1e10
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()
    if end_idx < start_idx:
        end_idx = start_idx
    # Compute score via softmax
    start_prob = torch.softmax(start_logits, dim=0)[start_idx].item()
    end_prob = torch.softmax(end_logits, dim=0)[end_idx].item()
    score = (start_prob + end_prob) / 2
    # Map token positions back to character positions in the context
    char_start = offset_mapping[start_idx][0].item()
    char_end = offset_mapping[end_idx][1].item()
    # offset_mapping is relative to the full input; we need offset relative to context
    # The context starts after question + sep tokens, so find the char offset of the context
    answer = tokenizer.decode(input_ids[start_idx:end_idx + 1], skip_special_tokens=True)
    return {'answer': answer, 'score': score, 'start': char_start, 'end': char_end}

df = load_data()

if len(df) == 0:
    st.error("Data not found. Run Notebook 01 first.")
    st.stop()

# Settings
st.sidebar.header("QA Settings")
context_type = st.sidebar.radio("Context Source", ["Retrieve from reviews", "Manual context"])
n_retrieve = st.sidebar.slider("Number of reviews to search", 5, 50, 20)

filters_col1, filters_col2 = st.columns(2)

insurers = sorted(df['assureur'].dropna().unique().tolist()) if 'assureur' in df.columns else []
selected_insurer = filters_col1.selectbox("Filter by Insurer", ["All"] + insurers)

categories = sorted(df['category'].dropna().unique().tolist()) if 'category' in df.columns else []
selected_category = filters_col2.selectbox("Filter by Category", ["All"] + categories)

question = st.text_input(
    "Your Question",
    placeholder="e.g. How long does the claims process take? What do customers say about pricing?"
)

if context_type == "Manual context":
    manual_context = st.text_area("Context", placeholder="Paste text here for QA...", height=200)
else:
    manual_context = None

if st.button("❓ Find Answer", type="primary") and question.strip():

    if context_type == "Manual context" and manual_context:
        context = manual_context
        source_info = "manual context"
    else:
        # Filter reviews
        filtered = df.copy()
        if selected_insurer != "All" and 'assureur' in filtered.columns:
            filtered = filtered[filtered['assureur'] == selected_insurer]
        if selected_category != "All" and 'category' in filtered.columns:
            filtered = filtered[filtered['category'] == selected_category]

        # Keyword relevance
        keywords = [w.lower() for w in question.split() if len(w) > 3]
        if keywords:
            mask = pd.Series([False] * len(filtered))
            for kw in keywords:
                mask = mask | filtered['text'].str.lower().str.contains(kw, na=False)
            relevant = filtered[mask.values].head(n_retrieve)
        else:
            relevant = filtered.head(n_retrieve)

        if len(relevant) == 0:
            st.warning("No relevant reviews found.")
            st.stop()

        context = " ".join(relevant['text'].fillna('').tolist())[:3000]  # Limit context length
        source_info = f"{len(relevant)} retrieved reviews"

    st.subheader("🎯 Extracted Answers")

    with st.spinner("Loading QA model and extracting answers..."):
        tokenizer, model = load_qa_model()

        # Split context into chunks for multiple answers
        chunk_size = 500
        context_chunks = [context[i:i+chunk_size] for i in range(0, min(len(context), 3000), chunk_size)]

        answers = []
        for chunk in context_chunks[:6]:  # Process up to 6 chunks
            if len(chunk.strip()) > 50:
                try:
                    result = extractive_qa(question, chunk, tokenizer, model)
                    if result['score'] > 0.1:
                        answers.append(result)
                except Exception:
                    pass

        # Sort by confidence
        answers.sort(key=lambda x: x['score'], reverse=True)

    if answers:
        for i, ans in enumerate(answers[:3]):
            confidence = ans['score']
            answer_text = ans['answer']

            st.markdown(f"**Answer {i+1}** (confidence: {confidence:.2%})")
            st.markdown(f'> *"{answer_text}"*')

            # Show context around the answer
            start = ans['start']
            end = ans['end']
            if context_type != "Manual context":
                with st.expander("Show context"):
                    st.write(context[max(0, start-100):end+200])
            st.markdown("---")
    else:
        st.warning("Could not extract a confident answer. Try rephrasing the question.")

    st.caption(f"Answers extracted from: {source_info}")

    # Show retrieved reviews
    if context_type != "Manual context":
        st.subheader("📋 Source Reviews")
        display_cols = [c for c in ['text', 'sentiment', 'star_rating', 'category', 'assureur'] if c in relevant.columns]
        st.dataframe(relevant[display_cols].reset_index(drop=True), use_container_width=True)
