"""
Streamlit app to explore and compare different retrieval methods.

Usage:
    streamlit run retrieval_explorer.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_retriever import HybridRetriever
from utils.data_loader import HotpotQADataset


@st.cache_data
def load_data():
    """Load HotpotQA dataset."""
    if os.path.basename(os.getcwd()) == 'experiments':
        data_dir = "../data/raw"
    else:
        data_dir = "data/raw"

    dataset = HotpotQADataset(data_dir=data_dir)
    try:
        data = dataset.load_dev_distractor()
        return data, None
    except FileNotFoundError:
        return None, f"Dataset not found at {data_dir}. Please download it first."


@st.cache_data
def build_corpus(data, max_items):
    """Build corpus from HotpotQA data."""
    corpus = []
    doc_id = 0

    for item in data[:max_items]:
        for title, sentences in item['context']:
            for sent_id, sent in enumerate(sentences):
                corpus.append({
                    'id': doc_id,
                    'text': sent,
                    'title': title,
                    'sent_id': sent_id,
                    'source_question_id': item['_id']
                })
                doc_id += 1

    return corpus


@st.cache_resource
def initialize_retrievers(corpus, alpha):
    """Initialize all retrievers."""
    bm25 = BM25Retriever()
    bm25.index(corpus)

    dense = DenseRetriever()
    dense.index(corpus)

    hybrid = HybridRetriever(alpha=alpha)
    hybrid.index(corpus)

    return bm25, dense, hybrid


def display_document(doc, rank, score, is_supporting):
    """Display a single retrieved document."""
    if is_supporting:
        st.success(f"**✓ SUPPORTING FACT** - Rank #{rank} | Score: {score:.4f}")
    else:
        st.info(f"Rank #{rank} | Score: {score:.4f}")

    st.markdown(f"**Title:** {doc['title']}")
    st.markdown(f"**Sentence ID:** {doc.get('sent_id', 'N/A')}")
    st.markdown(f"**Text:** {doc['text']}")


def main():
    st.set_page_config(page_title="RAG Retrieval Explorer", layout="wide")

    st.title("🔍 RAG Retrieval Methods Explorer")
    st.markdown("Compare BM25, Dense, and Hybrid retrieval side-by-side")

    # Load data
    data, error = load_data()

    if error:
        st.error(error)
        st.code("wget -P data/raw http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")
        st.stop()

    st.success(f"✓ Loaded {len(data)} questions from HotpotQA")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    question_idx = st.sidebar.number_input(
        "Question Index",
        min_value=0,
        max_value=len(data)-1,
        value=0,
        help="Select which question to analyze"
    )

    corpus_size = st.sidebar.slider(
        "Corpus Size",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Number of questions to build corpus from"
    )

    top_k = st.sidebar.slider(
        "Top-K Results",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of top results to display"
    )

    alpha = st.sidebar.slider(
        "Hybrid Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Weight for dense retrieval (1-alpha for BM25)"
    )

    # Build corpus
    with st.spinner(f"Building corpus from {corpus_size} questions..."):
        corpus = build_corpus(data, corpus_size)

    st.sidebar.success(f"✓ Corpus: {len(corpus)} documents")

    # Initialize retrievers
    with st.spinner("Initializing retrievers (may take a moment on first run)..."):
        bm25, dense, hybrid = initialize_retrievers(corpus, alpha)

    st.sidebar.success("✓ Retrievers ready")

    # Display question
    question_data = data[question_idx]

    st.header("📋 Question Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Type", question_data['type'])
    with col2:
        st.metric("Level", question_data['level'])
    with col3:
        st.metric("Supporting Facts", len(question_data['supporting_facts']))

    st.markdown(f"**Question:** {question_data['question']}")
    st.markdown(f"**Answer:** `{question_data['answer']}`")

    with st.expander("View Supporting Facts"):
        for title, sent_id in question_data['supporting_facts']:
            st.markdown(f"- **[{title}]** Sentence {sent_id}")

    # Retrieve results
    question = question_data['question']
    support_set = set((title, sent_id) for title, sent_id in question_data['supporting_facts'])

    with st.spinner("Running retrieval..."):
        bm25_results = bm25.retrieve(question, top_k=top_k)
        dense_results = dense.retrieve(question, top_k=top_k)
        hybrid_results = hybrid.retrieve(question, top_k=top_k)

    # Summary metrics
    st.header("📊 Retrieval Performance Summary")

    def count_hits(results, k):
        """Count supporting facts in top-k."""
        hits = 0
        for doc, _ in results[:k]:
            if (doc['title'], doc['sent_id']) in support_set:
                hits += 1
        return hits

    total_supporting = len(support_set)

    metrics_data = []
    for k in [1, 3, 5, 10]:
        if k <= top_k:
            bm25_hits = count_hits(bm25_results, k)
            dense_hits = count_hits(dense_results, k)
            hybrid_hits = count_hits(hybrid_results, k)
            metrics_data.append({
                f"Top-{k}": f"BM25: {bm25_hits}/{total_supporting} | Dense: {dense_hits}/{total_supporting} | Hybrid: {hybrid_hits}/{total_supporting}"
            })

    for metric in metrics_data:
        for k, v in metric.items():
            st.markdown(f"**{k}:** {v}")

    # Display results side-by-side
    st.header("📑 Retrieved Documents Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔤 BM25 (Sparse)")
        st.caption("Lexical keyword matching")
        for i, (doc, score) in enumerate(bm25_results[:top_k], 1):
            is_supporting = (doc['title'], doc['sent_id']) in support_set
            with st.container():
                display_document(doc, i, score, is_supporting)
                st.divider()

    with col2:
        st.subheader("🧠 Dense (Neural)")
        st.caption("Semantic embedding similarity")
        for i, (doc, score) in enumerate(dense_results[:top_k], 1):
            is_supporting = (doc['title'], doc['sent_id']) in support_set
            with st.container():
                display_document(doc, i, score, is_supporting)
                st.divider()

    with col3:
        st.subheader(f"⚖️ Hybrid (α={alpha})")
        st.caption("Combined lexical + semantic")
        for i, (doc, score) in enumerate(hybrid_results[:top_k], 1):
            is_supporting = (doc['title'], doc['sent_id']) in support_set
            with st.container():
                display_document(doc, i, score, is_supporting)
                st.divider()

    # Analysis hints
    st.header("💡 Analysis Questions")

    with st.expander("Click to see analysis prompts"):
        st.markdown("""
        ### Questions to Consider:

        1. **Coverage**: Which retriever found the most supporting facts in top-5?
        2. **Precision**: Are there many irrelevant documents ranked highly?
        3. **Keyword Matching**: Does BM25 retrieve docs with obvious keyword overlap?
        4. **Semantic Understanding**: Does Dense retriever find relevant docs without exact keywords?
        5. **Complementarity**: Does Hybrid combine the best of both methods?
        6. **Failure Cases**: What supporting facts were missed by all retrievers? Why?

        ### Experiment Ideas:

        - Try different `alpha` values for Hybrid (sidebar)
        - Compare questions of different types (bridge vs comparison)
        - Look at questions where BM25 beats Dense (and vice versa)
        - Increase corpus size to see if retrieval improves
        """)

    # Quick navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Navigation")
    if st.sidebar.button("← Previous Question"):
        if question_idx > 0:
            st.rerun()
    if st.sidebar.button("Next Question →"):
        if question_idx < len(data) - 1:
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tips")
    st.sidebar.info("""
    - 🟢 Green = Supporting Fact
    - 🔵 Blue = Retrieved but not supporting
    - Adjust Top-K to see more results
    - Try different alpha values for Hybrid
    """)


if __name__ == "__main__":
    main()
