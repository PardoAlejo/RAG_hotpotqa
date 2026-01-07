"""
Example RAG pipeline using HotpotQA dataset.

This script demonstrates a complete RAG pipeline:
1. Load data
2. Build corpus from context
3. Retrieve documents
4. (Optional) Rerank
5. Generate answers with LLM
6. Evaluate results
"""

import sys
sys.path.append('..')

from utils.data_loader import HotpotQADataset
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_retriever import HybridRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from llm_integration.simple_concatenation import SimpleConcatenationStrategy
from evaluation.hotpot_metrics import HotpotQAEvaluator


def build_corpus_from_hotpot(data, max_items=None):
    """
    Build a corpus from HotpotQA data.

    Args:
        data: HotpotQA dataset
        max_items: Maximum number of items to process (None for all)

    Returns:
        List of document dictionaries
    """
    corpus = []
    doc_id = 0

    if max_items:
        data = data[:max_items]

    for item in data:
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


def main():
    """Run example RAG pipeline."""

    print("=" * 70)
    print("RAG Pipeline Example with HotpotQA")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading HotpotQA dataset...")
    dataset = HotpotQADataset(data_dir="../data/raw")

    try:
        dev_data = dataset.load_dev_distractor()
        print(f"Loaded {len(dev_data)} examples")

        # Get statistics
        stats = dataset.get_statistics(dev_data)
        print(f"\nDataset statistics:")
        print(f"  Total: {stats['total']}")
        print(f"  Types: {stats['types']}")
        print(f"  Levels: {stats['levels']}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download the dataset first:")
        print("wget -P ../data/raw http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")
        return

    # 2. Build corpus
    print("\n[2/6] Building corpus from context...")
    # Use first 100 examples for demo (remove limit for full dataset)
    corpus = build_corpus_from_hotpot(dev_data, max_items=100)
    print(f"Built corpus with {len(corpus)} documents")

    # 3. Initialize retriever
    print("\n[3/6] Initializing retriever...")
    print("Options: BM25Retriever, DenseRetriever, HybridRetriever")
    print("Using: BM25Retriever (fastest for demo)")

    retriever = BM25Retriever()
    retriever.index(corpus)
    print("Indexing complete!")

    # 4. Retrieve documents for a sample question
    print("\n[4/6] Testing retrieval...")
    sample = dev_data[0]
    question = sample['question']
    print(f"\nQuestion: {question}")
    print(f"Gold Answer: {sample['answer']}")

    retrieved_docs = retriever.retrieve(question, top_k=5)
    print(f"\nRetrieved {len(retrieved_docs)} documents:")
    for i, (doc, score) in enumerate(retrieved_docs):
        print(f"\n  [{i+1}] Score: {score:.4f}")
        print(f"      Title: {doc['title']}")
        print(f"      Text: {doc['text'][:100]}...")

    # 5. Optional: Rerank
    print("\n[5/6] Re-ranking (optional)...")
    print("Skipping reranking in this demo (requires additional models)")
    print("To enable: Uncomment the reranking code below")

    # Uncomment to use reranking:
    # reranker = CrossEncoderReranker()
    # reranked_docs = reranker.rerank(question, retrieved_docs, top_k=3)
    # print(f"Reranked top 3 documents")

    # 6. Evaluate (simple example)
    print("\n[6/6] Evaluation example...")
    evaluator = HotpotQAEvaluator()

    # Example evaluation (you would replace predicted answer with LLM output)
    # For demo, we'll use the gold answer to show perfect scores
    metrics = evaluator.evaluate_example(
        answer_pred=sample['answer'],
        answer_gold=sample['answer'],
        sp_pred=sample['supporting_facts'],
        sp_gold=sample['supporting_facts']
    )

    evaluator.print_metrics(metrics)

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Implement LLM client in llm_integration/ strategies")
    print("2. Generate actual answers using retrieved context")
    print("3. Evaluate on full dev set")
    print("4. Experiment with different retrieval/reranking methods")
    print("5. Try different LLM integration strategies")


if __name__ == "__main__":
    main()
