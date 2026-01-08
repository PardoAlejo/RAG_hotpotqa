"""
Evaluation script for retrieval methods.

Usage:
    python evaluate_retrieval.py --retriever bm25 --num_questions 100
    python evaluate_retrieval.py --retriever dense --top_k 10
    python evaluate_retrieval.py --retriever hybrid --alpha 0.7 --num_questions 50
"""

import sys
import os
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.model_configs import (
    get_full_model_name,
    list_available_models,
    DEFAULT_MODEL
)
from utils.data_loader import HotpotQADataset
from tqdm import tqdm


def build_corpus(data, max_items=None):
    """Build corpus from HotpotQA data."""
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


def calculate_metrics(results, supporting_facts, k_values=[1, 3, 5, 10, 20]):
    """Calculate retrieval metrics."""
    support_set = set((title, sent_id) for title, sent_id in supporting_facts)
    total_supporting = len(support_set)

    metrics = {}

    for k in k_values:
        if k > len(results):
            continue

        # Count hits in top-k
        hits = 0
        for doc, _ in results[:k]:
            if (doc['title'], doc['sent_id']) in support_set:
                hits += 1

        # Recall@K
        recall = hits / total_supporting if total_supporting > 0 else 0

        # Precision@K
        precision = hits / k if k > 0 else 0

        metrics[f'recall@{k}'] = recall
        metrics[f'precision@{k}'] = precision

    return metrics


def evaluate_retriever(retriever, retriever_name, questions, top_k=20):
    """Evaluate a retriever on a set of questions."""
    print(f"\nEvaluating {retriever_name}...")

    all_metrics = []
    success_examples = []
    failure_examples = []

    for question_data in tqdm(questions, desc="Processing questions"):
        question = question_data['question']
        supporting_facts = question_data['supporting_facts']

        # Retrieve
        results = retriever.retrieve(question, top_k=top_k)

        # Calculate metrics
        metrics = calculate_metrics(results, supporting_facts)
        all_metrics.append(metrics)

        # Track examples
        recall_5 = metrics.get('recall@5', 0)
        example = {
            'question': question,
            'answer': question_data['answer'],
            'type': question_data['type'],
            'level': question_data['level'],
            'recall@5': recall_5,
            'supporting_facts': supporting_facts,
            'retrieved': [(doc['title'], doc.get('sent_id'), score) for doc, score in results[:5]]
        }

        if recall_5 == 1.0:
            success_examples.append(example)
        elif recall_5 == 0.0:
            failure_examples.append(example)

    # Aggregate metrics
    aggregated = {}
    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics]
        aggregated[metric_name] = {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }

    return {
        'aggregated_metrics': aggregated,
        'success_examples': success_examples[:5],  # Top 5
        'failure_examples': failure_examples[:5],  # First 5 failures
        'num_questions': len(questions)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods")
    parser.add_argument('--retriever', type=str, required=True,
                        choices=['bm25', 'dense', 'hybrid'],
                        help='Retriever type to evaluate')
    parser.add_argument('--num_questions', type=int, default=100,
                        help='Number of questions to evaluate on')
    parser.add_argument('--corpus_size', type=int, default=100,
                        help='Number of questions to build corpus from')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of documents to retrieve')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha for hybrid retrieval (ignored for others)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=list_available_models(),
                        help=f'Embedding model for dense/hybrid retrieval (default: {DEFAULT_MODEL})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    args = parser.parse_args()

    # Determine data path
    if os.path.basename(os.getcwd()) == 'experiments':
        data_dir = "../data/raw"
    else:
        data_dir = "data/raw"

    # Load data
    print("Loading HotpotQA dataset...")
    dataset = HotpotQADataset(data_dir=data_dir)

    try:
        data = dataset.load_dev_distractor()
        print(f"Loaded {len(data)} examples")
    except FileNotFoundError:
        print(f"Dataset not found at {data_dir}")
        sys.exit(1)

    # Build corpus
    print(f"\nBuilding corpus from {args.corpus_size} questions...")
    corpus = build_corpus(data, max_items=args.corpus_size)
    print(f"Built corpus with {len(corpus)} documents")

    # Select questions for evaluation
    eval_questions = data[:args.num_questions]

    # Initialize retriever
    print(f"\nInitializing {args.retriever} retriever...")
    model_name = get_full_model_name(args.model)

    if args.retriever == 'bm25':
        retriever = BM25Retriever()
        retriever_name = "BM25"
    elif args.retriever == 'dense':
        retriever = DenseRetriever(model_name=model_name)
        retriever_name = f"Dense ({args.model})"
    elif args.retriever == 'hybrid':
        retriever = HybridRetriever(alpha=args.alpha, dense_model_name=model_name)
        retriever_name = f"Hybrid (alpha={args.alpha}, model={args.model})"

    retriever.index(corpus)

    # Evaluate
    results = evaluate_retriever(retriever, retriever_name, eval_questions, args.top_k)

    # Print results
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {retriever_name}")
    print("=" * 80)
    print(f"\nEvaluated on {results['num_questions']} questions")
    print(f"Corpus size: {len(corpus)} documents")
    print(f"Top-K: {args.top_k}")

    print("\n--- Aggregated Metrics ---")
    for metric_name, values in sorted(results['aggregated_metrics'].items()):
        print(f"{metric_name:15s}: {values['mean']:.4f} (min: {values['min']:.4f}, max: {values['max']:.4f})")

    print(f"\n--- Success Examples (Perfect Recall@5) ---")
    print(f"Found {len(results['success_examples'])} perfect retrievals")
    for i, ex in enumerate(results['success_examples'][:3], 1):
        print(f"\n{i}. {ex['question']}")
        print(f"   Answer: {ex['answer']}")
        print(f"   Type: {ex['type']}, Level: {ex['level']}")

    print(f"\n--- Failure Examples (Zero Recall@5) ---")
    print(f"Found {len(results['failure_examples'])} complete failures")
    for i, ex in enumerate(results['failure_examples'][:3], 1):
        print(f"\n{i}. {ex['question']}")
        print(f"   Answer: {ex['answer']}")
        print(f"   Type: {ex['type']}, Level: {ex['level']}")
        print(f"   Missing: {ex['supporting_facts']}")

    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{args.retriever}_{timestamp}.json"

    results['config'] = vars(args)
    results['timestamp'] = datetime.now().isoformat()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
