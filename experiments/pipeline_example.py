"""Simple RAG pipeline example."""

import sys
import os

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrievers.bm25_retriever import BM25Retriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from evaluation.hotpot_metrics import HotpotQAEvaluator
from utils.data_loader import HotpotQADataset

# Load data
print("Loading HotpotQA dataset...")
# Determine correct path whether running from experiments/ or project root
if os.path.basename(os.getcwd()) == 'experiments':
    data_dir = "../data/raw"
else:
    data_dir = "data/raw"
dataset = HotpotQADataset(data_dir=data_dir)

try:
    train_data = dataset.load_dev_distractor()
    print(f"Loaded {len(train_data)} examples")
except FileNotFoundError:
    print("Dataset not found. Please download it first:")
    print(f"wget -P {data_dir} http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")
    sys.exit(1)

# Prepare corpus from context
print("\nBuilding corpus...")
corpus = []
for item in train_data[:100]:  # Use subset for demo
    for title, sentences in item['context']:
        for sent in sentences:
            corpus.append({'text': sent, 'title': title})

print(f"Built corpus with {len(corpus)} documents")

# Initialize retriever
print("\nInitializing BM25 retriever...")
retriever = BM25Retriever()
retriever.index(corpus)

# Retrieve documents
sample = train_data[0]
question = sample['question']
print(f"\nQuestion: {question}")
print(f"Gold Answer: {sample['answer']}")

results = retriever.retrieve(question, top_k=5)
print(f"\nRetrieved {len(results)} documents:")
for i, (doc, score) in enumerate(results):
    print(f"  [{i+1}] Score: {score:.4f} - {doc['text'][:80]}...")

# Optional: Rerank (uncomment to use)
# print("\nReranking...")
# reranker = CrossEncoderReranker()
# reranked = reranker.rerank(question, results, top_k=3)
# print(f"Reranked top 3 documents")

# Evaluate
print("\nEvaluating...")
evaluator = HotpotQAEvaluator()
metrics = evaluator.evaluate_example(
    answer_pred=sample['answer'],  # Replace with your LLM-generated answer
    answer_gold=sample['answer'],
    sp_pred=sample['supporting_facts'],
    sp_gold=sample['supporting_facts']
)

evaluator.print_metrics(metrics)
print("\nDone!")
