# RAG Learning with HotpotQA

A comprehensive framework for learning and experimenting with Retrieval-Augmented Generation (RAG) systems using the HotpotQA dataset.

## Overview

This project provides a modular implementation of various RAG components:

- **Multiple Retrieval Systems**: BM25, Dense Retrieval, Hybrid approaches
- **Re-ranking Methods**: Cross-Encoder, MonoT5, LLM-based reranking
- **LLM Integration Strategies**: Simple concatenation, cited generation, chain-of-thought, fusion-in-decoder
- **Official Evaluation**: HotpotQA's official metrics implementation

## HotpotQA Dataset

HotpotQA is a dataset for multi-hop question answering that requires reasoning across multiple documents. It's particularly well-suited for RAG research because:

- Questions require finding and connecting information from multiple sources
- Includes supporting facts annotations (explainability)
- Two settings: distractor (easier) and fullwiki (harder, requires retrieval)

### Evaluation Metrics

The official HotpotQA evaluation measures:

1. **Answer Metrics**:
   - Exact Match (EM): Whether the predicted answer exactly matches the ground truth
   - F1 Score: Token-level overlap between prediction and ground truth
   - Precision: Fraction of predicted tokens that appear in ground truth
   - Recall: Fraction of ground truth tokens that appear in prediction

2. **Supporting Facts Metrics**:
   - Same metrics (EM, F1, Precision, Recall) for identifying correct supporting sentences
   - Evaluated as set matching of (title, sentence_id) tuples

3. **Joint Metrics**:
   - Combines answer and supporting facts performance
   - Calculated as: joint_metric = answer_metric × supporting_facts_metric
   - Encourages systems to both answer correctly AND identify correct evidence

## Project Structure

```
RAG_hotpotqa/
├── data/
│   ├── raw/              # Raw HotpotQA JSON files
│   └── processed/        # Preprocessed data
├── retrievers/           # Retrieval systems
│   ├── base_retriever.py
│   ├── bm25_retriever.py
│   ├── dense_retriever.py
│   └── hybrid_retriever.py
├── rerankers/            # Re-ranking systems
│   ├── base_reranker.py
│   ├── cross_encoder_reranker.py
│   └── llm_reranker.py
├── llm_integration/      # LLM integration strategies
│   ├── base_strategy.py
│   ├── simple_concatenation.py
│   └── cited_generation.py
├── evaluation/           # Evaluation metrics
│   ├── hotpot_metrics.py
│   └── hotpot_evaluate_v1_official.py
├── utils/                # Utility functions
│   └── data_loader.py
├── experiments/          # Experiment scripts
├── configs/              # Configuration files
└── notebooks/            # Jupyter notebooks
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download HotpotQA Dataset

Download the dataset files to `data/raw/`:

```bash
# Training set
wget -P data/raw http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

# Dev set (distractor setting)
wget -P data/raw http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

# Dev set (fullwiki setting)
wget -P data/raw http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
```

Or use curl:

```bash
curl -o data/raw/hotpot_train_v1.1.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
curl -o data/raw/hotpot_dev_distractor_v1.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
curl -o data/raw/hotpot_dev_fullwiki_v1.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
```

## Quick Start

### Load and Explore Data

```python
from utils.data_loader import HotpotQADataset

# Initialize dataset
dataset = HotpotQADataset(data_dir="data/raw")

# Load training data
train_data = dataset.load_train()

# Get statistics
stats = dataset.get_statistics(train_data)
print(stats)

# Examine a sample
sample = train_data[0]
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
print(f"Type: {sample['type']}")
print(f"Level: {sample['level']}")
```

### Build a Simple RAG Pipeline

```python
from retrievers.bm25_retriever import BM25Retriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from evaluation.hotpot_metrics import HotpotQAEvaluator

# Prepare corpus from context
corpus = []
for item in train_data[:1000]:  # Use subset for demo
    for title, sentences in item['context']:
        for sent in sentences:
            corpus.append({'text': sent, 'title': title})

# Initialize retriever
retriever = BM25Retriever()
retriever.index(corpus)

# Retrieve documents
question = "What is the capital of France?"
results = retriever.retrieve(question, top_k=10)

# Optional: Rerank
reranker = CrossEncoderReranker()
reranked = reranker.rerank(question, results, top_k=5)

# Evaluate (when you have predictions)
evaluator = HotpotQAEvaluator()
metrics = evaluator.evaluate_example(
    answer_pred="Paris",
    answer_gold="Paris",
    sp_pred=[["France", 0]],
    sp_gold=[["France", 0]]
)
evaluator.print_metrics(metrics)
```

## Retrieval Systems

### BM25 (Sparse Retrieval)
- Traditional lexical matching
- Fast and efficient
- Good baseline performance

### Dense Retrieval
- Neural embedding-based
- Captures semantic similarity
- Better for paraphrased queries

### Hybrid Retrieval
- Combines BM25 and Dense retrieval
- Best of both worlds
- Configurable alpha parameter for weighting

## Re-ranking Methods

### Cross-Encoder
- Joint encoding of query-document pairs
- High accuracy but slower
- Good for top-k refinement

### MonoT5
- T5-based pointwise reranker
- Predicts relevance directly
- Efficient for batch processing

### LLM Reranker
- Uses LLM prompting for relevance scoring
- Highest quality but most expensive
- Template provided for customization

## LLM Integration Strategies

### Simple Concatenation
- Concatenates all retrieved documents
- Straightforward approach
- Works well with shorter contexts

### Cited Generation
- Instructs LLM to cite sources
- Improves transparency
- Helps identify supporting facts

### Chain-of-Thought
- Step-by-step reasoning
- Particularly effective for multi-hop questions
- Better for complex reasoning tasks

### Fusion-in-Decoder
- Processes documents separately
- Better context integration
- Handles longer contexts

## Evaluation

The evaluation module implements the official HotpotQA metrics:

```python
from evaluation.hotpot_metrics import HotpotQAEvaluator

evaluator = HotpotQAEvaluator()

# Evaluate batch
predictions = [
    {'answer': 'predicted answer', 'sp': [['Title', 0], ['Title2', 1]]},
    # ... more predictions
]
ground_truths = [
    {'answer': 'gold answer', 'supporting_facts': [['Title', 0]]},
    # ... more ground truths
]

metrics = evaluator.evaluate_batch(predictions, ground_truths)
evaluator.print_metrics(metrics)
```

## Next Steps

1. **Configure LLM Client**: Add your LLM API keys and implement the LLM client interface in `llm_integration/` strategies

2. **Experiment**: Create experiment scripts in `experiments/` to test different combinations of:
   - Retrieval methods
   - Re-ranking approaches
   - LLM integration strategies

3. **Optimize**: Fine-tune hyperparameters:
   - Number of retrieved documents (top_k)
   - Hybrid retrieval alpha
   - Re-ranking thresholds

4. **Analyze**: Use notebooks to analyze:
   - Error cases
   - Performance by question type (bridge vs comparison)
   - Performance by difficulty level

## References

- HotpotQA Paper: [Yang et al., EMNLP 2018](https://arxiv.org/abs/1809.09600)
- HotpotQA Repository: https://github.com/hotpotqa/hotpot
- HotpotQA Website: http://hotpotqa.github.io/

## License

This project is for educational purposes. The HotpotQA dataset has its own license - please refer to the official repository.
