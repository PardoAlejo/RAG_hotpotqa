# Experiments Guide

This directory contains tools for exploring and evaluating RAG retrieval methods.

## Tools Overview

### 1. 🎨 Interactive Explorer (Streamlit)
**File:** `retrieval_explorer.py`

Visual, interactive tool to compare retrieval methods side-by-side for individual questions.

```bash
streamlit run retrieval_explorer.py
```

**Features:**
- Compare BM25, Dense, and Hybrid retrieval visually
- **🆕 Choose between 3 embedding models** with different speed/quality trade-offs
- Adjust parameters interactively (top-k, alpha, corpus size)
- See which documents are supporting facts (highlighted in green)
- Navigate through questions easily
- View model size, speed, and quality indicators
- Perfect for qualitative analysis and debugging

**Best for:** Understanding *why* retrievers behave the way they do

**Available Embedding Models:**
- **MiniLM-L6 (Fast)**: 80MB, ⚡⚡⚡ fastest, ⭐⭐⭐ good for quick experiments
- **DistilRoBERTa (Balanced)**: 290MB, ⚡⚡ medium speed, ⭐⭐⭐⭐ best quality/speed trade-off
- **MPNet-Base (Highest Quality)**: 420MB, ⚡ slower but ⭐⭐⭐⭐⭐ highest quality

---

### 2. 📊 Evaluation Script
**File:** `evaluate_retrieval.py`

Quantitative evaluation of a single retriever on N questions.

```bash
# Evaluate BM25 on 100 questions
python evaluate_retrieval.py --retriever bm25 --num_questions 100

# Evaluate Dense retrieval with default model
python evaluate_retrieval.py --retriever dense --num_questions 50

# Evaluate Dense with MPNet model
python evaluate_retrieval.py --retriever dense --model all-mpnet-base-v2 --num_questions 50

# Evaluate Hybrid with alpha=0.7 and DistilRoBERTa
python evaluate_retrieval.py --retriever hybrid --alpha 0.7 --model all-distilroberta-v1 --num_questions 100
```

**Parameters:**
- `--retriever`: `bm25`, `dense`, or `hybrid`
- `--num_questions`: Number of questions to evaluate (default: 100)
- `--corpus_size`: Number of questions to build corpus from (default: 100)
- `--top_k`: Number of documents to retrieve (default: 20)
- `--alpha`: For hybrid retrieval only (default: 0.5)
- `--model`: Embedding model for dense/hybrid (choices: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `all-distilroberta-v1`)
- `--output`: Output JSON file name (auto-generated if not specified)

**Outputs:**
- Aggregated metrics (Recall@K, Precision@K)
- Success examples (perfect retrieval)
- Failure examples (zero recall)
- JSON file with full results

**Best for:** Getting quantitative performance numbers

---

### 3. 🔬 Experiment Runner
**File:** `run_experiment.py`

Run systematic experiments comparing multiple retrievers and generate a report template.

```bash
# Compare all three retrievers
python run_experiment.py --name "Initial Comparison" --compare_all --num_questions 100

# Compare all embedding models (Dense retrieval only)
python run_experiment.py --name "Model Comparison" --compare_models --num_questions 100

# Compare specific retrievers with a specific model
python run_experiment.py --name "BM25 vs Dense" --retrievers bm25 dense --model all-mpnet-base-v2 --num_questions 200

# Test hybrid with different model
python run_experiment.py --name "Hybrid MPNet" --retrievers hybrid --model all-mpnet-base-v2 --num_questions 50

# Quick test
python run_experiment.py --name "Quick Test" --compare_all --num_questions 20
```

**Parameters:**
- `--name`: Experiment name (required)
- `--compare_all`: Test all retrievers (bm25, dense, hybrid) with selected model
- `--compare_models`: **🆕 Compare all embedding models** (runs dense with each model)
- `--retrievers`: Specify which retrievers to test
- `--model`: Embedding model to use (default: all-MiniLM-L6-v2)
- `--num_questions`: Number of questions (default: 100)
- `--corpus_size`: Corpus size (default: 100)
- `--top_k`: Documents to retrieve (default: 20)

**Outputs:**
- Individual result JSON files for each retriever
- Comparison table across retrievers
- Markdown report template to fill in

**Best for:** Systematic experiments with documentation

---

## Suggested Workflow

### Phase 1: Explore (Qualitative)
1. Start Streamlit app: `streamlit run retrieval_explorer.py`
2. Look at 10-20 questions manually
3. Note patterns: When does BM25 win? When does Dense win?
4. Try different alpha values for Hybrid
5. **Write down observations**

### Phase 2: Measure (Quantitative)
1. Run evaluation on each retriever:
   ```bash
   python evaluate_retrieval.py --retriever bm25 --num_questions 100
   python evaluate_retrieval.py --retriever dense --num_questions 100
   python evaluate_retrieval.py --retriever hybrid --num_questions 100
   ```
2. Compare the metrics
3. **Document which performs best**

### Phase 3: Experiment (Systematic)
1. Run full comparison:
   ```bash
   python run_experiment.py --name "Full Comparison Study" --compare_all --num_questions 200
   ```
2. Fill in the generated report template
3. **Analyze trade-offs and insights**

### Phase 4: Iterate
1. Based on findings, test variations:
   - Different alpha values for hybrid
   - Larger corpus sizes
   - Different question types (filter by `type` field)
2. Keep refining your understanding
3. **Build up your learnings incrementally**

---

## Example Learning Path

### Week 1: BM25 Deep Dive
```bash
# Day 1-2: Explore manually
streamlit run retrieval_explorer.py
# Focus on question_idx 0-50, observe BM25 behavior

# Day 3-4: Quantitative analysis
python evaluate_retrieval.py --retriever bm25 --num_questions 100

# Day 5: Document findings in a report
```

### Week 2: Dense vs BM25
```bash
# Compare both
python run_experiment.py --name "BM25 vs Dense" --retrievers bm25 dense --num_questions 100

# Use Streamlit to understand differences
streamlit run retrieval_explorer.py
```

### Week 3: Model Comparison
```bash
# Compare all embedding models
python run_experiment.py --name "Embedding Model Comparison" --compare_models --num_questions 100

# Or compare individually
python evaluate_retrieval.py --retriever dense --model all-MiniLM-L6-v2 --num_questions 100
python evaluate_retrieval.py --retriever dense --model all-mpnet-base-v2 --num_questions 100
python evaluate_retrieval.py --retriever dense --model all-distilroberta-v1 --num_questions 100

# Understand quality vs speed trade-offs
```

### Week 4: Hybrid Optimization
```bash
# Test different alpha values with different models
python evaluate_retrieval.py --retriever hybrid --alpha 0.3 --model all-mpnet-base-v2 --num_questions 100
python evaluate_retrieval.py --retriever hybrid --alpha 0.5 --model all-mpnet-base-v2 --num_questions 100
python evaluate_retrieval.py --retriever hybrid --alpha 0.7 --model all-mpnet-base-v2 --num_questions 100

# Find optimal balance
```

---

## Understanding the Metrics

### Recall@K
Fraction of supporting facts found in top-K results.
- **Recall@5 = 1.0** means all supporting facts are in top 5
- **Recall@5 = 0.5** means half of supporting facts are in top 5
- **Higher is better**

### Precision@K
Fraction of top-K results that are supporting facts.
- **Precision@5 = 0.4** means 2/5 retrieved docs are supporting facts
- **Higher is better**
- Trade-off with recall

### When to Use Each Metric
- **Recall**: How many relevant docs did we find?
- **Precision**: How many retrieved docs are actually relevant?
- For RAG: High recall is usually more important (better to have relevant docs mixed with some noise than miss relevant docs entirely)

---

## Tips

1. **Start small**: Use `--num_questions 20` for quick iterations
2. **Increase gradually**: Scale up to 100, 200, 500 as you understand the patterns
3. **Use Streamlit for debugging**: When metrics look strange, use Streamlit to see what's actually being retrieved
4. **Save your reports**: Build a collection of experiment reports to track your learning
5. **Question types matter**: Bridge vs comparison questions may behave differently
6. **Corpus size matters**: Larger corpus = harder retrieval task

---

## Output Files

All experiments generate timestamped files:
- `results_{retriever}_{timestamp}.json` - Full evaluation results
- `report_{timestamp}.md` - Report template to fill in

Keep these organized! They're your learning record.
