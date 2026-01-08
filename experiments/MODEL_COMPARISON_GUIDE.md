# Model Comparison Guide

## Overview

You can now compare different embedding models across all your experiments!

## Available Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **MiniLM-L6** | 80MB | ⚡⚡⚡ | ⭐⭐⭐ | Quick experiments, development |
| **DistilRoBERTa** | 290MB | ⚡⚡ | ⭐⭐⭐⭐ | Best quality/speed trade-off |
| **MPNet-Base** | 420MB | ⚡ | ⭐⭐⭐⭐⭐ | Highest quality, production |

## How to Compare Models

### 1. Interactive Comparison (Streamlit)

```bash
streamlit run retrieval_explorer.py
```

**Features:**
- Dropdown menu to select different models
- See model size, speed, and quality indicators
- **Models are cached** - switching back to a previous model is instant!
- Compare retrieval quality visually for the same question

**Usage:**
1. Start with MiniLM-L6 (fast, loads quickly)
2. Try a question, note the results
3. Switch to MPNet-Base (slower first time, but cached after)
4. Compare results - is the quality improvement worth the size?
5. Try DistilRoBERTa for the sweet spot

---

### 2. Quantitative Evaluation

```bash
# Test a specific model
python evaluate_retrieval.py --retriever dense --model all-mpnet-base-v2 --num_questions 100

# Compare different models manually
python evaluate_retrieval.py --retriever dense --model all-MiniLM-L6-v2 --num_questions 100
python evaluate_retrieval.py --retriever dense --model all-distilroberta-v1 --num_questions 100
```

---

### 3. Automated Model Comparison

```bash
# Compare ALL models automatically (recommended!)
python run_experiment.py --name "Model Comparison Study" --compare_models --num_questions 100
```

**This will:**
- Run dense retrieval with all 3 models
- Generate comparison table
- Create a report template with model specifications
- Show which model performs best for your data

**Expected output:**
```
COMPARISON ACROSS RETRIEVERS
================================
Metric               dense(MiniLM-L6)  dense(DistilRoBERTa)  dense(MPNet-Base)
--------------------------------------------------------------------------------
recall@1                       0.2500                0.2800             0.3100
recall@5                       0.5600                0.6100             0.6500
recall@10                      0.7200                0.7600             0.7900
precision@5                    0.2240                0.2440             0.2600

Best Performer: dense(MPNet-Base) with Recall@5 = 0.6500
```

---

## Experiment Ideas

### Experiment 1: Speed vs Quality
**Question:** Is the performance improvement worth the extra size?

```bash
python run_experiment.py --name "Speed vs Quality" --compare_models --num_questions 200
```

**What to analyze:**
- Recall@5 improvement from MiniLM → DistilRoBERTa → MPNet
- Is the improvement linear with size? Or diminishing returns?
- For your use case, where's the sweet spot?

---

### Experiment 2: Model Impact on Hybrid Retrieval
**Question:** Does the embedding model matter as much in hybrid retrieval?

```bash
# Test hybrid with different models
python run_experiment.py --name "Hybrid MiniLM" --retrievers hybrid --model all-MiniLM-L6-v2 --num_questions 100
python run_experiment.py --name "Hybrid MPNet" --retrievers hybrid --model all-mpnet-base-v2 --num_questions 100
```

**Hypothesis:** BM25 might compensate for weaker embedding models

---

### Experiment 3: Question Type Analysis
**Question:** Do certain models work better for bridge vs comparison questions?

**Method:**
1. Run model comparison: `python run_experiment.py --name "All Models" --compare_models --num_questions 200`
2. Use Streamlit to manually check 10 bridge questions with each model
3. Repeat for 10 comparison questions
4. Note patterns in the report

---

## Understanding the Trade-offs

### When to use each model:

**MiniLM-L6 (Fast)**
- ✅ Rapid prototyping
- ✅ Limited compute/memory
- ✅ Good enough baseline
- ❌ May miss subtle semantic matches

**DistilRoBERTa (Balanced)** ⭐ Recommended
- ✅ Production use for most cases
- ✅ Good balance of quality and speed
- ✅ Handles semantic similarity well
- ✅ Reasonable size (290MB)

**MPNet-Base (Highest Quality)**
- ✅ Maximum accuracy needed
- ✅ Have sufficient compute/memory
- ✅ Complex semantic queries
- ❌ Slowest, largest (420MB)

---

## Practical Workflow

### Phase 1: Development (Use MiniLM-L6)
- Fast iteration
- Test retrieval logic
- Debug issues

### Phase 2: Testing (Compare Models)
- Run `--compare_models` experiment
- Identify which model works best for your data
- Check if improvement justifies the size

### Phase 3: Production (Use Best Model)
- If quality critical: MPNet-Base
- For most cases: DistilRoBERTa
- If speed/size critical: MiniLM-L6

---

## Tips

1. **Cache is your friend**: In Streamlit, once a model is loaded, switching back is instant
2. **Start small**: Test on 50-100 questions first, then scale to 500+
3. **Question diversity matters**: Ensure your test set has both bridge and comparison questions
4. **Document your findings**: Use the auto-generated report templates
5. **Trade-offs vary**: What works for one dataset might not work for another

---

## Common Patterns from Research

Based on sentence-transformer benchmarks:

- **MPNet-Base** typically scores 2-3% higher than DistilRoBERTa
- **DistilRoBERTa** typically scores 1-2% higher than MiniLM-L6
- Performance gap is **larger** on complex multi-hop questions
- Performance gap is **smaller** on simple factual lookups

Test this on HotpotQA and document your findings!
