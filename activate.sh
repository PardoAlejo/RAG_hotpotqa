#!/bin/bash
# Activation script for RAG HotpotQA conda environment

# Activate the conda environment
source ~/miniconda3/bin/activate rag_hotpotqa

echo "================================"
echo "RAG HotpotQA Environment Active"
echo "================================"
echo ""
echo "Python version: $(python --version)"
echo "Environment: rag_hotpotqa"
echo ""
echo "Quick commands:"
echo "  - Download data: cd data/raw && wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
echo "  - Run example: cd experiments && python example_pipeline.py"
echo "  - Deactivate: conda deactivate"
echo ""
