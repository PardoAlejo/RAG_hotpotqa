"""
Run systematic experiments comparing different retrieval methods.

Usage:
    python run_experiment.py --name "Initial BM25 Test" --num_questions 100
    python run_experiment.py --name "Comparison Study" --compare_all
"""

import sys
import os
import argparse
import json
from datetime import datetime
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_evaluation(retriever, num_questions, corpus_size, top_k, alpha=0.5):
    """Run evaluation for a single retriever configuration."""
    cmd = [
        'python', 'evaluate_retrieval.py',
        '--retriever', retriever,
        '--num_questions', str(num_questions),
        '--corpus_size', str(corpus_size),
        '--top_k', str(top_k),
    ]

    if retriever == 'hybrid':
        cmd.extend(['--alpha', str(alpha)])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{retriever}_{timestamp}.json"
    cmd.extend(['--output', output_file])

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        with open(output_file, 'r') as f:
            return json.load(f), output_file
    else:
        print(f"Error running evaluation for {retriever}")
        return None, None


def compare_results(results_list):
    """Compare results from multiple retrievers."""
    print("\n" + "=" * 80)
    print("COMPARISON ACROSS RETRIEVERS")
    print("=" * 80)

    # Extract metrics for comparison
    retrievers = [r['config']['retriever'] for r in results_list]

    print(f"\n{'Metric':<20} " + " ".join(f"{r:>12}" for r in retrievers))
    print("-" * 80)

    # Compare key metrics
    for metric in ['recall@1', 'recall@5', 'recall@10', 'precision@5']:
        values = []
        for result in results_list:
            if metric in result['aggregated_metrics']:
                values.append(result['aggregated_metrics'][metric]['mean'])
            else:
                values.append(0.0)

        print(f"{metric:<20} " + " ".join(f"{v:>12.4f}" for v in values))

    # Winner summary
    print("\n" + "-" * 80)
    print("Best Performer:")
    best_idx = max(range(len(values)), key=lambda i: values[i])
    print(f"  {retrievers[best_idx]} with Recall@5 = {values[best_idx]:.4f}")


def generate_report_template(experiment_name, results_files):
    """Generate a markdown report template."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"report_{timestamp}.md"

    with open(report_file, 'w') as f:
        f.write(f"# Experiment Report: {experiment_name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Setup\n\n")
        f.write("**Dataset:**\n")
        f.write("- Questions evaluated: [FILL IN]\n")
        f.write("- Corpus size: [FILL IN]\n")
        f.write("- Top-K retrieved: [FILL IN]\n\n")

        f.write("**Hardware:**\n")
        f.write("- CPU/GPU: [FILL IN]\n")
        f.write("- Memory: [FILL IN]\n\n")

        f.write("## Methods Tested\n\n")
        for result_file in results_files:
            with open(result_file, 'r') as rf:
                result = json.load(rf)
                retriever = result['config']['retriever']
                f.write(f"### {retriever.upper()}\n")
                f.write(f"- Configuration: [DESCRIBE PARAMETERS]\n")
                f.write(f"- Results file: `{result_file}`\n\n")

        f.write("## Results\n\n")
        f.write("### Quantitative Metrics\n\n")
        f.write("| Retriever | Recall@1 | Recall@5 | Recall@10 | Precision@5 |\n")
        f.write("|-----------|----------|----------|-----------|-------------|\n")

        for result_file in results_files:
            with open(result_file, 'r') as rf:
                result = json.load(rf)
                retriever = result['config']['retriever']
                metrics = result['aggregated_metrics']

                r1 = metrics.get('recall@1', {}).get('mean', 0)
                r5 = metrics.get('recall@5', {}).get('mean', 0)
                r10 = metrics.get('recall@10', {}).get('mean', 0)
                p5 = metrics.get('precision@5', {}).get('mean', 0)

                f.write(f"| {retriever:9s} | {r1:.4f}   | {r5:.4f}   | {r10:.4f}    | {p5:.4f}      |\n")

        f.write("\n### Key Findings\n\n")
        f.write("1. **What worked well:**\n")
        f.write("   - [FILL IN YOUR OBSERVATIONS]\n\n")

        f.write("2. **What didn't work:**\n")
        f.write("   - [FILL IN YOUR OBSERVATIONS]\n\n")

        f.write("3. **Surprising observations:**\n")
        f.write("   - [FILL IN YOUR OBSERVATIONS]\n\n")

        f.write("## Qualitative Analysis\n\n")
        f.write("### Success Cases\n\n")
        f.write("**Example 1:** [Pick a question where retrieval worked perfectly]\n")
        f.write("- Question: [FILL IN]\n")
        f.write("- Why it worked: [ANALYZE]\n\n")

        f.write("### Failure Cases\n\n")
        f.write("**Example 1:** [Pick a question where retrieval failed]\n")
        f.write("- Question: [FILL IN]\n")
        f.write("- Why it failed: [ANALYZE]\n")
        f.write("- Root cause: [RETRIEVAL ISSUE / MULTI-HOP COMPLEXITY / OTHER]\n\n")

        f.write("## Comparison to Previous Experiments\n\n")
        f.write("- Improvement over baseline: [FILL IN]\n")
        f.write("- Trade-offs observed: [SPEED / ACCURACY / COMPLEXITY]\n\n")

        f.write("## Computational Cost\n\n")
        f.write("| Retriever | Indexing Time | Query Time (avg) | Memory Usage |\n")
        f.write("|-----------|---------------|------------------|-------------|\n")
        f.write("| BM25      | [FILL IN]     | [FILL IN]        | [FILL IN]   |\n")
        f.write("| Dense     | [FILL IN]     | [FILL IN]        | [FILL IN]   |\n")
        f.write("| Hybrid    | [FILL IN]     | [FILL IN]        | [FILL IN]   |\n\n")

        f.write("## Conclusions\n\n")
        f.write("1. [KEY TAKEAWAY 1]\n")
        f.write("2. [KEY TAKEAWAY 2]\n")
        f.write("3. [KEY TAKEAWAY 3]\n\n")

        f.write("## Next Steps\n\n")
        f.write("- Hypothesis for next experiment: [FILL IN]\n")
        f.write("- What to test next: [FILL IN]\n")
        f.write("- Parameters to tune: [FILL IN]\n\n")

        f.write("## Raw Data\n\n")
        for result_file in results_files:
            f.write(f"- [{result_file}](./{result_file})\n")

    print(f"\n✓ Report template generated: {report_file}")
    print(f"  Fill in the marked sections with your observations.\n")

    return report_file


def main():
    parser = argparse.ArgumentParser(description="Run systematic experiments")
    parser.add_argument('--name', type=str, required=True,
                        help='Experiment name for the report')
    parser.add_argument('--num_questions', type=int, default=100,
                        help='Number of questions to evaluate')
    parser.add_argument('--corpus_size', type=int, default=100,
                        help='Corpus size for retrieval')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of documents to retrieve')
    parser.add_argument('--compare_all', action='store_true',
                        help='Run all retrievers and compare')
    parser.add_argument('--retrievers', nargs='+',
                        choices=['bm25', 'dense', 'hybrid'],
                        help='Specific retrievers to test')
    args = parser.parse_args()

    print("=" * 80)
    print(f"EXPERIMENT: {args.name}")
    print("=" * 80)

    # Determine which retrievers to run
    if args.compare_all:
        retrievers_to_test = ['bm25', 'dense', 'hybrid']
    elif args.retrievers:
        retrievers_to_test = args.retrievers
    else:
        print("Error: Specify either --compare_all or --retrievers")
        sys.exit(1)

    # Run evaluations
    results_list = []
    result_files = []

    for retriever in retrievers_to_test:
        result, result_file = run_evaluation(
            retriever,
            args.num_questions,
            args.corpus_size,
            args.top_k
        )

        if result:
            results_list.append(result)
            result_files.append(result_file)

    # Compare results if multiple retrievers
    if len(results_list) > 1:
        compare_results(results_list)

    # Generate report template
    report_file = generate_report_template(args.name, result_files)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved:")
    for rf in result_files:
        print(f"  - {rf}")
    print(f"\nReport template: {report_file}")
    print("\nNext step: Fill in the report with your observations!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
