"""
HotpotQA official evaluation metrics.

Based on the official evaluation script from:
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

Implements the exact metrics used in the HotpotQA benchmark.
"""

import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Any


def normalize_answer(s: str) -> str:
    """
    Normalize answer text (official implementation).

    Applies:
    1. Lowercase conversion
    2. Punctuation removal
    3. Article removal (a, an, the)
    4. Whitespace normalization
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Calculate exact match score (official implementation)."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Calculate F1, precision, and recall (official implementation).

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # Special handling for yes/no/noanswer responses
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def update_answer(metrics: Dict[str, float], prediction: str, gold: str) -> Tuple[float, float, float]:
    """
    Update answer metrics (official implementation).

    Args:
        metrics: Dictionary to update with metrics
        prediction: Predicted answer
        gold: Ground truth answer

    Returns:
        Tuple of (em, precision, recall)
    """
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def update_sp(
    metrics: Dict[str, float],
    prediction: List[List],
    gold: List[List]
) -> Tuple[float, float, float]:
    """
    Update supporting facts metrics (official implementation).

    Args:
        metrics: Dictionary to update with metrics
        prediction: List of [title, sent_id] predictions
        gold: List of [title, sent_id] ground truths

    Returns:
        Tuple of (em, precision, recall)
    """
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


class HotpotQAEvaluator:
    """
    HotpotQA evaluator implementing official metrics.

    Metrics:
    1. Answer metrics: EM, F1, Precision, Recall
    2. Supporting Facts metrics: EM, F1, Precision, Recall
    3. Joint metrics: Combined answer + supporting facts metrics
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = {
            'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
            'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
            'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0
        }

    def evaluate_example(
        self,
        answer_pred: str,
        answer_gold: str,
        sp_pred: List[List] = None,
        sp_gold: List[List] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single example (official implementation logic).

        Args:
            answer_pred: Predicted answer
            answer_gold: Ground truth answer
            sp_pred: Predicted supporting facts [[title, sent_id], ...]
            sp_gold: Ground truth supporting facts [[title, sent_id], ...]

        Returns:
            Dictionary with all metrics for this example
        """
        metrics = {
            'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
            'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
            'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0
        }

        # Evaluate answer
        em, prec, recall = update_answer(metrics, answer_pred, answer_gold)

        # Evaluate supporting facts if provided
        can_eval_joint = True
        if sp_pred is not None and sp_gold is not None:
            sp_em, sp_prec, sp_recall = update_sp(metrics, sp_pred, sp_gold)
        else:
            can_eval_joint = False

        # Calculate joint metrics (official way)
        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.0
            joint_em = em * sp_em

            metrics['joint_em'] = joint_em
            metrics['joint_f1'] = joint_f1
            metrics['joint_prec'] = joint_prec
            metrics['joint_recall'] = joint_recall

        return metrics

    def evaluate_batch(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions (official implementation logic).

        Args:
            predictions: List of dicts with 'answer' and optionally 'sp' keys
            ground_truths: List of dicts with 'answer' and 'supporting_facts' keys

        Returns:
            Dictionary with averaged metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match ground truths")

        metrics = {
            'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
            'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
            'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0
        }

        for pred, gold in zip(predictions, ground_truths):
            can_eval_joint = True

            # Evaluate answer
            em, prec, recall = update_answer(
                metrics, pred.get('answer', ''), gold.get('answer', '')
            )

            # Evaluate supporting facts if available
            if 'sp' in pred and 'supporting_facts' in gold:
                sp_em, sp_prec, sp_recall = update_sp(
                    metrics, pred['sp'], gold['supporting_facts']
                )
            else:
                can_eval_joint = False
                sp_em, sp_prec, sp_recall = 0, 0, 0

            # Calculate joint metrics (official way)
            if can_eval_joint:
                joint_prec = prec * sp_prec
                joint_recall = recall * sp_recall
                if joint_prec + joint_recall > 0:
                    joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                else:
                    joint_f1 = 0.0
                joint_em = em * sp_em

                metrics['joint_em'] += joint_em
                metrics['joint_f1'] += joint_f1
                metrics['joint_prec'] += joint_prec
                metrics['joint_recall'] += joint_recall

        # Average all metrics
        N = len(ground_truths)
        for k in metrics.keys():
            metrics[k] /= N

        return metrics

    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a formatted way."""
        print("\n" + "=" * 60)
        print("HotpotQA Official Evaluation Metrics")
        print("=" * 60)

        print("\nAnswer Metrics:")
        print(f"  Exact Match (EM):  {metrics.get('em', 0.0):.4f}")
        print(f"  F1 Score:          {metrics.get('f1', 0.0):.4f}")
        print(f"  Precision:         {metrics.get('prec', 0.0):.4f}")
        print(f"  Recall:            {metrics.get('recall', 0.0):.4f}")

        print("\nSupporting Facts Metrics:")
        print(f"  Exact Match (EM):  {metrics.get('sp_em', 0.0):.4f}")
        print(f"  F1 Score:          {metrics.get('sp_f1', 0.0):.4f}")
        print(f"  Precision:         {metrics.get('sp_prec', 0.0):.4f}")
        print(f"  Recall:            {metrics.get('sp_recall', 0.0):.4f}")

        print("\nJoint Metrics (Answer × Supporting Facts):")
        print(f"  Exact Match (EM):  {metrics.get('joint_em', 0.0):.4f}")
        print(f"  F1 Score:          {metrics.get('joint_f1', 0.0):.4f}")
        print(f"  Precision:         {metrics.get('joint_prec', 0.0):.4f}")
        print(f"  Recall:            {metrics.get('joint_recall', 0.0):.4f}")

        print("=" * 60 + "\n")
