"""Post-prediction route comparison metrics shared by API and offline evaluation."""
from __future__ import annotations
from collections import Counter


def sequence_similarity(predicted: list[int], reference: list[int]) -> float:
    """Normalized Levenshtein similarity for ordered arc sequences."""
    if not predicted and not reference:
        return 1.0
    previous = list(range(len(reference) + 1))
    for i, value in enumerate(predicted, 1):
        current = [i]
        for j, target in enumerate(reference, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (value != target)))
        previous = current
    return max(0.0, 1 - previous[-1] / max(len(predicted), len(reference), 1))


def route_sequence_metrics(predicted: list[int], reference: list[int]) -> dict[str, float | int]:
    """Return multiset edge metrics and sequence similarity."""
    predicted_counts, reference_counts = Counter(predicted), Counter(reference)
    overlap = sum((predicted_counts & reference_counts).values())
    precision = overlap / max(len(predicted), 1)
    recall = overlap / max(len(reference), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "route_edge_precision": float(precision),
        "route_edge_recall": float(recall),
        "route_edge_f1": float(f1),
        "sequence_similarity": float(sequence_similarity(predicted, reference)),
        "predicted_route_arc_count": len(predicted),
        "ground_truth_route_arc_count": len(reference),
        "comparison_available": True,
    }
