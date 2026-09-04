"""Vector normalization and similarity helpers."""

import math


def normalize(vector: list[float]) -> list[float]:
    """Return an L2-normalized copy of a vector."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity for equal-length vectors."""
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    left_norm = normalize(left)
    right_norm = normalize(right)
    return sum(a * b for a, b in zip(left_norm, right_norm))
