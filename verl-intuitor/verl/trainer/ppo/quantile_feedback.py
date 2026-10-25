"""Exact empirical-quantile comparisons for SIF-Q (no sampling grid)."""

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np


def validate_peer_threshold(peer_threshold: float) -> float:
    threshold = float(peer_threshold)
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError(f"peer_threshold must be finite and in (0, 1], got {peer_threshold}")
    return threshold


def validate_use_direct_adv(use_direct_adv: bool) -> bool:
    if not isinstance(use_direct_adv, bool):
        raise TypeError(f"use_direct_adv must be a bool, got {type(use_direct_adv).__name__}")
    return use_direct_adv


def _sorted_net_support(a: Sequence[float], b: Sequence[float], *, exact: bool = False) -> float | Fraction:
    """Integrate two descending, nonempty, finite profiles using integer boundaries."""
    li, lj = len(a), len(b)
    total = li * lj  # Python integers cannot overflow; no array of this size is allocated.
    i = j = position = numerator = 0
    while position < total:
        next_i = (i + 1) * lj
        next_j = (j + 1) * li
        boundary = min(next_i, next_j)
        sign = int(a[i] > b[j]) - int(a[i] < b[j])
        numerator += (boundary - position) * sign
        position = boundary
        if position == next_i:
            i += 1
        if position == next_j:
            j += 1
    return Fraction(numerator, total) if exact else numerator / total


def _sort_profile(scores: Sequence[float]) -> list[float]:
    values = [float(value) for value in scores]
    if not values:
        raise ValueError("Each response must have at least one valid token")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Valid-token self-certainties must be finite")
    return sorted(values, reverse=True)


def quantile_net_support(scores_i: Sequence[float], scores_j: Sequence[float]) -> float:
    """Return integral sign(Q_i(u) - Q_j(u)) du over the full unit interval.

    Equality is an exact tie; there is no tolerance, length threshold, or grid.
    Sorting costs O(L log L); the boundary merge costs O(L_i + L_j).
    """
    return _sorted_net_support(_sort_profile(scores_i), _sort_profile(scores_j))


def pairwise_quantile_support(profiles: Sequence[Sequence[float]], *, exact: bool = False) -> np.ndarray:
    """Sort each response once, then construct an antisymmetric peer matrix.

    Caller supplies only valid responses belonging to one prompt. CPU lists
    avoid a GPU synchronization for every scalar comparison in the merge.
    With exact=True, retain rational supports in an object array until group
    aggregation so cancelling evidence gives exactly zero without a tolerance.
    """
    sorted_profiles = [_sort_profile(profile) for profile in profiles]
    count = len(sorted_profiles)
    support = np.zeros((count, count), dtype=object if exact else np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            value = _sorted_net_support(sorted_profiles[i], sorted_profiles[j], exact=exact)
            support[i, j] = value
            support[j, i] = -value
    return support
