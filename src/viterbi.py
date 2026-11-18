"""Log-space Viterbi decoding for variable-size candidate sets."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True, slots=True)
class ViterbiResult:
    state_indices: list[int]
    log_probability: float | None
    status: str
    message: str | None = None


def viterbi_decode(emission_log_probs: Sequence[np.ndarray], transition_log_probs: Sequence[np.ndarray]) -> ViterbiResult:
    """Return one candidate index per observation or an explicit failure status."""
    if not emission_log_probs:
        return ViterbiResult([], None, "failed", "No observations were provided")
    emissions = [np.asarray(row, dtype=float) for row in emission_log_probs]
    if any(row.ndim != 1 or row.size == 0 for row in emissions):
        return ViterbiResult([], None, "failed", "An observation has no candidate states")
    if any(not np.isfinite(row).all() for row in emissions):
        return ViterbiResult([], None, "failed", "Emission scores contain invalid values")
    if len(transition_log_probs) != len(emissions) - 1:
        return ViterbiResult([], None, "failed", "Transition count must be one less than observation count")
    if len(emissions) == 1:
        state = int(np.argmax(emissions[0]))
        return ViterbiResult([state], float(emissions[0][state]), "success")
    scores = emissions[0].copy()
    backpointers: list[np.ndarray] = []
    for time_index in range(1, len(emissions)):
        transition = np.asarray(transition_log_probs[time_index - 1], dtype=float)
        expected_shape = (len(scores), len(emissions[time_index]))
        if transition.shape != expected_shape:
            return ViterbiResult([], None, "failed", f"Transition matrix {time_index - 1} has shape {transition.shape}, expected {expected_shape}")
        if np.isnan(transition).any() or np.isposinf(transition).any():
            return ViterbiResult([], None, "failed", "Transition matrix contains invalid log probabilities")
        combined = scores[:, None] + transition
        predecessor = np.argmax(combined, axis=0)
        best = combined[predecessor, np.arange(transition.shape[1])]
        scores = best + emissions[time_index]
        backpointers.append(predecessor)
        if not np.isfinite(scores).any():
            return ViterbiResult([], None, "failed", f"No connected candidate path reaches observation {time_index}")
    state = int(np.argmax(scores))
    final_score = float(scores[state])
    path = [state]
    for predecessor in reversed(backpointers):
        state = int(predecessor[state])
        path.append(state)
    path.reverse()
    return ViterbiResult(path, final_score, "success")
