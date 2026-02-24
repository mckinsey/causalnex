"""
Ripple effect engine — the core of the global relationship graph.

Applies Pearl's *do*-calculus via ``InferenceEngine.do_intervention`` and
ranks every non-intervened node by the **Jensen-Shannon Divergence** (JSD)
between its baseline and post-intervention marginal distributions.

JSD is preferred over KL divergence because it is:

* symmetric: JSD(P || Q) == JSD(Q || P)
* bounded to [0, 1] when using log base 2
* well-defined even when probabilities are near zero
"""

from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon Divergence (base 2) between two distributions.

    Equivalent to ``scipy.spatial.distance.jensenshannon(p, q, base=2)``
    but avoids the external dependency on a specific scipy version.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Normalise just in case
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # Use log2 and guard against log(0)
    kl_pm = np.sum(np.where(p > 0, p * np.log2(p / np.where(m > 0, m, 1.0)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log2(q / np.where(m > 0, m, 1.0)), 0.0))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))


def compute_ripple_effect(
    bn: BayesianNetwork,
    event_node: str,
    event_state: Hashable,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Simulate a single-event intervention and rank node impact by JSD.

    Args:
        bn: a fitted ``BayesianNetwork``.
        event_node: node to intervene on (e.g. ``"evt_ukraine_war"``).
        event_state: the state to force (e.g. ``2`` for "major").
        top_k: return only the *k* most-affected nodes. ``None`` returns all.

    Returns:
        A list of ``(node_name, jsd_score)`` sorted descending by impact.
    """
    ie = InferenceEngine(bn)

    baseline = ie.query()
    ie.do_intervention(event_node, event_state)
    post = ie.query()
    ie.reset_do(event_node)

    impact = _rank_by_jsd(baseline, post, exclude={event_node})
    return impact[:top_k] if top_k else impact


def compound_ripple_effect(
    bn: BayesianNetwork,
    interventions: Dict[str, Hashable],
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Simulate multiple simultaneous interventions and rank impact.

    Args:
        bn: a fitted ``BayesianNetwork``.
        interventions: ``{node_name: forced_state}`` for every event in the
            compound scenario.
        top_k: return only the *k* most-affected nodes.

    Returns:
        A list of ``(node_name, jsd_score)`` sorted descending by impact.
    """
    ie = InferenceEngine(bn)
    baseline = ie.query()

    for node, state in interventions.items():
        ie.do_intervention(node, state)

    post = ie.query()

    for node in interventions:
        ie.reset_do(node)

    impact = _rank_by_jsd(baseline, post, exclude=set(interventions.keys()))
    return impact[:top_k] if top_k else impact


def _rank_by_jsd(
    baseline: Dict[str, Dict[Hashable, float]],
    post: Dict[str, Dict[Hashable, float]],
    exclude: set,
) -> List[Tuple[str, float]]:
    """Compute JSD for every node not in *exclude* and return sorted list."""
    scores = []
    for node in baseline:
        if node in exclude:
            continue
        # Ensure consistent state ordering
        states = sorted(baseline[node].keys())
        p = np.array([baseline[node][s] for s in states])
        q = np.array([post.get(node, baseline[node])[s] for s in states])
        scores.append((node, _jsd(p, q)))

    return sorted(scores, key=lambda x: x[1], reverse=True)
