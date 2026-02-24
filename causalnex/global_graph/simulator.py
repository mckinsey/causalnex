"""
High-level narrative simulator for what-if scenarios.

``GlobalRippleSimulator`` wraps the ripple engine and path tracer to provide
a single entry-point for running named scenarios. Each scenario returns a
ranked impact list together with a human-readable narrative summary.
"""

from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.global_graph.ripple_engine import _jsd
from causalnex.global_graph.path_tracer import trace_propagation_paths


class GlobalRippleSimulator:
    """Run named what-if scenarios on a fitted global relationship graph.

    Example::

        sim = GlobalRippleSimulator(bn, sm, NODE_REGISTRY)
        result = sim.simulate_scenario(
            "Russia-Ukraine Escalation + OPEC Cut",
            {"evt_ukraine_war": 2, "evt_opec_cut": 2},
        )
        print(result["narrative"])
    """

    def __init__(
        self,
        bn: BayesianNetwork,
        sm: StructureModel,
        node_registry: Optional[Dict[str, Dict]] = None,
    ):
        self.bn = bn
        self.sm = sm
        self.node_registry = node_registry or {}
        self._ie = InferenceEngine(bn)
        self._baseline = self._ie.query()

    def simulate_scenario(
        self,
        scenario_name: str,
        events: Dict[str, Hashable],
        top_k: int = 10,
    ) -> dict:
        """Run a named what-if scenario.

        Args:
            scenario_name: human-readable label.
            events: ``{node_name: forced_state}`` for the conditions to impose.
            top_k: number of most-affected nodes to include.

        Returns:
            A dict with keys ``scenario``, ``interventions``,
            ``ranked_impact``, ``post_marginals``, ``narrative``, and
            ``propagation_paths``.
        """
        ie = InferenceEngine(self.bn)
        baseline = ie.query()

        for node, state in events.items():
            ie.do_intervention(node, state)
        post = ie.query()
        for node in events:
            ie.reset_do(node)

        # Rank impact
        impact: List[Tuple[str, float]] = []
        non_intervened = set(self.bn.nodes) - set(events.keys())
        for node in non_intervened:
            states = sorted(baseline[node].keys())
            p = np.array([baseline[node][s] for s in states])
            q = np.array([post.get(node, baseline[node])[s] for s in states])
            impact.append((node, _jsd(p, q)))

        impact.sort(key=lambda x: x[1], reverse=True)
        ranked = impact[:top_k]

        # Trace propagation paths for top affected nodes
        paths: Dict[str, List[List[str]]] = {}
        for affected_node, _ in ranked[:5]:
            for event_node in events:
                node_paths = trace_propagation_paths(
                    self.sm, event_node, affected_node, max_paths=3
                )
                if node_paths:
                    paths.setdefault(affected_node, []).extend(node_paths)

        narrative = self._build_narrative(baseline, post, events, ranked)

        return {
            "scenario": scenario_name,
            "interventions": events,
            "ranked_impact": ranked,
            "post_marginals": post,
            "narrative": narrative,
            "propagation_paths": paths,
        }

    def _build_narrative(
        self,
        baseline: dict,
        post: dict,
        events: dict,
        top_affected: List[Tuple[str, float]],
    ) -> str:
        """Generate a human-readable narrative for the top affected nodes."""
        lines = [f"Scenario interventions: {events}", ""]
        for node, jsd in top_affected:
            baseline_dominant = max(baseline[node], key=baseline[node].get)
            post_marginals = post.get(node, baseline[node])
            post_dominant = max(post_marginals, key=post_marginals.get)
            label = self.node_registry.get(node, {}).get("label", node)

            if baseline_dominant == post_dominant:
                direction = "unchanged (dominant state held)"
            else:
                direction = (
                    f"shifted from state {baseline_dominant} to state {post_dominant}"
                )
            lines.append(f"  {label}: JSD={jsd:.4f}, {direction}")

        return "\n".join(lines)
