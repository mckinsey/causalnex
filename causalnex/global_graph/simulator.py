"""
High-level narrative simulator for what-if scenarios.

``GlobalRippleSimulator`` wraps the ripple engine and path tracer to provide
a single entry-point for running named scenarios. Each scenario returns a
ranked impact list together with a human-readable narrative summary.

When an :class:`~causalnex.global_graph.registry.OntologyRegistry` is
provided, the narrative is enriched with entity group names, relationship
types, and edge descriptions drawn from the config.
"""

from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.global_graph.registry import OntologyRegistry
from causalnex.global_graph.ripple_engine import _jsd
from causalnex.global_graph.path_tracer import trace_propagation_paths


class GlobalRippleSimulator:
    """Run named what-if scenarios on a fitted global relationship graph.

    Example::

        from causalnex.global_graph.registry import get_default_registry

        registry = get_default_registry()
        sim = GlobalRippleSimulator(bn, sm, registry=registry)
        result = sim.simulate_scenario(
            "Russia-Ukraine Escalation + OPEC Cut",
            {"evt_ukraine_war": 2, "evt_opec_cut": 2},
        )
        print(result["narrative"])

    Args:
        bn: a fitted :class:`BayesianNetwork`.
        sm: the :class:`StructureModel` (DAG) that produced *bn*.
        node_registry: **deprecated** — legacy dict of
            ``{node_id: {"label": ..., ...}}``.  Use *registry* instead.
        registry: an :class:`OntologyRegistry` for rich metadata.
    """

    def __init__(
        self,
        bn: BayesianNetwork,
        sm: StructureModel,
        node_registry: Optional[Dict[str, Dict]] = None,
        registry: Optional[OntologyRegistry] = None,
    ):
        self.bn = bn
        self.sm = sm
        self.registry = registry
        # Legacy fallback: if no registry but a node_registry dict is given
        self.node_registry = node_registry or {}
        self._ie = InferenceEngine(bn)
        self._baseline = self._ie.query()

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def _label_for(self, node_id: str) -> str:
        """Return a human-readable label for *node_id*."""
        if self.registry:
            try:
                return self.registry.get_node(node_id).get("label", node_id)
            except KeyError:
                pass
        return self.node_registry.get(node_id, {}).get("label", node_id)

    def _type_label_for(self, node_id: str) -> str:
        """Return the entity group label for *node_id* (e.g. 'Commodity')."""
        if self.registry:
            try:
                node_meta = self.registry.get_node(node_id)
                type_id = node_meta.get("type", "")
                type_meta = self.registry.get_node_type(type_id)
                return type_meta.get("label", type_id)
            except KeyError:
                pass
        return ""

    def _edge_description(self, source: str, target: str) -> str:
        """Return a description for the edge source→target."""
        if self.registry:
            for edge in self.registry.edges:
                if edge["source"] == source and edge["target"] == target:
                    desc = edge.get("description", "")
                    if desc:
                        return desc
                    etype = edge.get("edge_type", "")
                    if etype:
                        try:
                            return self.registry.get_edge_type(etype).get(
                                "label", etype
                            )
                        except KeyError:
                            return etype
        return ""

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

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

        narrative = self._build_narrative(baseline, post, events, ranked, paths)

        return {
            "scenario": scenario_name,
            "interventions": events,
            "ranked_impact": ranked,
            "post_marginals": post,
            "narrative": narrative,
            "propagation_paths": paths,
        }

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------

    def _build_narrative(
        self,
        baseline: dict,
        post: dict,
        events: dict,
        top_affected: List[Tuple[str, float]],
        paths: Dict[str, List[List[str]]],
    ) -> str:
        """Generate a human-readable narrative for the top affected nodes."""
        lines: List[str] = []

        # Describe interventions
        lines.append("SCENARIO INTERVENTIONS:")
        for node, state in events.items():
            label = self._label_for(node)
            type_label = self._type_label_for(node)
            prefix = f"[{type_label}] " if type_label else ""
            lines.append(f"  {prefix}{label} → forced to state {state}")
        lines.append("")

        # Describe impacts
        lines.append("RANKED IMPACTS (by Jensen-Shannon Divergence):")
        for node, jsd in top_affected:
            baseline_dominant = max(baseline[node], key=baseline[node].get)
            post_marginals = post.get(node, baseline[node])
            post_dominant = max(post_marginals, key=post_marginals.get)
            label = self._label_for(node)
            type_label = self._type_label_for(node)
            prefix = f"[{type_label}] " if type_label else ""

            if baseline_dominant == post_dominant:
                direction = "dominant state held, distribution shifted"
            else:
                direction = (
                    f"shifted from state {baseline_dominant} "
                    f"to state {post_dominant}"
                )
            lines.append(f"  {prefix}{label}: JSD={jsd:.4f}, {direction}")

        # Describe key propagation paths
        if paths:
            lines.append("")
            lines.append("KEY PROPAGATION PATHS:")
            for target_node, target_paths in paths.items():
                target_label = self._label_for(target_node)
                for path in target_paths[:2]:  # at most 2 paths per target
                    path_labels = [self._label_for(n) for n in path]
                    chain = " → ".join(path_labels)
                    lines.append(f"  {chain}")
                    # Add edge descriptions along the path
                    for i in range(len(path) - 1):
                        desc = self._edge_description(path[i], path[i + 1])
                        if desc:
                            lines.append(f"    ({desc})")

        return "\n".join(lines)
