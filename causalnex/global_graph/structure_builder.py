"""
Hybrid structure learning: expert skeleton + optional DYNOTEARS temporal edges.

The graph is built in three layers:

1. **Expert skeleton** — edges that are definitively true by domain knowledge,
   loaded from the :class:`~causalnex.global_graph.registry.OntologyRegistry`
   (backed by YAML config files by default).
2. **DYNOTEARS learned edges** — inter-slice temporal relationships from
   time-series data (optional).
3. **DAG enforcement** — cycle removal that prefers dropping learned edges
   over expert edges, falling back to ``threshold_till_dag()`` when needed.
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx

from causalnex.structure import StructureModel
from causalnex.global_graph.registry import (
    OntologyRegistry,
    get_default_registry,
    validate_node_name,
)

# Legacy constant kept for backward compatibility.  New callers should pass
# an ``OntologyRegistry`` to ``build_expert_skeleton`` instead.
DEFAULT_EXPERT_EDGES: List[Tuple[str, str]] = [
    ("fig_mbs", "evt_opec_cut"),
    ("fig_powell", "evt_fed_rate_hike"),
    ("fig_biden", "evt_ukraine_war"),
    ("evt_ukraine_war", "res_oil_brent"),
    ("evt_ukraine_war", "res_wheat"),
    ("evt_opec_cut", "res_oil_brent"),
    ("res_oil_brent", "cur_usd_strength"),
    ("res_oil_brent", "cur_eur_strength"),
    ("evt_fed_rate_hike", "cur_usd_strength"),
    ("cur_usd_strength", "ctr_usa_gdp"),
    ("cur_eur_strength", "ctr_russia_gdp"),
    ("res_oil_brent", "ctr_russia_gdp"),
    ("ctr_china_trade", "res_semiconductor"),
    ("ctr_china_trade", "cur_cny_strength"),
]


def build_expert_skeleton(
    edges: Optional[List[Tuple[str, str]]] = None,
    registry: Optional[OntologyRegistry] = None,
) -> StructureModel:
    """Build a ``StructureModel`` from expert-defined causal edges.

    Edge source priority:

    1. If *edges* is provided, use that list directly (legacy mode).
    2. Else if *registry* is provided, use ``registry.to_expert_edges()``.
    3. Else load the default registry from the shipped YAML configs.

    Each edge is tagged with ``origin="expert"`` so that the DAG enforcement
    step can prioritise preserving them over data-learned edges.

    When built from a registry, additional metadata (``edge_type``,
    ``description``, ``weight``) is stored as edge attributes.

    Args:
        edges: list of ``(cause, effect)`` tuples.
        registry: an :class:`OntologyRegistry` instance.

    Returns:
        A ``StructureModel`` with expert edges.
    """
    sm = StructureModel()

    if edges is not None:
        # Legacy path: plain tuple list
        for u, v in edges:
            validate_node_name(u)
            validate_node_name(v)
            sm.add_edge(u, v, origin="expert")
        return sm

    # Registry path: rich metadata on edges
    if registry is None:
        registry = get_default_registry()

    typed_edges = registry.to_typed_expert_edges()
    for edge in typed_edges:
        src = edge["source"]
        tgt = edge["target"]
        validate_node_name(src)
        validate_node_name(tgt)
        sm.add_edge(
            src,
            tgt,
            origin=edge.get("origin", "expert"),
            edge_type=edge.get("edge_type", ""),
            weight=edge.get("weight", 1.0),
            description=edge.get("description", ""),
        )

    return sm


def merge_temporal_into_static(
    expert_sm: StructureModel,
    temporal_sm: StructureModel,
) -> StructureModel:
    """Merge inter-slice DYNOTEARS edges into the expert skeleton.

    DYNOTEARS creates nodes with ``_lagN`` suffixes.  This function extracts
    **inter-slice** edges (source has ``_lag1`` or higher) and maps them back
    to the base node names present in *expert_sm*.  Intra-slice (``_lag0``)
    edges are skipped because same-timestep relationships are already captured
    by expert knowledge.

    Only edges whose both base nodes already exist in *expert_sm* are added,
    and only if adding the edge would not create an immediate reverse-edge
    cycle.

    Args:
        expert_sm: the expert skeleton.
        temporal_sm: output of ``from_pandas_dynamic``.

    Returns:
        A new ``StructureModel`` with merged edges.
    """
    merged = StructureModel()
    for u, v, data in expert_sm.edges(data=True):
        merged.add_edge(u, v, **data)

    for u, v, data in temporal_sm.edges(data=True):
        # Only import inter-slice edges (lag > 0)
        if "_lag0" in u or "_lag" not in u:
            continue

        u_base = u.rsplit("_lag", 1)[0]
        v_base = v.rsplit("_lag", 1)[0] if "_lag" in v else v

        if u_base not in merged.nodes or v_base not in merged.nodes:
            continue
        if u_base == v_base:
            continue
        if merged.has_edge(v_base, u_base):
            continue
        if merged.has_edge(u_base, v_base):
            continue

        merged.add_edge(
            u_base,
            v_base,
            origin="learned",
            weight=data.get("weight", 1.0),
        )

    return merged


def enforce_dag(sm: StructureModel) -> StructureModel:
    """Ensure *sm* is a directed acyclic graph.

    Cycles are broken by removing edges in the following priority order:

    1. The weakest ``origin="learned"`` edge in the cycle.
    2. If the cycle is entirely expert edges, fall back to
       ``StructureModel.threshold_till_dag()``.

    Args:
        sm: a ``StructureModel`` that may contain cycles.

    Returns:
        The same ``StructureModel`` instance, modified in-place to be a DAG.
    """
    while not nx.is_directed_acyclic_graph(sm):
        try:
            cycle = nx.find_cycle(sm)
        except nx.NetworkXNoCycle:
            break

        # nx.find_cycle returns list of (u, v) or (u, v, key) tuples
        cycle_edges = [(e[0], e[1]) for e in cycle]
        removable = []
        for u, v in cycle_edges:
            edge_data = sm[u][v]
            if edge_data.get("origin") != "expert":
                removable.append((u, v, abs(edge_data.get("weight", 0.0))))

        if removable:
            u, v, _ = min(removable, key=lambda x: x[2])
            sm.remove_edge(u, v)
        else:
            sm.threshold_till_dag()
            break

    return sm
