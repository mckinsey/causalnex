"""
Causal path tracing between nodes in the global relationship graph.

Uses ``networkx.all_simple_paths`` to enumerate directed paths from a
source event to a target node, revealing *why* an intervention propagates
the way it does.
"""

from typing import List

import networkx as nx

from causalnex.structure import StructureModel


def trace_propagation_paths(
    sm: StructureModel,
    source_node: str,
    target_node: str,
    max_paths: int = 5,
    cutoff: int = 6,
) -> List[List[str]]:
    """Find directed causal paths from *source_node* to *target_node*.

    Args:
        sm: the ``StructureModel`` (DAG).
        source_node: the intervention origin (e.g. ``"evt_ukraine_war"``).
        target_node: the affected node (e.g. ``"ctr_russia_gdp"``).
        max_paths: maximum number of paths to return.
        cutoff: maximum path length to search (prevents exponential
            blow-up on dense graphs).

    Returns:
        A list of paths (each path is a list of node names), sorted by
        ascending path length. An empty list is returned when no directed
        path exists.
    """
    if source_node not in sm.nodes or target_node not in sm.nodes:
        return []

    paths = list(
        nx.all_simple_paths(sm, source=source_node, target=target_node, cutoff=cutoff)
    )
    return sorted(paths, key=len)[:max_paths]
