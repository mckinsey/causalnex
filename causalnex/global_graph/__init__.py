"""
Global relationship causal graph for modeling ripple effects across
world figures, news events, resources, currencies, country metrics, and
20+ configurable entity groups.

The ontology (node types, edge types, node instances, causal edges) is
stored in YAML config files by default, loaded via
:class:`~causalnex.global_graph.registry.OntologyRegistry`.  Swap in a
database backend for production-scale deployments.
"""

from causalnex.global_graph.registry import (
    OntologyRegistry,
    RegistryBackend,
    YAMLBackend,
    InMemoryBackend,
    get_default_registry,
    validate_node_name,
)
from causalnex.global_graph.node_registry import (
    NODE_REGISTRY,
    NodeType,
)
from causalnex.global_graph.data_ingestion import discretize_dataframe
from causalnex.global_graph.structure_builder import (
    build_expert_skeleton,
    merge_temporal_into_static,
    enforce_dag,
)
from causalnex.global_graph.bayesian_fitting import (
    build_bayesian_network,
    set_expert_cpds,
)
from causalnex.global_graph.ripple_engine import (
    compute_ripple_effect,
    compound_ripple_effect,
)
from causalnex.global_graph.path_tracer import trace_propagation_paths
from causalnex.global_graph.simulator import GlobalRippleSimulator
from causalnex.global_graph.visualization import visualize_impact_heatmap
from causalnex.global_graph.demo import run_demo

__all__ = [
    # Registry (new — preferred API)
    "OntologyRegistry",
    "RegistryBackend",
    "YAMLBackend",
    "InMemoryBackend",
    "get_default_registry",
    "validate_node_name",
    # Legacy compatibility
    "NODE_REGISTRY",
    "NodeType",
    # Pipeline stages
    "discretize_dataframe",
    "build_expert_skeleton",
    "merge_temporal_into_static",
    "enforce_dag",
    "build_bayesian_network",
    "set_expert_cpds",
    "compute_ripple_effect",
    "compound_ripple_effect",
    "trace_propagation_paths",
    "GlobalRippleSimulator",
    "visualize_impact_heatmap",
    "run_demo",
]
