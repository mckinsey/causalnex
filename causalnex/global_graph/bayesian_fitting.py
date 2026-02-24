"""
BayesianNetwork construction and CPD fitting for the global relationship graph.

Wraps ``causalnex.network.BayesianNetwork`` with sensible defaults for
sparse geopolitical data (K2 Bayesian prior to avoid zero-probability CPD
entries) and provides helpers for manually injecting expert CPD tables.
"""

from typing import Optional

import networkx as nx
import pandas as pd

from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel


def build_bayesian_network(
    sm: StructureModel,
    df_discretized: pd.DataFrame,
    method: str = "BayesianEstimator",
    bayes_prior: str = "K2",
) -> BayesianNetwork:
    """Build and fit a ``BayesianNetwork`` from a DAG and discretized data.

    Steps:

    1. Validate that *sm* is a single weakly-connected DAG (fall back to
       the largest subgraph if disconnected).
    2. Construct a ``BayesianNetwork``.
    3. Call ``fit_node_states`` → ``fit_cpds`` using the requested estimator.

    Args:
        sm: a DAG (must satisfy ``nx.is_directed_acyclic_graph``).
        df_discretized: a DataFrame whose columns match the nodes of *sm*
            and whose values are discrete integer states.
        method: CPD estimation method —
            ``"BayesianEstimator"`` (default) or ``"MaximumLikelihoodEstimator"``.
        bayes_prior: prior for ``BayesianEstimator``.  ``"K2"`` adds a
            pseudo-count of 1 regardless of cardinality (safe for sparse data).

    Returns:
        A fitted ``BayesianNetwork`` ready for inference.

    Raises:
        ValueError: if *sm* contains cycles.
    """
    if not nx.is_directed_acyclic_graph(sm):
        raise ValueError("StructureModel contains cycles — call enforce_dag() first.")

    if nx.number_weakly_connected_components(sm) > 1:
        sm = sm.get_largest_subgraph()

    bn = BayesianNetwork(sm)
    bn.fit_node_states(df_discretized)

    if method == "BayesianEstimator":
        bn.fit_cpds(df_discretized, method=method, bayes_prior=bayes_prior)
    else:
        bn.fit_cpds(df_discretized, method=method)

    return bn


def set_expert_cpds(
    bn: BayesianNetwork,
    cpds: Optional[dict] = None,
) -> BayesianNetwork:
    """Inject manually-defined CPD tables into *bn*.

    Args:
        bn: a fitted ``BayesianNetwork``.
        cpds: mapping of ``{node_name: pd.DataFrame}`` where each DataFrame
            has:

            - row index named after the node and containing its states;
            - ``MultiIndex`` columns named after parent nodes with their
              states.

            When *None*, a small set of illustrative expert CPDs for
            ``evt_opec_cut`` is applied (assumes ``fig_mbs`` is a parent with
            states {0, 1, 2}).

    Returns:
        The same ``BayesianNetwork`` instance with CPDs replaced.
    """
    if cpds is None:
        cpds = _default_expert_cpds(bn)

    for node, cpd_df in cpds.items():
        if node in bn.nodes:
            bn.set_cpd(node, cpd_df)

    return bn


def _default_expert_cpds(bn: BayesianNetwork) -> dict:
    """Build a small default CPD dict for demonstration purposes."""
    cpds = {}

    if "evt_opec_cut" in bn.nodes and "fig_mbs" in [
        p for p, _ in bn.edges if _ == "evt_opec_cut"
    ]:
        parents = sorted(bn.structure.predecessors("evt_opec_cut"))
        if parents == ["fig_mbs"]:
            node_states = sorted(bn.node_states["evt_opec_cut"])
            parent_states = sorted(bn.node_states["fig_mbs"])

            # P(evt_opec_cut | fig_mbs)
            data = {
                (ps,): [
                    0.7 if ps == 0 else (0.4 if ps == 1 else 0.1),  # state 0
                    0.2 if ps == 0 else (0.4 if ps == 1 else 0.2),  # state 1
                    0.1 if ps == 0 else (0.2 if ps == 1 else 0.7),  # state 2
                ]
                for ps in parent_states
            }
            cpd_df = pd.DataFrame(data, index=node_states)
            cpd_df.index.name = "evt_opec_cut"
            cpd_df.columns = pd.MultiIndex.from_tuples(
                cpd_df.columns, names=["fig_mbs"]
            )
            cpds["evt_opec_cut"] = cpd_df

    return cpds
