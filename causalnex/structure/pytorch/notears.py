# Copyright 2019-2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tools to learn a ``StructureModel`` which describes the conditional dependencies between variables in a dataset.
"""

import logging
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_array

from causalnex.structure.pytorch.core import NotearsMLP
from causalnex.structure.pytorch.dist_type import DistTypeContinuous, dist_type_aliases
from causalnex.structure.structuremodel import StructureModel

__all__ = ["from_numpy", "from_pandas"]


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def from_numpy(
    X: np.ndarray,
    dist_type_schema: Dict[int, str] = None,
    lasso_beta: float = 0.0,
    ridge_beta: float = 0.0,
    use_bias: bool = False,
    hidden_layer_units: Iterable[int] = None,
    w_threshold: float = None,
    max_iter: int = 100,
    tabu_edges: List[Tuple[int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
    use_gpu: bool = True,
    **kwargs,
) -> StructureModel:
    """
    Learn the `StructureModel`, the graph structure with lasso regularisation
    describing conditional dependencies between variables in data presented as a numpy array.

    Based on DAGs with NO TEARS.
    @inproceedings{zheng2018dags,
        author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
        booktitle = {Advances in Neural Information Processing Systems},
        title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
        year = {2018},
        codebase = {https://github.com/xunzheng/notears}
    }

    Args:
        X: 2d input data, axis=0 is data rows, axis=1 is data columns. Data must be row oriented.
        dist_type_schema: The dist type schema corresponding to the passed in data X.
        It maps the positional column in X to the string alias of a dist type.
        A list of alias names can be found in ``dist_type/__init__.py``.
        If None, assumes that all data in X is continuous.

        lasso_beta: Constant that multiplies the lasso term (l1 regularisation).
        NOTE when using nonlinearities, the l1 loss only applies to the dag_layer.

        use_bias: Whether to fit a bias parameter in the NOTEARS algorithm.

        ridge_beta: Constant that multiplies the ridge term (l2 regularisation).
        When using nonlinear layers use of this parameter is recommended.

        hidden_layer_units: An iterable where its length determine the number of layers used,
        and the numbers determine the number of nodes used for the layer in order.

        w_threshold: fixed threshold for absolute edge weights.

        max_iter: max number of dual ascent steps during optimisation.

        tabu_edges: list of edges(from, to) not to be included in the graph.

        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.

        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

        use_gpu: use gpu if it is set to True and CUDA is available.

        **kwargs: additional arguments for NOTEARS MLP model

    Returns:
        StructureModel: a graph of conditional dependencies between data variables.

    Raises:
        ValueError: If schema does not correspond to columns.
    """
    # n examples, d properties
    if not X.size:
        raise ValueError("Input data X is empty, cannot learn any structure")
    logging.info("Learning structure using 'NOTEARS' optimisation.")

    # Check array for NaN or inf values
    check_array(X)

    if dist_type_schema is not None:

        # make sure that there is one provided key per column
        if set(range(X.shape[1])).symmetric_difference(set(dist_type_schema.keys())):
            raise ValueError(
                f"Difference indices and expected indices. Got {dist_type_schema} schema"
            )

    # if dist_type_schema is None, assume all columns are continuous, else init the alias mapped object
    dist_types = (
        [DistTypeContinuous(idx=idx) for idx in np.arange(X.shape[1])]
        if dist_type_schema is None
        else [
            dist_type_aliases[alias](idx=idx) for idx, alias in dist_type_schema.items()
        ]
    )

    # shape of X before preprocessing
    _, d_orig = X.shape
    # perform dist type pre-processing (i.e. column expansion)
    for dist_type in dist_types:
        # NOTE: preprocess_X must be called first to perform possible column expansions
        X = dist_type.preprocess_X(X)
        tabu_edges = dist_type.preprocess_tabu_edges(tabu_edges)
        tabu_parent_nodes = dist_type.preprocess_tabu_nodes(tabu_parent_nodes)
        tabu_child_nodes = dist_type.preprocess_tabu_nodes(tabu_child_nodes)
    # shape of X after preprocessing
    _, d = X.shape

    # if None or empty, convert into a list with single item
    if hidden_layer_units is None:
        hidden_layer_units = [0]
    elif isinstance(hidden_layer_units, list) and not hidden_layer_units:
        hidden_layer_units = [0]

    # if no hidden layer units, still take 1 iteration step with bounds
    hidden_layer_bnds = hidden_layer_units[0] if hidden_layer_units[0] else 1

    # Flip i and j because Pytorch flattens the vector in another direction
    bnds = [
        (0, 0)
        if i == j
        else (0, 0)
        if tabu_edges is not None and (i, j) in tabu_edges
        else (0, 0)
        if tabu_parent_nodes is not None and i in tabu_parent_nodes
        else (0, 0)
        if tabu_child_nodes is not None and j in tabu_child_nodes
        else (None, None)
        for j in range(d)
        for _ in range(hidden_layer_bnds)
        for i in range(d)
    ]
    model = NotearsMLP(
        n_features=d,
        dist_types=dist_types,
        hidden_layer_units=hidden_layer_units,
        lasso_beta=lasso_beta,
        ridge_beta=ridge_beta,
        bounds=bnds,
        use_bias=use_bias,
        use_gpu=use_gpu,
        **kwargs,
    )
    model.fit(X, max_iter=max_iter)
    sm = StructureModel(model.adj)

    if w_threshold:
        sm.remove_edges_below_threshold(w_threshold)

    # extract the mean effect and add as edge attribute
    mean_effect = model.adj_mean_effect
    for u, v, edge_dict in sm.edges.data(True):
        sm.add_edge(
            u,
            v,
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=mean_effect[u, v],
        )

    # set bias as node attribute
    bias = model.bias
    for node in sm.nodes():
        value = None
        if bias is not None:
            value = bias[node]
        sm.nodes[node]["bias"] = value

    # attach each dist_type object to corresponding node(s)
    for dist_type in dist_types:
        sm = dist_type.add_to_node(sm)

    # preserve the structure_learner as a graph attribute
    sm.graph["structure_learner"] = model

    # collapse the adj down and store as graph attr
    adj = deepcopy(model.adj)
    for dist_type in dist_types:
        adj = dist_type.collapse_adj(adj)
    sm.graph["graph_collapsed"] = StructureModel(adj[:d_orig, :d_orig])

    return sm


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def from_pandas(
    X: pd.DataFrame,
    dist_type_schema: Dict[Union[str, int], str] = None,
    lasso_beta: float = 0.0,
    ridge_beta: float = 0.0,
    use_bias: bool = False,
    hidden_layer_units: Iterable[int] = None,
    max_iter: int = 100,
    w_threshold: float = None,
    tabu_edges: List[Tuple[str, str]] = None,
    tabu_parent_nodes: List[str] = None,
    tabu_child_nodes: List[str] = None,
    use_gpu: bool = True,
    **kwargs,
) -> StructureModel:
    """
    Learn the `StructureModel`, the graph structure describing conditional dependencies between variables
    in data presented as a pandas dataframe.

    The optimisation is to minimise a score function :math:`F(W)` over the graph's
    weighted adjacency matrix, :math:`W`, subject to the a constraint function :math:`h(W)`,
    where :math:`h(W) == 0` characterises an acyclic graph.
    :math:`h(W) > 0` is a continuous, differentiable function that encapsulated how acyclic the graph is
    (less == more acyclic).
    Full details of this approach to structure learning are provided in the publication:

    Based on DAGs with NO TEARS.
    @inproceedings{zheng2018dags,
        author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
        booktitle = {Advances in Neural Information Processing Systems},
        title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
        year = {2018},
        codebase = {https://github.com/xunzheng/notears}
    }

    Args:
        X: 2d input data, axis=0 is data rows, axis=1 is data columns. Data must be row oriented.

        dist_type_schema: The dist type schema corresponding to the passed in data X.
        It maps the pandas column name in X to the string alias of a dist type.
        A list of alias names can be found in ``dist_type/__init__.py``.
        If None, assumes that all data in X is continuous.

        lasso_beta: Constant that multiplies the lasso term (l1 regularisation).
        NOTE when using nonlinearities, the l1 loss only applies to the dag_layer.

        use_bias: Whether to fit a bias parameter in the NOTEARS algorithm.

        ridge_beta: Constant that multiplies the ridge term (l2 regularisation).
        When using nonlinear layers use of this parameter is recommended.

        hidden_layer_units: An iterable where its length determine the number of layers used,
        and the numbers determine the number of nodes used for the layer in order.

        w_threshold: fixed threshold for absolute edge weights.

        max_iter: max number of dual ascent steps during optimisation.

        tabu_edges: list of edges(from, to) not to be included in the graph.

        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.

        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

        use_gpu: use gpu if it is set to True and CUDA is available

        **kwargs: additional arguments for NOTEARS MLP model

    Returns:
         StructureModel: graph of conditional dependencies between data variables.

    Raises:
        ValueError: If X does not contain data.
    """

    data = deepcopy(X)

    # if dist_type_schema is not None, convert dist_type_schema from cols to idx
    dist_type_schema = (
        dist_type_schema
        if dist_type_schema is None
        else {X.columns.get_loc(col): alias for col, alias in dist_type_schema.items()}
    )
    non_numeric_cols = data.select_dtypes(exclude="number").columns

    if len(non_numeric_cols) > 0:
        raise ValueError(
            "All columns must have numeric data. "
            f"Consider mapping the following columns to int {non_numeric_cols}"
        )

    col_idx = {c: i for i, c in enumerate(data.columns)}
    idx_col = {i: c for c, i in col_idx.items()}

    if tabu_edges:
        tabu_edges = [(col_idx[u], col_idx[v]) for u, v in tabu_edges]
    if tabu_parent_nodes:
        tabu_parent_nodes = [col_idx[n] for n in tabu_parent_nodes]
    if tabu_child_nodes:
        tabu_child_nodes = [col_idx[n] for n in tabu_child_nodes]

    g = from_numpy(
        X=data.values,
        dist_type_schema=dist_type_schema,
        lasso_beta=lasso_beta,
        ridge_beta=ridge_beta,
        use_bias=use_bias,
        hidden_layer_units=hidden_layer_units,
        w_threshold=w_threshold,
        max_iter=max_iter,
        tabu_edges=tabu_edges,
        tabu_parent_nodes=tabu_parent_nodes,
        tabu_child_nodes=tabu_child_nodes,
        use_gpu=use_gpu,
        **kwargs,
    )

    # set comprehension to ensure only unique dist types are extraced
    # NOTE: this prevents double-renaming caused by the same dist type used on expanded columns
    unique_dist_types = {node[1]["dist_type"] for node in g.nodes(data=True)}
    # use the dist types to update the idx_col mapping
    idx_col_expanded = deepcopy(idx_col)
    for dist_type in unique_dist_types:
        idx_col_expanded = dist_type.update_idx_col(idx_col_expanded)

    sm = StructureModel()
    # add expanded set of nodes
    sm.add_nodes_from(list(idx_col_expanded.values()))

    # recover the edge weights from g
    for u, v, edge_dict in g.edges.data(True):
        sm.add_edge(
            idx_col_expanded[u],
            idx_col_expanded[v],
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=edge_dict["mean_effect"],
        )

    # retrieve all graphs attrs
    for key, val in g.graph.items():
        sm.graph[key] = val

    # recover the node biases from g
    for node in g.nodes(data=True):
        node_name = idx_col_expanded[node[0]]
        sm.nodes[node_name]["bias"] = node[1]["bias"]

    # recover and preseve the node dist_types
    for node_data in g.nodes(data=True):
        node_name = idx_col_expanded[node_data[0]]
        sm.nodes[node_name]["dist_type"] = node_data[1]["dist_type"]

    # recover the collapsed model from g
    sm_collapsed = StructureModel()
    sm_collapsed.add_nodes_from(list(idx_col.values()))
    for u, v, edge_dict in g.graph["graph_collapsed"].edges.data(True):
        sm_collapsed.add_edge(
            idx_col[u],
            idx_col[v],
            origin="learned",
            weight=edge_dict["weight"],
        )
    sm.graph["graph_collapsed"] = sm_collapsed

    return sm
