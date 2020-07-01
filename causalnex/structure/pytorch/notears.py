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
from typing import List, Tuple

import numpy as np
import pandas as pd

from causalnex.structure.pytorch.core import NotearsMLP
from causalnex.structure.structuremodel import StructureModel

__all__ = ["from_numpy", "from_pandas"]


def from_numpy(
    X: np.ndarray,
    beta: float = 0.0,
    w_threshold: float = None,
    max_iter: int = 100,
    tabu_edges: List[Tuple[int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
    use_gpu: bool = True,
    **kwargs
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
        beta: Constant that multiplies the lasso term (l1 regularisation)
        w_threshold: fixed threshold for absolute edge weights.
        and the numbers determine the number of nodes used for the layer in order.e.g. [10, 10]
        max_iter: max number of dual ascent steps during optimisation.
        tabu_edges: list of edges(from, to) not to be included in the graph.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.
        use_gpu: use gpu if it is set to True and CUDA is available.
        **kwargs: additional arguments for NOTEARS MLP model
    Returns:
        StructureModel: a graph of conditional dependencies between data variables.
    Raises:
        ValueError: If X does not contain data.
    """
    # n examples, d properties
    if not X.size:
        raise ValueError("Input data X is empty, cannot learn any structure")
    logging.info("Learning structure using 'NOTEARS' optimisation.")

    _, d = X.shape

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
        else (0, None)
        for j in range(d)
        for i in range(d)
    ]

    model = NotearsMLP(n_features=d, lasso_beta=beta, bounds=bnds, **kwargs)

    model.fit(X, max_iter=max_iter, use_gpu=use_gpu)

    return StructureModel(model.get_adj(w_threshold))


def from_pandas(
    X: pd.DataFrame,
    beta: float = 0.0,
    max_iter: int = 100,
    w_threshold: float = None,
    tabu_edges: List[Tuple[str, str]] = None,
    tabu_parent_nodes: List[str] = None,
    tabu_child_nodes: List[str] = None,
    use_gpu: bool = True,
    **kwargs
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
        X: input pandas dataframe
        beta: Constant that multiplies the lasso term (l1 regularisation)
        w_threshold: fixed threshold for absolute edge weights.
        and the numbers determine the number of nodes used for the layer in order.e.g. [10, 10]
        max_iter: max number of dual ascent steps during optimisation.
        w_threshold: fixed threshold for absolute edge weights.
        tabu_edges: list of edges(from, to) not to be included in the graph.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.
        use_gpu: use gpu if it is set to True and CUDA is available
        **kwargs: additional arrguments for NOYEARS MLP
    Returns:
         StructureModel: graph of conditional dependencies between data variables.
    Raises:
        ValueError: If X does not contain data.
    """

    data = deepcopy(X)

    non_numeric_cols = data.select_dtypes(exclude="number").columns

    if len(non_numeric_cols) > 0:
        raise ValueError(
            "All columns must have numeric data. "
            "Consider mapping the following columns to int {non_numeric_cols}".format(
                non_numeric_cols=non_numeric_cols
            )
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
        data.values,
        beta,
        w_threshold,
        max_iter,
        tabu_edges,
        tabu_parent_nodes,
        tabu_child_nodes,
        use_gpu,
        **kwargs
    )

    sm = StructureModel()
    sm.add_nodes_from(data.columns)
    sm.add_weighted_edges_from(
        [(idx_col[u], idx_col[v], w) for u, v, w in g.edges.data("weight")],
        origin="learned",
    )

    return sm
