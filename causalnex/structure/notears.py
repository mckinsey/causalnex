# Copyright 2019-2020 QuantumBlack Visual Analytics Limited
#
# The methods found in this file are derived from a repository under Apache 2.0:
# DAGs with NO TEARS.
# @inproceedings{zheng2018dags,
#     author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
#     booktitle = {Advances in Neural Information Processing Systems},
#     title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
#     year = {2018},
#     codebase = {https://github.com/xunzheng/notears}
# }
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
import warnings
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt

from causalnex.structure.structuremodel import StructureModel

__all__ = ["from_numpy", "from_pandas", "from_numpy_lasso", "from_pandas_lasso"]


def from_numpy(
    X: np.ndarray,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
    tabu_edges: List[Tuple[int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
) -> StructureModel:
    """
    Learn the `StructureModel`, the graph structure describing conditional dependencies between variables
    in data presented as a numpy array.

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
        max_iter: max number of dual ascent steps during optimisation.
        h_tol: exit if h(W) < h_tol (as opposed to strict definition of 0).
        w_threshold: fixed threshold for absolute edge weights.
        tabu_edges: list of edges(from, to) not to be included in the graph.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

    Returns:
        StructureModel: a graph of conditional dependencies between data variables.

    Raises:
        ValueError: If X does not contain data.
    """

    # n examples, d properties
    _, d = X.shape

    _assert_all_finite(X)

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
        for i in range(d)
        for j in range(d)
    ]

    return _learn_structure(X, bnds, max_iter, h_tol, w_threshold)


def from_numpy_lasso(
    X: np.ndarray,
    beta: float,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
    tabu_edges: List[Tuple[int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
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
        beta: Constant that multiplies the lasso term.
        max_iter: max number of dual ascent steps during optimisation.
        h_tol: exit if h(W) < h_tol (as opposed to strict definition of 0).
        w_threshold: fixed threshold for absolute edge weights.
        tabu_edges: list of edges(from, to) not to be included in the graph.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

    Returns:
        StructureModel: a graph of conditional dependencies between data variables.

    Raises:
        ValueError: If X does not contain data.
    """

    # n examples, d properties
    _, d = X.shape

    _assert_all_finite(X)

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
        for i in range(d)
        for j in range(d)
    ] * 2

    return _learn_structure_lasso(X, beta, bnds, max_iter, h_tol, w_threshold)


def from_pandas(
    X: pd.DataFrame,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
    tabu_edges: List[Tuple[str, str]] = None,
    tabu_parent_nodes: List[str] = None,
    tabu_child_nodes: List[str] = None,
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
        X: input data.
        max_iter: max number of dual ascent steps during optimisation.
        h_tol: exit if h(W) < h_tol (as opposed to strict definition of 0).
        w_threshold: fixed threshold for absolute edge weights.
        tabu_edges: list of edges(from, to) not to be included in the graph.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

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
        data.values,
        max_iter,
        h_tol,
        w_threshold,
        tabu_edges,
        tabu_parent_nodes,
        tabu_child_nodes,
    )

    sm = StructureModel()
    sm.add_nodes_from(data.columns)
    sm.add_weighted_edges_from(
        [(idx_col[u], idx_col[v], w) for u, v, w in g.edges.data("weight")],
        origin="learned",
    )

    return sm


def from_pandas_lasso(
    X: pd.DataFrame,
    beta: float,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
    tabu_edges: List[Tuple[str, str]] = None,
    tabu_parent_nodes: List[str] = None,
    tabu_child_nodes: List[str] = None,
) -> StructureModel:
    """
    Learn the `StructureModel`, the graph structure with lasso regularisation
    describing conditional dependencies between variables in data presented as a pandas dataframe.

    Based on DAGs with NO TEARS.
    @inproceedings{zheng2018dags,
        author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
        booktitle = {Advances in Neural Information Processing Systems},
        title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
        year = {2018},
        codebase = {https://github.com/xunzheng/notears}
    }

    Args:
        X: input data.
        beta: Constant that multiplies the lasso term.
        max_iter: max number of dual ascent steps during optimisation.
        h_tol: exit if h(W) < h_tol (as opposed to strict definition of 0).
        w_threshold: fixed threshold for absolute edge weights.
        tabu_edges: list of edges(from, to) not to be included in the graph.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

    Returns:
         StructureModel: graph of conditional dependencies between data variables.

    Raises:
        ValueError: If X does not contain data.
    """

    data = deepcopy(X)

    non_numeric_cols = data.select_dtypes(exclude="number").columns

    if not non_numeric_cols.empty:
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

    g = from_numpy_lasso(
        data.values,
        beta,
        max_iter,
        h_tol,
        w_threshold,
        tabu_edges,
        tabu_parent_nodes,
        tabu_child_nodes,
    )

    sm = StructureModel()
    sm.add_nodes_from(data.columns)
    sm.add_weighted_edges_from(
        [(idx_col[u], idx_col[v], w) for u, v, w in g.edges.data("weight")],
        origin="learned",
    )

    return sm


def _learn_structure(
    X: np.ndarray,
    bnds,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
) -> StructureModel:
    """
    Based on initial implementation at https://github.com/xunzheng/notears
    """

    def _h(w: np.ndarray) -> float:
        """
        Constraint function of the NOTEARS algorithm.

        Args:
            w:  current adjacency matrix.

        Returns:
            float: DAGness of the adjacency matrix (0 == DAG, >0 == cyclic).
        """

        W = w.reshape([d, d])
        return np.trace(slin.expm(W * W)) - d

    def _func(w: np.ndarray) -> float:
        """
        Objective function that the NOTEARS algorithm tries to minimise.

        Args:
            w: current adjacency matrix.

        Returns:
            float: objective.
        """

        W = w.reshape([d, d])
        loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), "fro"))
        h = _h(W)
        return loss + 0.5 * rho * h * h + alpha * h

    def _grad(w: np.ndarray) -> np.ndarray:
        """
        Gradient function used to compute next step in NOTEARS algorithm.

        Args:
            w: the current adjacency matrix.

        Returns:
            np.ndarray: gradient vector.
        """

        W = w.reshape([d, d])
        loss_grad = -1.0 / n * X.T.dot(X).dot(np.eye(d, d) - W)
        E = slin.expm(W * W)
        obj_grad = loss_grad + (rho * (np.trace(E) - d) + alpha) * E.T * W * 2
        return obj_grad.flatten()

    if X.size == 0:
        raise ValueError("Input data X is empty, cannot learn any structure")
    logging.info("Learning structure using 'NOTEARS' optimisation.")

    # n examples, d properties
    n, d = X.shape
    # initialise matrix to zeros
    w_est, w_new = np.zeros(d * d), np.zeros(d * d)

    # initialise weights and constraints
    rho, alpha, h, h_new = 1.0, 0.0, np.inf, np.inf

    # start optimisation
    for n_iter in range(max_iter):
        while (rho < 1e20) and (h_new > 0.25 * h or h_new == np.inf):
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=_grad, bounds=bnds)
            w_new = sol.x
            h_new = _h(w_new)
            if h_new > 0.25 * h:
                rho *= 10

        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol:
            break
        if h > h_tol and n_iter == max_iter - 1:
            warnings.warn("Failed to converge. Consider increasing max_iter.")

    w_est[np.abs(w_est) <= w_threshold] = 0
    return StructureModel(w_est.reshape([d, d]))


def _learn_structure_lasso(
    X: np.ndarray,
    beta: float,
    bnds,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
) -> StructureModel:
    """
    Based on initial implementation at https://github.com/xunzheng/notears
    """

    def _h(w_vec: np.ndarray) -> float:
        """
        Constraint function of the NOTEARS algorithm with lasso regularisation.

        Args:
            w_vec:  weight vector (wpos and wneg).

        Returns:
            float: DAGness of the adjacency matrix (0 == DAG, >0 == cyclic).
        """

        W = w_vec.reshape([d, d])
        return np.trace(slin.expm(W * W)) - d

    def _func(w_vec: np.ndarray) -> float:
        """
        Objective function that the NOTEARS algorithm with lasso regularisation tries to minimise.

        Args:
            w_vec: weight vector (wpos and wneg).

        Returns:
            float: objective.
        """

        w_pos = w_vec[: d ** 2]
        w_neg = w_vec[d ** 2 :]

        wmat_pos = w_pos.reshape([d, d])
        wmat_neg = w_neg.reshape([d, d])

        wmat = wmat_pos - wmat_neg
        loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - wmat), "fro"))
        h_val = _h(wmat)
        return loss + 0.5 * rho * h_val * h_val + alpha * h_val + beta * w_vec.sum()

    def _grad(w_vec: np.ndarray) -> np.ndarray:
        """
        Gradient function used to compute next step in NOTEARS algorithm with lasso regularisation.

        Args:
            w_vec: weight vector (wpos and wneg).

        Returns:
            np.ndarray: gradient vector.
        """

        w_pos = w_vec[: d ** 2]
        w_neg = w_vec[d ** 2 :]

        grad_vec = np.zeros(2 * d ** 2)
        wmat_pos = w_pos.reshape([d, d])
        wmat_neg = w_neg.reshape([d, d])

        wmat = wmat_pos - wmat_neg

        loss_grad = -1.0 / n * X.T.dot(X).dot(np.eye(d, d) - wmat)
        exp_hdmrd = slin.expm(wmat * wmat)
        obj_grad = (
            loss_grad
            + (rho * (np.trace(exp_hdmrd) - d) + alpha) * exp_hdmrd.T * wmat * 2
        )
        lbd_grad = beta * np.ones(d * d)
        grad_vec[: d ** 2] = obj_grad.flatten() + lbd_grad
        grad_vec[d ** 2 :] = -obj_grad.flatten() + lbd_grad

        return grad_vec

    if X.size == 0:
        raise ValueError("Input data X is empty, cannot learn any structure")
    logging.info(
        "Learning structure using 'NOTEARS' optimisation with lasso regularisation."
    )

    n, d = X.shape
    w_est, w_new = np.zeros(2 * d * d), np.zeros(2 * d * d)
    rho, alpha, h_val, h_new = 1.0, 0.0, np.inf, np.inf
    for n_iter in range(max_iter):
        while (rho < 1e20) and (h_new > 0.25 * h_val or h_new == np.inf):
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=_grad, bounds=bnds)
            w_new = sol.x
            h_new = _h(
                w_new[: d ** 2].reshape([d, d]) - w_new[d ** 2 :].reshape([d, d])
            )
            if h_new > 0.25 * h_val:
                rho *= 10

        w_est, h_val = w_new, h_new
        alpha += rho * h_val
        if h_val <= h_tol:
            break
        if h_val > h_tol and n_iter == max_iter - 1:
            warnings.warn("Failed to converge. Consider increasing max_iter.")

    w_new = w_est[: d ** 2].reshape([d, d]) - w_est[d ** 2 :].reshape([d, d])
    w_new[np.abs(w_new) < w_threshold] = 0
    return StructureModel(w_new.reshape([d, d]))


def _assert_all_finite(X: np.ndarray):
    """Throw a ValueError if X contains NaN or Infinity.

    Based on Sklearn method to handle NaN & Infinity.
        @inproceedings{sklearn_api,
        author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
                    Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
                    Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
                    and Jaques Grobler and Robert Layton and Jake VanderPlas and
                    Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
        title     = {{API} design for machine learning software: experiences from the scikit-learn
                    project},
        booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
        year      = {2013},
        pages = {108--122},
        }

    Args:
        X: Array to validate

    Raises:
        ValueError: If X contains NaN or Infinity
    """

    if not np.isfinite(X).all():
        raise ValueError(
            f"Input contains NaN, infinity or a value too large for {X.dtype}."
        )
