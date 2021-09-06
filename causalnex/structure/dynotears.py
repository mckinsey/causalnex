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
Tools to learn a Dynamic Bayesian Network which describe the conditional dependencies between variables in a time-series
dataset.
"""

import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt

from causalnex.structure import StructureModel
from causalnex.structure.transformers import DynamicDataTransformer


def from_pandas_dynamic(  # pylint: disable=too-many-arguments
    time_series: Union[pd.DataFrame, List[pd.DataFrame]],
    p: int,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
    tabu_edges: List[Tuple[int, int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
) -> StructureModel:
    """
    Learn the graph structure of a Dynamic Bayesian Network describing conditional dependencies between variables in
    data. The input data is a time series or a list of realisations of a same time series.
    The optimisation is to minimise a score function F(W, A) over the graph's contemporaneous (intra-slice) weighted
    adjacency matrix, W, and lagged (inter-slice) weighted adjacency matrix, A, subject to the a constraint function
    h(W), where h_value(W) == 0 characterises an acyclic graph. h(W) > 0 is a continuous, differentiable function that
    encapsulated how acyclic the graph is (less = more acyclic).

    Based on "DYNOTEARS: Structure Learning from Time-Series Data".
    https://arxiv.org/abs/2002.00498
    @inproceedings{pamfil2020dynotears,
        title={DYNOTEARS: Structure Learning from Time-Series Data},
        author={Pamfil, Roxana and Sriwattanaworachai, Nisara and Desai, Shaan and Pilgerstorfer,
        Philip and Georgatzis, Konstantinos and Beaumont, Paul and Aragam, Bryon},
        booktitle={International Conference on Artificial Intelligence and Statistics},
        pages={1595--1605},
        year={2020}year={2020},
    }
    Args:
        time_series: pd.DataFrame or List of pd.DataFrame instances.
        If a list is provided each element of the list being an realisation of a time series (i.e. time series governed
        by the same processes)
        The columns of the data frame represent the variables in the model, and the *index represents the time index*.
        Successive events, therefore, must be indexed with one integer of difference between them too.
        p: Number of past interactions we allow the model to create. The state of a variable at time `t` is affected by
        past variables up to a `t-p`, as well as by other variables at `t`.
        lambda_w: parameter for l1 regularisation of intra-slice edges
        lambda_a: parameter for l1 regularisation of inter-slice edges
        max_iter: max number of dual ascent steps during optimisation.
        h_tol: exit if h(W) < h_tol (as opposed to strict definition of 0).
        w_threshold: fixed threshold for absolute edge weights.
        tabu_edges: list of edges(lag, from, to) not to be included in the graph. `lag == 0` implies that the edge is
        forbidden in the INTRA graph (W), while lag > 0 implies an INTER-slice weight equal zero.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

    Returns:
        StructureModel representing the model learnt. The node names are noted as `{var}_lag{l}`, where `var` is the
        original variable name as in the give in the input data frames and `l`, in 0,1,2..p is the correspondent
        time lag.
    """
    time_series = [time_series] if not isinstance(time_series, list) else time_series

    X, Xlags = DynamicDataTransformer(p=p).fit_transform(time_series, return_df=False)

    col_idx = {c: i for i, c in enumerate(time_series[0].columns)}
    idx_col = {i: c for c, i in col_idx.items()}

    if tabu_edges:
        tabu_edges = [(lag, col_idx[u], col_idx[v]) for lag, u, v in tabu_edges]
    if tabu_parent_nodes:
        tabu_parent_nodes = [col_idx[n] for n in tabu_parent_nodes]
    if tabu_child_nodes:
        tabu_child_nodes = [col_idx[n] for n in tabu_child_nodes]

    g = from_numpy_dynamic(
        X,
        Xlags,
        lambda_w,
        lambda_a,
        max_iter,
        h_tol,
        w_threshold,
        tabu_edges,
        tabu_parent_nodes,
        tabu_child_nodes,
    )

    sm = StructureModel()
    sm.add_nodes_from(
        [f"{var}_lag{l_val}" for var in col_idx.keys() for l_val in range(p + 1)]
    )
    sm.add_weighted_edges_from(
        [
            (
                _format_name_from_pandas(idx_col, u),
                _format_name_from_pandas(idx_col, v),
                w,
            )
            for u, v, w in g.edges.data("weight")
        ],
        origin="learned",
    )

    return sm


def _format_name_from_pandas(idx_col: Dict[int, str], from_numpy_node: str) -> str:
    """
    Helper function for `from_pandas_dynamic`. converts a node from the `from_numpy_dynamic` format to the `from_pandas`
    format
    Args:
        idx_col: map from variable to intdex
        from_numpy_node: nodes in the structure model output by `from_numpy_dynamic`.
    Returns:
        nodes in from_pandas_dynamic format
    """
    idx, lag_val = from_numpy_node.split("_lag")
    return f"{idx_col[int(idx)]}_lag{lag_val}"


def from_numpy_dynamic(  # pylint: disable=too-many-arguments
    X: np.ndarray,
    Xlags: np.ndarray,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
    tabu_edges: List[Tuple[int, int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
) -> StructureModel:
    """
    Learn the graph structure of a Dynamic Bayesian Network describing conditional dependencies between variables in
    data. The input data is time series data present in numpy arrays X and Xlags.

    The optimisation is to minimise a score function F(W, A) over the graph's contemporaneous (intra-slice) weighted
    adjacency matrix, W, and lagged (inter-slice) weighted adjacency matrix, A, subject to the a constraint function
    h(W), where h_value(W) == 0 characterises an acyclic graph. h(W) > 0 is a continuous, differentiable function that
    encapsulated how acyclic the graph is (less = more acyclic).

    Based on "DYNOTEARS: Structure Learning from Time-Series Data".
    https://arxiv.org/abs/2002.00498
    @inproceedings{pamfil2020dynotears,
        title={DYNOTEARS: Structure Learning from Time-Series Data},
        author={Pamfil, Roxana and Sriwattanaworachai, Nisara and Desai, Shaan and Pilgerstorfer,
        Philip and Georgatzis, Konstantinos and Beaumont, Paul and Aragam, Bryon},
        booktitle={International Conference on Artificial Intelligence and Statistics},
        pages={1595--1605},
        year={2020}year={2020},
    }

    Args:
        X (np.ndarray): 2d input data, axis=1 is data columns, axis=0 is data rows. Each column represents one variable,
        and each row represents x(m,t) i.e. the mth time series at time t.
        Xlags (np.ndarray): shifted data of X with lag orders stacking horizontally. Xlags=[shift(X,1)|...|shift(X,p)]
        lambda_w (float): l1 regularization parameter of intra-weights W
        lambda_a (float): l1 regularization parameter of inter-weights A
        max_iter: max number of dual ascent steps during optimisation
        h_tol (float): exit if h(W) < h_tol (as opposed to strict definition of 0)
        w_threshold: fixed threshold for absolute edge weights.
        tabu_edges: list of edges(lag, from, to) not to be included in the graph. `lag == 0` implies that the edge is
        forbidden in the INTRA graph (W), while lag > 0 implies an INTER weight equal zero.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.
    Returns:
        W (np.ndarray): d x d estimated weighted adjacency matrix of intra slices
        A (np.ndarray): d x pd estimated weighted adjacency matrix of inter slices

    Raises:
        ValueError: If X or Xlags does not contain data, or dimensions of X and Xlags do not conform
    """
    _, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    bnds_w = 2 * [
        (0, 0)
        if i == j
        else (0, 0)
        if tabu_edges is not None and (0, i, j) in tabu_edges
        else (0, 0)
        if tabu_parent_nodes is not None and i in tabu_parent_nodes
        else (0, 0)
        if tabu_child_nodes is not None and j in tabu_child_nodes
        else (0, None)
        for i in range(d_vars)
        for j in range(d_vars)
    ]

    bnds_a = []
    for k in range(1, p_orders + 1):
        bnds_a.extend(
            2
            * [
                (0, 0)
                if tabu_edges is not None and (k, i, j) in tabu_edges
                else (0, 0)
                if tabu_parent_nodes is not None and i in tabu_parent_nodes
                else (0, 0)
                if tabu_child_nodes is not None and j in tabu_child_nodes
                else (0, None)
                for i in range(d_vars)
                for j in range(d_vars)
            ]
        )

    bnds = bnds_w + bnds_a
    w_est, a_est = _learn_dynamic_structure(
        X, Xlags, bnds, lambda_w, lambda_a, max_iter, h_tol
    )

    w_est[np.abs(w_est) < w_threshold] = 0
    a_est[np.abs(a_est) < w_threshold] = 0
    sm = _matrices_to_structure_model(w_est, a_est)
    return sm


def _matrices_to_structure_model(
    w_est: np.ndarray, a_est: np.ndarray
) -> StructureModel:
    """
    Converts the matrices output by dynotears (W and A) into a StructureModel
    We use the following convention:
    - {var}_lag{l} where l is the lag value (i.e. from how many previous timestamps the edge is coming
    - if we deal with a intra_slice_node, `l == 0`
    Args:
        w_est: Intra-slice weight matrix
        a_est: Inter-slice matrix

    Returns:
        StructureModel representing the structure learnt

    """
    sm = StructureModel()
    lag_cols = [
        f"{var}_lag{l_val}"
        for l_val in range(1 + (a_est.shape[0] // a_est.shape[1]))
        for var in range(a_est.shape[1])
    ]
    sm.add_nodes_from(lag_cols)
    sm.add_edges_from(
        [
            (lag_cols[i], lag_cols[j], dict(weight=w_est[i, j]))
            for i in range(w_est.shape[0])
            for j in range(w_est.shape[1])
            if w_est[i, j] != 0
        ]
    )
    sm.add_edges_from(
        [
            (lag_cols[i + w_est.shape[0]], lag_cols[j], dict(weight=a_est[i, j]))
            for i in range(a_est.shape[0])
            for j in range(a_est.shape[1])
            if a_est[i, j] != 0
        ]
    )
    return sm


def _reshape_wa(
    wa_vec: np.ndarray, d_vars: int, p_orders: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function for `_learn_dynamic_structure`. Transform adjacency vector to matrix form

    Args:
        wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights
        d_vars (int): number of variables in the model
        p_orders (int): number of past indexes we to use
    Returns:
        intra- and inter-slice adjacency matrices
    """

    w_tilde = wa_vec.reshape([2 * (p_orders + 1) * d_vars, d_vars])
    w_plus = w_tilde[:d_vars, :]
    w_minus = w_tilde[d_vars : 2 * d_vars, :]
    w_mat = w_plus - w_minus
    a_plus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars ** 2)[::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_minus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars ** 2)[1::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_mat = a_plus - a_minus
    return w_mat, a_mat


def _learn_dynamic_structure(
    X: np.ndarray,
    Xlags: np.ndarray,
    bnds: List[Tuple[float, float]],
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learn the graph structure of a Dynamic Bayesian Network describing conditional dependencies between data variables.

    The optimisation is to minimise a score function F(W, A) over the graph's contemporaneous (intra-slice) weighted
    adjacency matrix, W, and lagged (inter-slice) weighted adjacency matrix, A, subject to the a constraint function
    h(W), where h_value(W) == 0 characterises an acyclic graph. h(W) > 0 is a continuous, differentiable function that
    encapsulated how acyclic the graph is (less = more acyclic).

    Based on "DYNOTEARS: Structure Learning from Time-Series Data".
    https://arxiv.org/abs/2002.00498
    @inproceedings{pamfil2020dynotears,
        title={DYNOTEARS: Structure Learning from Time-Series Data},
        author={Pamfil, Roxana and Sriwattanaworachai, Nisara and Desai, Shaan and Pilgerstorfer,
        Philip and Georgatzis, Konstantinos and Beaumont, Paul and Aragam, Bryon},
        booktitle={International Conference on Artificial Intelligence and Statistics},
        pages={1595--1605},
        year={2020}year={2020},
    }

    Args:
        X (np.ndarray): 2d input data, axis=1 is data columns, axis=0 is data rows. Each column represents one variable,
        and each row represents x(m,t) i.e. the mth time series at time t.
        Xlags (np.ndarray): shifted data of X with lag orders stacking horizontally. Xlags=[shift(X,1)|...|shift(X,p)]
        bnds: Box constraints of L-BFGS-B to ban self-loops in W, enforce non-negativity of w_plus, w_minus, a_plus,
        a_minus, and help with stationarity in A
        lambda_w (float): l1 regularization parameter of intra-weights W
        lambda_a (float): l1 regularization parameter of inter-weights A
        max_iter (int): max number of dual ascent steps during optimisation
        h_tol (float): exit if h(W) < h_tol (as opposed to strict definition of 0)

    Returns:
        W (np.ndarray): d x d estimated weighted adjacency matrix of intra slices
        A (np.ndarray): d x pd estimated weighted adjacency matrix of inter slices

    Raises:
        ValueError: If X or Xlags does not contain data, or dimensions of X and Xlags do not conform
    """
    if X.size == 0:
        raise ValueError("Input data X is empty, cannot learn any structure")
    if Xlags.size == 0:
        raise ValueError("Input data Xlags is empty, cannot learn any structure")
    if X.shape[0] != Xlags.shape[0]:
        raise ValueError("Input data X and Xlags must have the same number of rows")
    if Xlags.shape[1] % X.shape[1] != 0:
        raise ValueError(
            "Number of columns of Xlags must be a multiple of number of columns of X"
        )

    n, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    def _h(wa_vec: np.ndarray) -> float:
        """
        Constraint function of the dynotears

        Args:
            wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

        Returns:
            float: DAGness of the intra-slice adjacency matrix W (0 == DAG, >0 == cyclic)
        """

        _w_mat, _ = _reshape_wa(wa_vec, d_vars, p_orders)
        return np.trace(slin.expm(_w_mat * _w_mat)) - d_vars

    def _func(wa_vec: np.ndarray) -> float:
        """
        Objective function that the dynotears tries to minimise

        Args:
            wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

        Returns:
            float: objective
        """

        _w_mat, _a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        loss = (
            0.5
            / n
            * np.square(
                np.linalg.norm(
                    X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat), "fro"
                )
            )
        )
        _h_value = _h(wa_vec)
        l1_penalty = lambda_w * (wa_vec[: 2 * d_vars ** 2].sum()) + lambda_a * (
            wa_vec[2 * d_vars ** 2 :].sum()
        )
        return loss + 0.5 * rho * _h_value * _h_value + alpha * _h_value + l1_penalty

    def _grad(wa_vec: np.ndarray) -> np.ndarray:
        """
        Gradient function used to compute next step in dynotears

        Args:
            wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

        Returns:
            gradient vector
        """

        _w_mat, _a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        e_mat = slin.expm(_w_mat * _w_mat)
        loss_grad_w = (
            -1.0
            / n
            * (X.T.dot(X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat)))
        )
        obj_grad_w = (
            loss_grad_w
            + (rho * (np.trace(e_mat) - d_vars) + alpha) * e_mat.T * _w_mat * 2
        )
        obj_grad_a = (
            -1.0
            / n
            * (Xlags.T.dot(X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat)))
        )

        grad_vec_w = np.append(
            obj_grad_w, -obj_grad_w, axis=0
        ).flatten() + lambda_w * np.ones(2 * d_vars ** 2)
        grad_vec_a = obj_grad_a.reshape(p_orders, d_vars ** 2)
        grad_vec_a = np.hstack(
            (grad_vec_a, -grad_vec_a)
        ).flatten() + lambda_a * np.ones(2 * p_orders * d_vars ** 2)
        return np.append(grad_vec_w, grad_vec_a, axis=0)

    # initialise matrix, weights and constraints
    wa_est = np.zeros(2 * (p_orders + 1) * d_vars ** 2)
    wa_new = np.zeros(2 * (p_orders + 1) * d_vars ** 2)
    rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf

    for n_iter in range(max_iter):
        while (rho < 1e20) and (h_new > 0.25 * h_value or h_new == np.inf):
            wa_new = sopt.minimize(
                _func, wa_est, method="L-BFGS-B", jac=_grad, bounds=bnds
            ).x
            h_new = _h(wa_new)
            if h_new > 0.25 * h_value:
                rho *= 10

        wa_est = wa_new
        h_value = h_new
        alpha += rho * h_value
        if h_value <= h_tol:
            break
        if h_value > h_tol and n_iter == max_iter - 1:
            warnings.warn("Failed to converge. Consider increasing max_iter.")
    return _reshape_wa(wa_est, d_vars, p_orders)
