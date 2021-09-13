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
Module of methods to sample variables of a single data type.
"""
import warnings
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.gaussian_process.kernels import Kernel

from causalnex.structure.data_generators import (
    generate_structure,
    nonlinear_sem_generator,
    sem_generator,
)
from causalnex.structure.structuremodel import StructureModel


def generate_continuous_data(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "gaussian",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
    kernel: Optional[Kernel] = None,
) -> np.ndarray:
    """
    Simulate samples from SEM with specified type of noise.
    The order of the columns on the returned array is the one provided by `sm.nodes`

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        kernel: A kernel from sklearn.gaussian_process.kernels like RBF(1) or
            Matern(1) or any combinations thereof. The kernels are used to
            create the latent variable for the binary / categorical variables
            and are directly used for continuous variables.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'gaussian'/'normal' (alias), 'student-t',
            'exponential', 'gumbel'.
        noise_scale: The standard deviation of the noise.
        intercept: Whether to use an intercept for each feature.
        seed: Random state
    Returns:
        x_mat: [n_samples,d_nodes] sample matrix
    Raises:
        ValueError: if distribution isn't gaussian/normal/student-t/exponential/gumbel
    """
    if kernel is None:
        df = sem_generator(
            graph=sm,
            default_type="continuous",
            n_samples=n_samples,
            distributions={"continuous": distribution},
            noise_std=noise_scale,
            intercept=intercept,
            seed=seed,
        )
    else:
        df = nonlinear_sem_generator(
            graph=sm,
            kernel=kernel,
            default_type="continuous",
            n_samples=n_samples,
            distributions={"continuous": distribution},
            noise_std=noise_scale,
            seed=seed,
        )
    return df[list(sm.nodes())].values


def generate_binary_data(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "logit",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
    kernel: Optional[Kernel] = None,
) -> np.ndarray:
    """
    Simulate samples from SEM with specified type of noise.
    The order of the columns on the returned array is the one provided by `sm.nodes`

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        kernel: A kernel from sklearn.gaussian_process.kernels like RBF(1) or
            Matern(1) or any combinations thereof. The kernels are used to
            create the latent variable for the binary / categorical variables
            and are directly used for continuous variables.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'probit'/'normal' (alias),
            'logit' (default).
        noise_scale: The standard deviation of the noise. The binary and
            categorical features are created using a latent variable approach.
            The noise standard deviation determines how much weight the "mean"
            estimate has on the feature value.
        intercept: Whether to use an intercept for the latent variable of each feature.
        seed: Random state
    Returns:
        x_mat: [n_samples,d_nodes] sample matrix
    Raises:
        ValueError: if distribution isn't 'probit', 'normal', 'logit'
    """
    if kernel is None:
        df = sem_generator(
            graph=sm,
            default_type="binary",
            n_samples=n_samples,
            distributions={"binary": distribution},
            noise_std=noise_scale,
            intercept=intercept,
            seed=seed,
        )
    else:
        df = nonlinear_sem_generator(
            graph=sm,
            kernel=kernel,
            default_type="binary",
            n_samples=n_samples,
            distributions={"binary": distribution},
            noise_std=noise_scale,
            seed=seed,
        )
    return df[list(sm.nodes())].values


def generate_continuous_dataframe(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "gaussian",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
    kernel: Optional[Kernel] = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.
    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        kernel: A kernel from sklearn.gaussian_process.kernels like RBF(1) or
            Matern(1) or any combinations thereof. The kernels are used to
            create the latent variable for the binary / categorical variables
            and are directly used for continuous variables.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'gaussian'/'normal' (alias), 'student-t',
            'exponential', 'gumbel'.
        noise_scale: The standard deviation of the noise.
        intercept: Whether to use an intercept for each feature.
        seed: Random state
    Returns:
        Dataframe with the node names as column names
    Raises:
        ValueError: if distribution is not 'gaussian', 'normal', 'student-t',
            'exponential', 'gumbel'
    """
    if kernel is None:
        return sem_generator(
            graph=sm,
            default_type="continuous",
            n_samples=n_samples,
            distributions={"continuous": distribution},
            noise_std=noise_scale,
            intercept=intercept,
            seed=seed,
        )

    return nonlinear_sem_generator(
        graph=sm,
        kernel=kernel,
        default_type="continuous",
        n_samples=n_samples,
        distributions={"continuous": distribution},
        noise_std=noise_scale,
        seed=seed,
    )


def generate_binary_dataframe(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "logit",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
    kernel: Optional[Kernel] = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        kernel: A kernel from sklearn.gaussian_process.kernels like RBF(1) or
            Matern(1) or any combinations thereof. The kernels are used to
            create the latent variable for the binary / categorical variables
            and are directly used for continuous variables.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'probit'/'normal' (alias),
            'logit' (default).
        noise_scale: The standard deviation of the noise. The binary and
            categorical features are created using a latent variable approach.
            The noise standard deviation determines how much weight the "mean"
            estimate has on the feature value.
        intercept: Whether to use an intercept for the latent variable of each feature.
        seed: Random state
    Returns:
        x_mat: [n_samples,d_nodes] sample matrix
    Raises:
        ValueError: if distribution is not 'probit', 'normal', 'logit'
    """
    if kernel is None:
        return sem_generator(
            graph=sm,
            default_type="binary",
            n_samples=n_samples,
            distributions={"binary": distribution},
            noise_std=noise_scale,
            intercept=intercept,
            seed=seed,
        )

    return nonlinear_sem_generator(
        graph=sm,
        kernel=kernel,
        default_type="binary",
        n_samples=n_samples,
        distributions={"binary": distribution},
        noise_std=noise_scale,
        seed=seed,
    )


def generate_count_dataframe(
    sm: nx.DiGraph,
    n_samples: int,
    zero_inflation_factor: float = 0.1,
    intercept: bool = False,
    seed: int = None,
    kernel: Optional[Kernel] = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        kernel: A kernel from sklearn.gaussian_process.kernels like RBF(1) or
            Matern(1) or any combinations thereof. The kernels are used to
            create the latent variable for the binary / categorical variables
            and are directly used for continuous variables.
        zero_inflation_factor: The probability of zero inflation for count data.
        intercept: Whether to use an intercept for the latent variable of each feature.
        seed: Random state
    Returns:
        x_mat: [n_samples, d_nodes] sample matrix
    Raises:
        ValueError: if ``zero_inflation_factor`` is not a float in [0, 1].
    """

    if kernel is None:
        return sem_generator(
            graph=sm,
            default_type="count",
            n_samples=n_samples,
            distributions={"count": zero_inflation_factor},
            noise_std=1,  # not used for poisson
            intercept=intercept,
            seed=seed,
        )

    return nonlinear_sem_generator(
        graph=sm,
        kernel=kernel,
        default_type="count",
        n_samples=n_samples,
        distributions={"count": zero_inflation_factor},
        noise_std=1,  # not used for poisson
        seed=seed,
    )


def generate_categorical_dataframe(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "logit",
    n_categories: int = 3,
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
    kernel: Optional[Kernel] = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        kernel: A kernel from sklearn.gaussian_process.kernels like RBF(1) or
            Matern(1) or any combinations thereof. The kernels are used to
            create the latent variable for the binary / categorical variables
            and are directly used for continuous variables.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'probit'/'normal' (alias),
            "logit"/"gumbel" (alias). Logit is default.
        n_categories: Number of categories per variable/node.
        noise_scale: The standard deviation of the noise. The categorical features
            are created using a latent variable approach. The noise standard
            deviation determines how much weight the "mean" estimate has on
            the feature value.
        intercept: Whether to use an intercept for the latent variable of each feature.
        seed: Random state
    Returns:
        x_mat: [n_samples, d_nodes] sample matrix
    Raises:
        ValueError: if distribution is not 'probit', 'normal', 'logit', 'gumbel'
    """

    if kernel is None:
        return sem_generator(
            graph=sm,
            default_type=f"categorical:{n_categories}",
            n_samples=n_samples,
            distributions={"categorical": distribution},
            noise_std=noise_scale,
            intercept=intercept,
            seed=seed,
        )

    return nonlinear_sem_generator(
        graph=sm,
        kernel=kernel,
        default_type=f"categorical:{n_categories}",
        n_samples=n_samples,
        distributions={"categorical": distribution},
        noise_std=noise_scale,
        seed=seed,
    )


def generate_structure_dynamic(  # pylint: disable=too-many-arguments
    num_nodes: int,
    p: int,
    degree_intra: float,
    degree_inter: float,
    graph_type_intra: str = "erdos-renyi",
    graph_type_inter: str = "erdos-renyi",
    w_min_intra: float = 0.5,
    w_max_intra: float = 0.5,
    w_min_inter: float = 0.5,
    w_max_inter: float = 0.5,
    w_decay: float = 1.0,
) -> StructureModel:
    """
    Generates a dynamic DAG at random.

    Args:
        num_nodes: Number of nodes
        p: maximum lag to be considered in the structure
        degree_intra: expected degree on nodes from the current state
        degree_inter: expected degree on nodes from the lagged nodes
        graph_type_intra:
            - erdos-renyi: constructs a graph such that the probability of any given edge is degree / (num_nodes - 1)
            - barabasi-albert: constructs a scale-free graph from an initial connected graph of (degree / 2) nodes
            - full: constructs a fully-connected graph - degree has no effect
        graph_type_inter:
            - erdos-renyi: constructs a graph such that the probability of any given edge is degree / (num_nodes - 1)
            - full: connect all past nodes to all present nodes
        w_min_intra: minimum weight for intra-slice nodes
        w_max_intra: maximum weight for intra-slice nodes
        w_min_inter: minimum weight for inter-slice nodes
        w_max_inter: maximum weight for inter-slice nodes
        w_decay: exponent of weights decay for slices that are farther apart. Default is 1.0, which implies no decay

    Raises:
        ValueError: if graph type unknown or `num_nodes < 2`

    Returns:
        StructureModel containing all simulated nodes and edges (intra- and inter-slice)
    """
    sm_intra = generate_structure(
        num_nodes=num_nodes,
        degree=degree_intra,
        graph_type=graph_type_intra,
        w_min=w_min_intra,
        w_max=w_max_intra,
    )
    sm_inter = _generate_inter_structure(
        num_nodes=num_nodes,
        p=p,
        degree=degree_inter,
        graph_type=graph_type_inter,
        w_min=w_min_inter,
        w_max=w_max_inter,
        w_decay=w_decay,
    )
    res = StructureModel()
    res.add_nodes_from(sm_inter.nodes)
    res.add_nodes_from([f"{u}_lag0" for u in sm_intra.nodes])
    res.add_weighted_edges_from(sm_inter.edges.data("weight"))
    res.add_weighted_edges_from(
        [(f"{u}_lag0", f"{v}_lag0", w) for u, v, w in sm_intra.edges.data("weight")]
    )
    return res


def _generate_inter_structure(
    num_nodes: int,
    p: int,
    degree: float,
    graph_type: str,
    w_min: float,
    w_max: float,
    w_decay: float = 1.0,
    neg: float = 0.5,
) -> StructureModel:
    """Simulate random DAG between two time slices.

    Args:
        num_nodes: number of nodes per slice
        p: number of slices that influence current slice
        degree: expected in-degree of current time slice
        graph_type: {'erdos-renyi' 'full'}
        w_min: minimum weight for inter-slice nodes
        w_max: maximum weight for inter-slice nodes
        w_decay: exponent of weights decay for slices that are farther apart. Default is 1.0, which implies no decay
        neg: the proportion of edge weights expected to be negative. By default, 50% of the edges are expected
            to be negative weight (`neg == 0.5`).

    Returns:
        G_inter: weighted, bipartite DAG for inter-slice connections

    Raises:
        ValueError: if graph type not known
    """
    if w_min > w_max:
        raise ValueError(
            "Absolute minimum weight must be less than or equal to maximum weight: "
            f"{w_min} > {w_max}"
        )

    if graph_type == "erdos-renyi":
        prob = degree / num_nodes
        b = (np.random.rand(p * num_nodes, num_nodes) < prob).astype(float)
    elif graph_type == "full":  # ignore degree, only for experimental use
        b = np.ones([p * num_nodes, num_nodes])
    else:
        raise ValueError(
            f"Unknown inter-slice graph type `{graph_type}`. "
            "Valid types are 'erdos-renyi' and 'full'"
        )
    u = []
    for i in range(p):
        u_i = np.random.uniform(low=w_min, high=w_max, size=[num_nodes, num_nodes]) / (
            w_decay ** i
        )
        u_i[np.random.rand(num_nodes, num_nodes) < neg] *= -1
        u.append(u_i)

    u = np.concatenate(u, axis=0) if u else np.empty(b.shape)
    a = (b != 0).astype(float) * u

    df = pd.DataFrame(
        a,
        index=[
            f"{var}_lag{l_val}" for l_val in range(1, p + 1) for var in range(num_nodes)
        ],
        columns=[f"{var}_lag0" for var in range(num_nodes)],
    )
    idxs, cols = list(df.index), list(df.columns)
    for i in idxs:
        df[i] = 0
    for i in cols:
        df.loc[i, :] = 0

    g_inter = StructureModel(df)
    return g_inter


def generate_dataframe_dynamic(  # pylint: disable=R0914
    g: StructureModel,
    n_samples: int = 1000,
    burn_in: int = 100,
    sem_type: str = "linear-gauss",
    noise_scale: float = 1.0,
    drift: np.ndarray = None,
) -> pd.DataFrame:
    """Simulate samples from dynamic SEM with specified type of noise.
    Args:
        g: Dynamic DAG
        n_samples: number of samples
        burn_in: number of samples to discard
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM
        drift: array of drift terms for each node, if None then the drift is 0
    Returns:
        X: [n,d] sample matrix, row t is X_t
        Y: [n,d*p] sample matrix, row t is [X_{t-1}, ..., X_{t-p}]
    Raises:
        ValueError: if sem_type isn't linear-gauss/linear_exp/linear-gumbel
    """
    s_types = ("linear-gauss", "linear-exp", "linear-gumbel")
    if sem_type not in s_types:
        raise ValueError(f"unknown sem type {sem_type}. Available types are: {s_types}")
    intra_nodes = sorted(el for el in g.nodes if "_lag0" in el)
    inter_nodes = sorted(el for el in g.nodes if "_lag0" not in el)
    w_mat = nx.to_numpy_array(g, nodelist=intra_nodes)
    a_mat = nx.to_numpy_array(g, nodelist=intra_nodes + inter_nodes)[
        len(intra_nodes) :, : len(intra_nodes)
    ]
    g_intra = nx.DiGraph(w_mat)
    g_inter = nx.bipartite.from_biadjacency_matrix(
        csr_matrix(a_mat), create_using=nx.DiGraph
    )
    d = w_mat.shape[0]
    p = a_mat.shape[0] // d
    total_length = n_samples + burn_in
    X = np.zeros([total_length, d])
    Xlags = np.zeros([total_length, p * d])
    ordered_vertices = list(nx.topological_sort(g_intra))
    if drift is None:
        drift = np.zeros(d)
    for t in range(total_length):
        for j in ordered_vertices:
            parents = list(g_intra.predecessors(j))
            parents_prev = list(g_inter.predecessors(j + p * d))
            X[t, j] = (
                drift[j]
                + X[t, parents].dot(w_mat[parents, j])
                + Xlags[t, parents_prev].dot(a_mat[parents_prev, j])
            )
            if sem_type == "linear-gauss":
                X[t, j] = X[t, j] + np.random.normal(scale=noise_scale)
            elif sem_type == "linear-exp":
                X[t, j] = X[t, j] + np.random.exponential(scale=noise_scale)
            elif sem_type == "linear-gumbel":
                X[t, j] = X[t, j] + np.random.gumbel(scale=noise_scale)

        if (t + 1) < total_length:
            Xlags[t + 1, :] = np.concatenate([X[t, :], Xlags[t, :]])[: d * p]
    return pd.concat(
        [
            pd.DataFrame(X[-n_samples:], columns=intra_nodes),
            pd.DataFrame(Xlags[-n_samples:], columns=inter_nodes),
        ],
        axis=1,
    )


def gen_stationary_dyn_net_and_df(  # pylint: disable=R0913, R0914
    num_nodes: int = 10,
    n_samples: int = 100,
    p: int = 1,
    degree_intra: float = 3,
    degree_inter: float = 3,
    graph_type_intra: str = "erdos-renyi",
    graph_type_inter: str = "erdos-renyi",
    w_min_intra: float = 0.5,
    w_max_intra: float = 0.5,
    w_min_inter: float = 0.5,
    w_max_inter: float = 0.5,
    w_decay: float = 1.0,
    sem_type: str = "linear-gauss",
    noise_scale: float = 1,
    max_data_gen_trials: int = 1000,
) -> Tuple[StructureModel, pd.DataFrame, List[str], List[str]]:
    """
    Generates a dynamic structure model as well a dataframe representing a time series realisation of that model.
    We do checks to verify the network is stationary, and iterate until the resulting network is stationary.
    Args:
        num_nodes: number of nodes in the intra-slice structure
        n_samples: number of points to sample from the model, as a time series
        p: lag value for the dynamic structure
        degree_intra: expected degree for intra_slice nodes
        degree_inter: expected degree for inter_slice nodes
        graph_type_intra:
            - erdos-renyi: constructs a graph such that the probability of any given edge is degree / (num_nodes - 1)
            - barabasi-albert: constructs a scale-free graph from an initial connected graph of (degree / 2) nodes
            - full: constructs a fully-connected graph - degree has no effect
        graph_type_inter:
            - erdos-renyi: constructs a graph such that the probability of any given edge is degree / (num_nodes - 1)
            - full: connect all past nodes to all present nodesw_min_intra:
        w_min_intra: minimum weight on intra-slice adjacency matrix
        w_max_intra: maximum weight on intra-slice adjacency matrix
        w_min_inter: minimum weight on inter-slice adjacency matrix
        w_max_inter: maximum weight on inter-slice adjacency matrix
        w_decay: exponent of weights decay for slices that are farther apart. Default is 1.0, which implies no decay
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM
        max_data_gen_trials: maximun number of attempts until obtaining a seemingly stationary model
    Returns:
        Tuple with:
        - the model created,as a Structure model
        - DataFrame representing the time series created from the model
        - Intra-slice nodes names
        - Inter-slice nodes names
    """

    with np.errstate(over="raise", invalid="raise"):
        burn_in = max(n_samples // 10, 50)

        simulate_flag = True
        g, intra_nodes, inter_nodes = None, None, None

        while simulate_flag:
            max_data_gen_trials -= 1
            if max_data_gen_trials <= 0:
                simulate_flag = False

            try:
                simulate_graphs_flag = True
                while simulate_graphs_flag:

                    g = generate_structure_dynamic(
                        num_nodes=num_nodes,
                        p=p,
                        degree_intra=degree_intra,
                        degree_inter=degree_inter,
                        graph_type_intra=graph_type_intra,
                        graph_type_inter=graph_type_inter,
                        w_min_intra=w_min_intra,
                        w_max_intra=w_max_intra,
                        w_min_inter=w_min_inter,
                        w_max_inter=w_max_inter,
                        w_decay=w_decay,
                    )
                    intra_nodes = sorted([el for el in g.nodes if "_lag0" in el])
                    inter_nodes = sorted([el for el in g.nodes if "_lag0" not in el])
                    # Exclude empty graphs from consideration unless input degree is 0
                    if (
                        (
                            [(u, v) for u, v in g.edges if u in intra_nodes]
                            and [(u, v) for u, v in g.edges if u in inter_nodes]
                        )
                        or degree_intra == 0
                        or degree_inter == 0
                    ):
                        simulate_graphs_flag = False

                # generate single time series
                df = (
                    generate_dataframe_dynamic(
                        g,
                        n_samples=n_samples + burn_in,
                        sem_type=sem_type,
                        noise_scale=noise_scale,
                    )
                    .loc[burn_in:, intra_nodes + inter_nodes]
                    .reset_index(drop=True)
                )

                if df.isna().any(axis=None):
                    continue
            except (OverflowError, FloatingPointError):
                continue
            if (df.abs().max().max() < 1e3) or (max_data_gen_trials <= 0):
                simulate_flag = False
        if max_data_gen_trials <= 0:
            warnings.warn(
                "Could not simulate data, returning constant dataframe", UserWarning
            )

            df = pd.DataFrame(
                np.ones((n_samples, num_nodes * (1 + p))),
                columns=intra_nodes + inter_nodes,
            )
    return g, df, intra_nodes, inter_nodes
