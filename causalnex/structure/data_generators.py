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
Module of methods to generate random StructureModel and datasets with various properties, for example, continuous data.
Structure generator based on implementation found in: from https://github.com/xunzheng/notears
git hash: 31923cb22517f7bb6420dd0b6ef23ca550702b97
"""
import warnings
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from causalnex.structure.categorical_variable_mapper import (
    VariableFeatureMapper,
    validate_schema,
)
from causalnex.structure.structuremodel import StructureModel


def generate_structure(
    num_nodes: int,
    degree: float,
    graph_type: str = "erdos-renyi",
    w_min: float = 0.5,
    w_max: float = 0.5,
) -> StructureModel:
    """Simulate random DAG with some expected degree.
    Notes:
        graph_type (str):
            - erdos-renyi: constructs a graph such that the probability of any given edge is degree / (num_nodes - 1)
            - barabasi-albert: constructs a scale-free graph from an initial connected graph of (degree / 2) nodes
            - full: constructs a fully-connected graph - degree has no effect
    Args:
        num_nodes: number of nodes
        degree: expected node degree, in + out
        graph_type (str):
            - erdos-renyi: constructs a graph such that the probability of any given edge is degree / (num_nodes - 1)
            - barabasi-albert: constructs a scale-free graph from an initial connected graph of (degree / 2) nodes
            - full: constructs a fully-connected graph - degree has no effect
        w_min (float): min absolute weight of an edge in the graph
        w_max (float): max absolute weight of an edge in the graph
    Raises:
        ValueError: if invalid arguments are provided
    Returns:
        weighted DAG
    """

    if num_nodes < 2:
        raise ValueError("DAG must have at least 2 nodes")

    w_min, w_max = abs(w_min), abs(w_max)

    if w_min > w_max:
        raise ValueError(
            "Absolute minimum weight must be less than or equal to maximum weight: {} > {}".format(
                w_min, w_max
            )
        )

    if graph_type == "erdos-renyi":
        p_threshold = float(degree) / (num_nodes - 1)
        p_edge = (np.random.rand(num_nodes, num_nodes) < p_threshold).astype(float)
        edge_flags = np.tril(p_edge, k=-1)

    elif graph_type == "barabasi-albert":
        m = int(round(degree / 2))
        edge_flags = np.zeros([num_nodes, num_nodes])
        bag = [0]
        for i in range(1, num_nodes):
            dest = np.random.choice(bag, size=m)
            for j in dest:
                edge_flags[i, j] = 1
            bag.append(i)
            bag.extend(dest)

    elif graph_type == "full":  # ignore degree
        edge_flags = np.tril(np.ones([num_nodes, num_nodes]), k=-1)

    else:
        raise ValueError("unknown graph type")

    # randomly permute edges - required because we limited ourselves to lower diagonal previously
    perms = np.random.permutation(np.eye(num_nodes, num_nodes))
    edge_flags = perms.T.dot(edge_flags).dot(perms)

    # random edge weights between w_min, w_max or between -w_min, -w_max
    edge_weights = np.random.uniform(low=w_min, high=w_max, size=[num_nodes, num_nodes])
    edge_weights[np.random.rand(num_nodes, num_nodes) < 0.5] *= -1

    adj_matrix = (edge_flags != 0).astype(float) * edge_weights
    graph = StructureModel(adj_matrix)
    return graph


def generate_continuous_data(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "gaussian",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate samples from SEM with specified type of noise.
    The order of the columns on the returned array is the one provided by `sm.nodes`
    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
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
    df = sem_generator(
        graph=sm,
        default_type="continuous",
        n_samples=n_samples,
        distributions={"continuous": distribution},
        noise_std=noise_scale,
        intercept=intercept,
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
) -> np.ndarray:
    """
    Simulate samples from SEM with specified type of noise.
    The order of the columns on the returned array is the one provided by `sm.nodes`
    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
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
    df = sem_generator(
        graph=sm,
        default_type="binary",
        n_samples=n_samples,
        distributions={"binary": distribution},
        noise_std=noise_scale,
        intercept=intercept,
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
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.
    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
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
    return sem_generator(
        graph=sm,
        default_type="continuous",
        n_samples=n_samples,
        distributions={"continuous": distribution},
        noise_std=noise_scale,
        intercept=intercept,
        seed=seed,
    )


def generate_binary_dataframe(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "logit",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.
    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
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
    return sem_generator(
        graph=sm,
        default_type="binary",
        n_samples=n_samples,
        distributions={"binary": distribution},
        noise_std=noise_scale,
        intercept=intercept,
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
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.
    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
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
    return sem_generator(
        graph=sm,
        default_type="categorical:{}".format(n_categories),
        n_samples=n_samples,
        distributions={"categorical": distribution},
        noise_std=noise_scale,
        intercept=intercept,
        seed=seed,
    )


def sem_generator(
    graph: nx.DiGraph,
    schema: Optional[Dict] = None,
    default_type: str = "continuous",
    noise_std: float = 1.0,
    n_samples: int = 1000,
    distributions: Dict[str, str] = None,
    intercept: bool = True,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generator for tabular data with mixed variable types from a DAG.
    Supported variable types: `'binary', 'categorical', 'continuous'`. The number
    of categories can be determined using a colon, e.g. `'categorical:5'`
    specifies a categorical feature with 5 categories.
    Notation: For binary and continuous variables, a ``variable'' refers to a
    ``node'', a ``feature'' refers to the one-hot column for categorical
    variables and is equivalent to a binary or continuous variable.
    Args:
        graph: A DAG in form of a networkx or StructureModel.
        schema: Dictionary with schema for a node/variable, if a node is missing
            uses ``default_type``. Format, {node_name: variable type}.
        default_type: The default data type for a node/variable not listed
            in the schema, or when the schema is empty.
        noise_std: The standard deviation of the noise. The binary and
            categorical features are created using a latent variable approach.
            The noise standard deviation determines how much weight the "mean"
            estimate has on the feature value.
        n_samples: The number of rows/observations to sample.
        distributions:
            ``continuous'': The type of distribution to use for the noise
                of a continuous variable. Options: 'gaussian'/'normal' (alias)
                (default), 'student-t', 'exponential', 'gumbel'.
            ``binary'': The type of distribution to use for the noise
                of the latent binary variable. Options: 'probit'/'normal' (alias),
                'logit' (default).
            ``categorical'': The type of distribution to use for the noise
                of a latent continuous feature. Options: 'probit'/'normal' (alias),
                'logit'/'gumbel' (alias) (default).
            ``weight'': The type of distribution to use for the linear coefficients.
                Options: 'gaussian'/'normal' (alias), 'uniform' (default).
            ``intercept'': The type of distribution to use for the intercept. For
                binary/categorical: this is the mean in the latent space.
                Options: 'gaussian'/'normal' (alias), 'uniform' (default).
        intercept: Whether to use an intercept for each feature. The intercept
            is sampled once and held constant for all rows. For binary or
            categorical the intercept determines the class imbalance.
        seed: Random State
    Returns:
        DataFrame with generated features, uses a one-hot coding for
        categorical features.
    Raises:
        ValueError: if the graph is not a DAG.
        ValueError: if schema variable type is not in `'binary', 'categorical',
            'continuous', 'continuous:X` (for variables with X categories).
        ValueError: if distributions['continuous'] is not 'gaussian', 'normal', 'student-t',
            'exponential', 'gumbel'.
        ValueError: if distributions['binary'] is not 'probit', 'normal', 'logit'.
        ValueError: if distributions['categorical'] is not 'probit', 'normal', 'logit', 'gumbel'.
        ValueError: if distributions['weight'] is not 'normal' / 'gaussian' (alias), 'uniform'.
        ValueError: if distributions['intercept'] is not 'normal' / 'gaussian' (alias), 'uniform'.
    Example:
        sm = StructureModel()
        sm.add_edges_from([('A', 'C'), ('D', 'C'), ('E', 'D')])
        sm.add_nodes_from(['B', 'F'])
        schema = {'B': 'binary', 'C': 'categorical:5',
                  'E': 'binary', 'F': 'continuous'}
        df = sem_generator(sm, schema, noise_scale=1,
                          n_samples=10000,
                          intercept=True,
                          )
    """

    np.random.seed(seed)

    if not nx.algorithms.is_directed_acyclic_graph(graph):
        raise ValueError("Provided graph is not a DAG.")

    distributions = _set_default_distributions(distributions=distributions)
    validated_schema = validate_schema(
        nodes=graph.nodes(), schema=schema, default_type=default_type
    )
    var_fte_mapper = VariableFeatureMapper(validated_schema)

    n_columns = var_fte_mapper.n_features

    # get dependence based on edges in graph (not via adjacency matrix)
    w_mat = _create_weight_matrix(
        edges_w_weights=graph.edges(data="weight"),
        variable_to_indices_dict=var_fte_mapper.var_indices_dict,
        weight_distribution=distributions["weight"],
        intercept_distribution=distributions["intercept"],
        intercept=intercept,
    )

    # pre-allocate array
    x_mat = np.empty([n_samples, n_columns + 1 if intercept else n_columns])
    # intercept, append ones to the feature matrix
    if intercept:
        x_mat[:, -1] = 1

    # loop over sorted features according to ancestry (no parents first)
    for j_node in nx.topological_sort(graph):
        # all feature indices corresponding to the node/variable
        j_idx_list = var_fte_mapper.get_indices(j_node)

        # get all parent feature indices for the variable/node
        parents_idx = var_fte_mapper.get_indices(list(graph.predecessors(j_node)))
        if intercept:
            parents_idx += [n_columns]

        # continuous variable
        if var_fte_mapper.is_var_of_type(j_node, "continuous"):
            x_mat[:, j_idx_list[0]] = _add_continuous_noise(
                mean=x_mat[:, parents_idx].dot(w_mat[parents_idx, j_idx_list[0]]),
                distribution=distributions["continuous"],
                noise_std=noise_std,
            )

        # binary variable
        elif var_fte_mapper.is_var_of_type(j_node, "binary"):
            x_mat[:, j_idx_list[0]] = _sample_binary_from_latent(
                latent_mean=x_mat[:, parents_idx].dot(
                    w_mat[parents_idx, j_idx_list[0]]
                ),
                distribution=distributions["binary"],
                noise_std=noise_std,
            )

        # categorical variable
        elif var_fte_mapper.is_var_of_type(j_node, "categorical"):
            x_mat[:, j_idx_list] = _sample_categories_from_latent(
                latent_mean=x_mat[:, parents_idx].dot(
                    w_mat[np.ix_(parents_idx, j_idx_list)]
                ),
                distribution=distributions["categorical"],
                noise_std=noise_std,
            )

    return pd.DataFrame(
        x_mat[:, :-1] if intercept else x_mat, columns=var_fte_mapper.feature_list
    )


def _add_continuous_noise(
    mean: np.ndarray, distribution: str, noise_std: float,
) -> np.ndarray:
    n_samples = mean.shape[0]

    # add noise to mean
    if distribution in ("gaussian", "normal"):
        x = mean + np.random.normal(scale=noise_std, size=n_samples)
    elif distribution == "student-t":
        x = mean + np.random.standard_t(df=5, size=n_samples) * noise_std
    elif distribution == "exponential":
        x = mean + np.random.exponential(scale=noise_std, size=n_samples)
    elif distribution == "gumbel":
        x = mean + np.random.gumbel(scale=noise_std, size=n_samples)
    else:
        _raise_dist_error(
            "continuous",
            distribution,
            ["gaussian", "normal", "student-t", "exponential", "gumbel"],
        )

    return x


def _sample_binary_from_latent(
    latent_mean: np.ndarray, distribution: str, noise_std: float,
) -> np.ndarray:
    n_samples = latent_mean.shape[0]

    # add noise to latent variable
    if distribution in ("normal", "probit"):
        eta = latent_mean + np.random.normal(scale=noise_std, size=n_samples)
    elif distribution == "logit":
        eta = latent_mean + np.random.logistic(scale=noise_std, size=n_samples)
    else:
        _raise_dist_error("binary", distribution, ["logit", "probit", "normal"])

    # using a latent variable approach
    return (eta > 0).astype(int)


def _sample_categories_from_latent(
    latent_mean: np.ndarray, distribution: str, noise_std: float,
) -> np.ndarray:
    one_hot = np.empty_like(latent_mean)
    n_samples, n_cardinality = latent_mean.shape

    if distribution in ("normal", "probit"):
        latent_mean += np.random.normal(
            scale=noise_std, size=(n_samples, n_cardinality)
        )
    elif distribution in ("logit", "gumbel"):
        latent_mean += np.random.gumbel(
            scale=noise_std, size=(n_samples, n_cardinality)
        )
    else:
        _raise_dist_error(
            "categorical", distribution, ["logit", "gumbel", "probit", "normal"]
        )

    x_cat = np.argmax(latent_mean, axis=1)

    for i in range(n_cardinality):
        one_hot[:, i] = (x_cat == i).astype(int)

    return one_hot


def _set_default_distributions(distributions: Dict[str, str]) -> Dict[str, str]:
    default_distributions = {
        "continuous": "gaussian",
        "binary": "logit",
        "categorical": "logit",
        "weight": "uniform",
        "intercept": "uniform",
    }

    if distributions is None:
        return default_distributions
    # overwrite default with input data (if set)
    default_distributions.update(distributions)
    return default_distributions


def _create_weight_matrix(
    edges_w_weights: List[Tuple],
    variable_to_indices_dict: Dict[Hashable, List[int]],
    weight_distribution: str,
    intercept_distribution: str,
    intercept: bool,
) -> np.ndarray:
    """
    Creates a weight matrix for a linear SEM model from the edges of a graph.
    If the edges are unweighted, samples the weight values from a specified
    distribution. Optionally adds an intercept to the weights using a separate
    distribution.
    """
    n_columns = sum(len(x) for x in variable_to_indices_dict.values())

    w_mat = np.zeros(shape=(n_columns + 1 if intercept else n_columns, n_columns))
    for node_from, node_to, weight in edges_w_weights:

        ix_from = variable_to_indices_dict[node_from]
        ix_to = variable_to_indices_dict[node_to]
        ix_mask_array = np.ix_(ix_from, ix_to)

        # we cannot assign the same weight for each category!
        n_weights = len(ix_from) * len(ix_to)
        if weight is None:
            if weight_distribution == "uniform":
                # zero mean, unit variance:
                w_mat[ix_mask_array] = np.random.uniform(
                    -np.sqrt(12) / 2, np.sqrt(12) / 2, size=(len(ix_from), len(ix_to))
                )
            elif weight_distribution in ("gaussian", "normal"):
                w_mat[ix_mask_array] = np.random.normal(
                    loc=0, scale=1, size=(len(ix_from), len(ix_to))
                )
            else:
                _raise_dist_error(
                    "weight", intercept_distribution, ["uniform", "gaussian", "normal"]
                )

        else:
            if n_weights == 1:
                w_mat[ix_mask_array] = weight
            elif n_weights > 1:
                # assign weight randomly to a category (through the
                # normalization, this affects all categories from or to)
                sparse_mask = np.random.uniform(size=(len(ix_from), len(ix_to)))
                sparse_mask = (sparse_mask == np.min(sparse_mask)).astype(int)
                w_mat[ix_mask_array] = sparse_mask * weight
    if intercept:
        if intercept_distribution == "uniform":
            # zero mean, unit variance:
            w_mat[-1, :] = np.random.uniform(
                -np.sqrt(12) / 2, np.sqrt(12) / 2, size=[1, n_columns]
            )
        elif intercept_distribution in ("gaussian", "normal"):
            w_mat[-1, :] = np.random.normal(loc=0, scale=1, size=[1, n_columns])
        else:
            _raise_dist_error(
                "intercept", intercept_distribution, ["uniform", "gaussian", "normal"]
            )

    return w_mat


def _raise_dist_error(name: str, dist: str, dist_options):
    raise ValueError(
        "Unknown {} distribution {}, ".format(name, dist)
        + "valid distributions are {}".format(
            ", ".join(valid_dist for valid_dist in dist_options)
        )
    )


def generate_structure_dynamic(  # pylint: disable=R0913
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

    Returns:
        StructureModel containing all simulated nodes and edges (intra- and inter-slice)
    Raises:
        ValueError: if graph type unknown or `num_nodes < 2`
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
    res.add_nodes_from(["{var}_lag0".format(var=u) for u in sm_intra.nodes])
    res.add_weighted_edges_from(sm_inter.edges.data("weight"))
    res.add_weighted_edges_from(
        [
            ("{var}_lag0".format(var=u), "{var}_lag0".format(var=v), w)
            for u, v, w in sm_intra.edges.data("weight")
        ]
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

    Returns:
        G_inter: weighted, bipartite DAG for inter-slice connections

    Raises:
        ValueError: if graph type not known
    """
    if w_min > w_max:
        raise ValueError(
            "Absolute minimum weight must be less than or equal to maximum weight: {} > {}".format(
                w_min, w_max
            )
        )

    if graph_type == "erdos-renyi":
        prob = float(degree) / num_nodes
        b = (np.random.rand(p * num_nodes, num_nodes) < prob).astype(float)
        # np.fill_diagonal(B, 0)  # get rid of autoregressive terms
    elif graph_type == "full":  # ignore degree, only for experimental use
        b = np.ones([p * num_nodes, num_nodes])
    else:
        raise ValueError("Unknown inter-slice graph type")
    u = []
    for i in range(p):
        u_i = np.random.uniform(low=w_min, high=w_max, size=[num_nodes, num_nodes]) / (
            w_decay ** i
        )
        u_i[np.random.rand(num_nodes, num_nodes) < 0.5] *= -1
        u.append(u_i)
    if u:
        u = np.concatenate(u, axis=0)
    else:
        u = np.array([]).reshape(b.shape)

    a = (b != 0).astype(float) * u

    df = pd.DataFrame(
        a,
        index=[
            "{var}_lag{l_val}".format(var=var, l_val=l_val)
            for l_val in range(1, p + 1)
            for var in range(num_nodes)
        ],
        columns=[
            "{var}_lag{l_val}".format(var=var, l_val=0) for var in range(num_nodes)
        ],
    )
    idxs = list(df.index)
    cols = list(df.columns)
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
    if sem_type not in ("linear-gauss", "linear-exp", "linear-gumbel"):
        raise ValueError("unknown sem type")
    intra_nodes = sorted([el for el in g.nodes if "_lag0" in el])
    inter_nodes = sorted([el for el in g.nodes if "_lag0" not in el])
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
