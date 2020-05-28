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
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

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
