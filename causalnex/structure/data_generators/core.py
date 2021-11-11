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
from typing import Dict, Hashable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, Kernel

from causalnex.structure import StructureModel
from causalnex.structure.categorical_variable_mapper import (
    VariableFeatureMapper,
    validate_schema,
)

# dict mapping distributions names to their functions
__distribution_mapper = {
    "gaussian": np.random.normal,
    "normal": np.random.normal,
    "student-t": np.random.standard_t,
    "gumbel": np.random.gumbel,
    "exponential": np.random.exponential,
    "probit": np.random.normal,
    "logit": np.random.logistic,
}


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
            "Absolute minimum weight must be less than or equal to maximum weight: "
            f"{w_min} > {w_max}"
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
        raise ValueError(
            f"Unknown graph type {graph_type}. "
            "Available types are ['erdos-renyi', 'barabasi-albert', 'full']"
        )

    # randomly permute edges - required because we limited ourselves to lower diagonal previously
    perms = np.random.permutation(np.eye(num_nodes, num_nodes))
    edge_flags = perms.T.dot(edge_flags).dot(perms)

    # random edge weights between w_min, w_max or between -w_min, -w_max
    edge_weights = np.random.uniform(low=w_min, high=w_max, size=[num_nodes, num_nodes])
    edge_weights[np.random.rand(num_nodes, num_nodes) < 0.5] *= -1

    adj_matrix = (edge_flags != 0).astype(float) * edge_weights
    graph = StructureModel(adj_matrix)
    return graph


def sem_generator(
    graph: nx.DiGraph,
    schema: Optional[Dict] = None,
    default_type: str = "continuous",
    noise_std: float = 1.0,
    n_samples: int = 1000,
    distributions: Dict[str, Union[str, float]] = None,
    intercept: bool = True,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generator for tabular data with mixed variable types from a DAG.

    NOTE: the root nodes of the DAG are sampled from a distribution with noise_std=1.0 always.
    This is so that increases in the noise_std are in relation to a fixed spread, and therefore
    actually have an impact on the fit. Not using this method causes the noise_std to only change
    the axis scaling.

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
            ``count``: The zero-inflation probability as a float.
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
        ValueError: if distributions['count'], the zero-inflation factor is not a float in [0, 1].

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
    distributions, var_fte_mapper, x_mat = _init_sem_data_gen(
        graph=graph,
        schema=schema,
        n_samples=n_samples,
        default_type=default_type,
        distributions=distributions,
        seed=seed,
    )

    # get dependence based on edges in graph (not via adjacency matrix)
    w_mat = _create_weight_matrix(
        edges_w_weights=graph.edges(data="weight"),
        variable_to_indices_dict=var_fte_mapper.var_indices_dict,
        weight_distribution=distributions["weight"],
        intercept_distribution=distributions["intercept"],
        intercept=intercept,
    )

    # intercept, append ones to the feature matrix
    if intercept:
        x_mat = np.append(x_mat, np.ones(shape=(n_samples, 1)), axis=1)
        intercept_idx = [x_mat.shape[1] - 1]

    # if intercept is used, the root nodes have len = 1
    root_node_len = 1 if intercept else 0

    # loop over sorted features according to ancestry (no parents first)
    for j_node in nx.topological_sort(graph):
        # all feature indices corresponding to the node/variable
        j_idx_list = var_fte_mapper.get_indices(j_node)

        # get all parent feature indices for the variable/node
        parents_idx = var_fte_mapper.get_indices(list(graph.predecessors(j_node)))
        if intercept:
            parents_idx += intercept_idx

        # if the data is a root node, must initialise the axis separate from noise parameter
        root_node = len(parents_idx) <= root_node_len

        # continuous variable
        if var_fte_mapper.is_var_of_type(j_node, "continuous"):
            x_mat[:, j_idx_list[0]] = _add_continuous_noise(
                mean=x_mat[:, parents_idx].dot(w_mat[parents_idx, j_idx_list[0]]),
                distribution=distributions["continuous"],
                noise_std=noise_std,
                root_node=root_node,
            )

        # binary variable
        elif var_fte_mapper.is_var_of_type(j_node, "binary"):
            x_mat[:, j_idx_list[0]] = _sample_binary_from_latent(
                latent_mean=x_mat[:, parents_idx].dot(
                    w_mat[parents_idx, j_idx_list[0]]
                ),
                distribution=distributions["binary"],
                noise_std=noise_std,
                root_node=root_node,
            )

        # count variable
        elif var_fte_mapper.is_var_of_type(j_node, "count"):
            x_mat[:, j_idx_list[0]] = _sample_count_from_latent(
                eta=x_mat[:, parents_idx].dot(w_mat[parents_idx, j_idx_list[0]]),
                zero_inflation_pct=distributions["count"],
                root_node=root_node,
            )

        # categorical variable
        elif var_fte_mapper.is_var_of_type(j_node, "categorical"):
            x_mat[:, j_idx_list] = _sample_categories_from_latent(
                latent_mean=x_mat[:, parents_idx].dot(
                    w_mat[np.ix_(parents_idx, j_idx_list)]
                ),
                distribution=distributions["categorical"],
                noise_std=noise_std,
                root_node=root_node,
            )

    return pd.DataFrame(
        x_mat[:, :-1] if intercept else x_mat, columns=var_fte_mapper.feature_list
    )


def _handle_distribution_sampling(
    distribution: str,
    distribution_func,
    noise_std: float,
    size: Tuple[int],
    root_node: bool,
):
    # force scale to be 1 for the root node
    if root_node:
        noise_std = 1

    # special sampling syntax
    if distribution == "student-t":
        return distribution_func(df=5, size=size) * noise_std

    # default sampling syntax
    return distribution_func(scale=noise_std, size=size)


def _add_continuous_noise(
    mean: np.ndarray,
    distribution: str,
    noise_std: float,
    root_node: bool,
) -> np.ndarray:
    n_samples = mean.shape[0]

    # try and get the requested distribution from the mapper
    distribution_func = __distribution_mapper.get(distribution, None)
    if distribution_func is None:
        _raise_dist_error(
            "continuous",
            distribution,
            ["gaussian", "normal", "student-t", "exponential", "gumbel"],
        )

    # add noise to mean
    mean += _handle_distribution_sampling(
        distribution=distribution,
        distribution_func=distribution_func,
        noise_std=noise_std,
        size=(n_samples,),
        root_node=root_node,
    )

    return mean


def _sample_binary_from_latent(
    latent_mean: np.ndarray,
    distribution: str,
    noise_std: float,
    root_node: bool,
    max_imbalance: float = 0.05,
) -> np.ndarray:
    n_samples = latent_mean.shape[0]

    # try and get the requested distribution from the mapper
    distribution_func = __distribution_mapper.get(distribution, None)
    if distribution_func is None:
        _raise_dist_error("binary", distribution, ["logit", "probit", "normal"])

    # add noise to mean
    latent_mean += _handle_distribution_sampling(
        distribution=distribution,
        distribution_func=distribution_func,
        noise_std=noise_std,
        size=(n_samples,),
        root_node=root_node,
    )

    # use an alternative threshold if 0 leads to heavy imbalance
    labels = (latent_mean > 0).astype(int)
    share_positive = np.mean(labels)
    if share_positive < max_imbalance:
        return (latent_mean > np.quantile(latent_mean, max_imbalance)).astype(int)
    if share_positive > (1 - max_imbalance):
        return (latent_mean > np.quantile(latent_mean, 1 - max_imbalance)).astype(int)
    return labels


def _sample_count_from_latent(
    eta: np.ndarray,
    root_node: bool,
    zero_inflation_pct: float = 0.05,
) -> np.ndarray:
    """
    Samples a zero-inflated poisson distribution.
    Returns:
        Samples from a Poisson distribution.
    Raises:
        ValueError: Unsupported zero-inflation factor.
    """
    if (
        not isinstance(zero_inflation_pct, (float, int))
        or zero_inflation_pct < 0
        or zero_inflation_pct > 1
    ):
        raise ValueError(
            "Unsupported zero-inflation factor, distribution['count'] needs to be a float in [0, 1]"
        )
    n_samples = eta.shape[0]

    # add noise manually if root node
    # uniform [0, 1] makes sure that the counts are small
    if root_node:
        eta += np.random.uniform(size=n_samples)

    zif = np.random.uniform(size=n_samples) < zero_inflation_pct
    count = _sample_poisson(expected_count=_exp_relu(eta))

    # inflate the zeros:
    count[zif] = 0
    return count


def _exp_relu(x):
    x[x < 0] = np.exp(x[x < 0])
    return x


def _sample_poisson(expected_count: np.ndarray, max_count: int = 5000) -> np.ndarray:
    """
    Samples from a poisson distribution using each element in ``latent_mean``
    as the Poisson parameter.

    Args:
        expected_count: Event rate of the Poisson process, can be of any array
            dimension. Defined on (0, infty).
        max_count: Bounds the count from above. The count sample is created
            with a while loop. This argument is the maximum number of loop
            iterations before stopping. Default value should run on most
            machines in reasonable amount of time.
    Returns:
        Sampled count of a Poisson distribution from the given mean.
    """
    # use log for numeric stability for large count values
    log_cond_intensity = -expected_count
    log_intensity_budget = np.copy(log_cond_intensity)

    count = np.zeros_like(expected_count)

    log_uni = np.log(np.random.uniform(size=expected_count.shape))
    mask = log_uni >= log_intensity_budget

    while np.any(mask) and count.max() < max_count:
        mask = log_uni >= log_intensity_budget
        count[mask] += 1
        log_cond_intensity[mask] += np.log(expected_count[mask] / count[mask])
        log_intensity_budget[mask] = np.logaddexp(
            log_intensity_budget[mask], log_cond_intensity[mask]
        )

    return count


def _sample_categories_from_latent(
    latent_mean: np.ndarray,
    distribution: str,
    noise_std: float,
    root_node: bool,
) -> np.ndarray:

    one_hot = np.empty_like(latent_mean)
    n_samples, n_cardinality = latent_mean.shape

    # try and get the requested distribution from the mapper
    distribution_func = __distribution_mapper.get(distribution, None)
    if distribution_func is None:
        _raise_dist_error(
            "categorical", distribution, ["logit", "gumbel", "probit", "normal"]
        )

    # add noise to mean
    latent_mean += _handle_distribution_sampling(
        distribution=distribution,
        distribution_func=distribution_func,
        noise_std=noise_std,
        size=(n_samples, n_cardinality),
        root_node=root_node,
    )

    x_cat = np.argmax(latent_mean, axis=1)

    for i in range(n_cardinality):
        one_hot[:, i] = (x_cat == i).astype(int)

    return one_hot


def _set_default_distributions(
    distributions: Dict[str, Union[str, float]]
) -> Dict[str, Union[str, float]]:
    default_distributions = {
        "continuous": "gaussian",
        "binary": "logit",
        "categorical": "logit",
        "weight": "uniform",
        "intercept": "uniform",
        "count": 0.05,
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
    valid_dists = ", ".join(valid_dist for valid_dist in dist_options)
    raise ValueError(
        f"Unknown {name} distribution {dist}, valid distributions are {valid_dists}"
    )


def _init_sem_data_gen(
    graph: nx.DiGraph,
    schema: Dict,
    n_samples: int,
    default_type: str,
    distributions: Dict[str, str],
    seed: int,
):
    np.random.seed(seed)

    if not nx.algorithms.is_directed_acyclic_graph(graph):
        raise ValueError("Provided graph is not a DAG.")

    distributions = _set_default_distributions(distributions=distributions)
    validated_schema = validate_schema(
        nodes=graph.nodes(), schema=schema, default_type=default_type
    )
    var_fte_mapper = VariableFeatureMapper(validated_schema)

    # pre-allocate array
    n_columns = var_fte_mapper.n_features
    x_mat = np.empty([n_samples, n_columns])

    return distributions, var_fte_mapper, x_mat


def nonlinear_sem_generator(
    graph: nx.DiGraph,
    kernel: Kernel = RBF(1),
    schema: Optional[Dict] = None,
    default_type: str = "continuous",
    noise_std: float = 1.0,
    n_samples: int = 1000,
    distributions: Dict[str, str] = None,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generator for non-linear tabular data with mixed variable types from a DAG.

    The nonlinearity can be controlled via the ``kernel``. Note that a
    ``DotProduct`` is equivalent to a linear function (without mean).

    Supported variable types: `'binary', 'categorical', 'continuous'`. The number
    of categories can be determined using a colon, e.g. `'categorical:5'`
    specifies a categorical feature with 5 categories.

    Notation: For binary and continuous variables, a ``variable'' refers to a
    ``node'', a ``feature'' refers to the one-hot column for categorical
    variables and is equivalent to a binary or continuous variable.

    Args:
        graph: A DAG in form of a networkx or StructureModel.
        kernel: A kernel from sklearn.gaussian_process.kernels like RBF(1) or
            Matern(1) or any combinations thereof. The kernels are used to
            create the latent variable for the binary / categorical variables
            and are directly used for continuous variables.
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
        ValueError: if distributions['count'], the zero-inflation factor is not a float in [0, 1].

    Example:
        sm = StructureModel()

        sm.add_edges_from([('A', 'C'), ('D', 'C'), ('E', 'D')])

        sm.add_nodes_from(['B', 'F'])

        schema = {'B': 'binary', 'C': 'categorical:5',
                  'E': 'binary', 'F': 'continuous'}

        df = sem_generator(sm, schema, kernel=RBF(1), noise_scale=1,
                          n_samples=10000)
    """
    distributions, var_fte_mapper, x_mat = _init_sem_data_gen(
        graph=graph,
        schema=schema,
        n_samples=n_samples,
        default_type=default_type,
        distributions=distributions,
        seed=seed,
    )

    # loop over sorted features according to ancestry (no parents first)
    for j_node in nx.topological_sort(graph):
        # all feature indices corresponding to the node/variable
        j_idx_list = var_fte_mapper.get_indices(j_node)

        # get all parent feature indices for the variable/node
        parents_idx = var_fte_mapper.get_indices(list(graph.predecessors(j_node)))

        # if the data is a root node, must initialise the axis separate from noise parameter
        root_node = len(parents_idx) <= 0

        # continuous variable
        if var_fte_mapper.is_var_of_type(j_node, "continuous"):
            x_mat[:, j_idx_list[0]] = _add_continuous_noise(
                mean=_gp_index(x_mat[:, parents_idx], kernel),
                distribution=distributions["continuous"],
                noise_std=noise_std,
                root_node=root_node,
            )

        # binary variable
        elif var_fte_mapper.is_var_of_type(j_node, "binary"):
            x_mat[:, j_idx_list[0]] = _sample_binary_from_latent(
                latent_mean=_gp_index(x_mat[:, parents_idx], kernel),
                distribution=distributions["binary"],
                noise_std=noise_std,
                root_node=root_node,
            )

        # count
        if var_fte_mapper.is_var_of_type(j_node, "count"):
            x_mat[:, j_idx_list[0]] = _sample_count_from_latent(
                eta=_gp_index(x_mat[:, parents_idx], kernel),
                zero_inflation_pct=distributions["count"],
                root_node=root_node,
            )

        # categorical variable
        elif var_fte_mapper.is_var_of_type(j_node, "categorical"):
            x_mat[:, j_idx_list] = _sample_categories_from_latent(
                latent_mean=np.concatenate(
                    [
                        np.expand_dims(_gp_index(x_mat[:, parents_idx], kernel), axis=1)
                        for _ in j_idx_list
                    ],
                    axis=1,
                ),
                distribution=distributions["categorical"],
                noise_std=noise_std,
                root_node=root_node,
            )
    return pd.DataFrame(x_mat, columns=var_fte_mapper.feature_list)


def _unconditional_sample(x, kernel):
    cov_mat = kernel(x)
    y = np.random.multivariate_normal(mean=np.zeros(shape=x.shape[0]), cov=cov_mat)
    return y.squeeze(), cov_mat


def _conditional_sample(
    x_new, x_old, f_old, kernel, cov_mat_old: np.ndarray = None, epsilon=0.00001
):

    cov_mat_new = kernel(x_new)
    cross_cov = kernel(x_old, x_new)
    # X_no.T @ inv(X_oo):
    reg_coef = np.linalg.solve(
        cov_mat_old + epsilon * np.eye(x_old.shape[0]), cross_cov
    ).T

    # calculate conditional mean and cov
    cond_cov = (cov_mat_new - reg_coef @ cross_cov) + epsilon * np.eye(x_new.shape[0])
    cond_mean = (reg_coef @ f_old).squeeze()

    # sample
    y_new = np.random.multivariate_normal(mean=cond_mean, cov=cond_cov).squeeze()
    return y_new, cov_mat_new


def _gp_index(
    x: np.ndarray,
    kernel: Kernel,
    max_chunk_size: int = 100,
) -> np.ndarray:
    """
    Sample a Gaussian process using input data.
    ``f(x) ~ GP(0, K)``

    If the number of samples is larger than ``max_chunk_size``, the sampling is
    split in sorted batches (first dimension) and sampled using a conditional
    multivariate normal.

    Args:
        x:
        kernel:
        max_chunk_size:

    Returns:
        A one-dimensional numpy array with a sample of f(x)
    """
    # if we dont have a parent, the input will have no columns
    if x.shape[1] == 0:
        return np.zeros(shape=(x.shape[0],))

    use_batches = x.shape[0] > max_chunk_size

    if not use_batches:
        y, _ = _unconditional_sample(x, kernel=kernel)
        return _scale_y(y)

    # if we need batches, we sort according to the first dimension
    ix_sort = np.argsort(x, axis=0)[:, 0].squeeze()
    reverse_ix = np.argsort(ix_sort).squeeze()

    # split into smaller pieces
    n_splits = (x.shape[0] // max_chunk_size) + 1
    x_splits = np.array_split(x[ix_sort, :], n_splits)

    outputs = []
    y, cov_mat = _unconditional_sample(x_splits[0], kernel=kernel)
    outputs.append(y)
    x_old = x_splits[0]
    for x_subset in x_splits[1:]:
        y, cov_mat = _conditional_sample(
            x_new=x_subset,
            x_old=x_old,
            f_old=outputs[-1],
            kernel=kernel,
            cov_mat_old=cov_mat,
        )
        outputs.append(y)
        x_old = x_subset

    y_all = _scale_y(np.concatenate(outputs))
    return y_all[reverse_ix]


def _scale_y(y):
    """Normalize variance to 1."""
    return y / y.std()
