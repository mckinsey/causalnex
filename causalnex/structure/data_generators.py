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

Structure and continuous data generator based on implementation found in: from https://github.com/xunzheng/notears
git hash: 31923cb22517f7bb6420dd0b6ef23ca550702b97


"""

import networkx as nx
import numpy as np
import pandas as pd

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
    sm: StructureModel,
    n_samples: int,
    sem_type: str = "linear-gauss",
    noise_scale: float = 1.0,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate samples from SEM with specified type of noise.
    The order of the columns on the returned array is the one provided by `sm.nodes`

    Args:
        sm: weigthed DAG - nodes must be zero-indexed
        n_samples: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM
        seed: Random state
    Returns:
        x_mat: [n_samples,d_nodes] sample matrix
    Raises:
        ValueError: if sem_type isn't linear-gauss/linear_exp/linear-gumbel
    """
    np.random.seed(seed)

    ordered_vertices = list(nx.topological_sort(sm))
    w_mat = nx.to_numpy_array(sm, nodelist=sm.nodes)
    d_nodes = w_mat.shape[0]
    x_mat = np.zeros([n_samples, d_nodes])
    vertices_to_idx = {c: i for i, c in enumerate(sm.nodes)}

    for j_name in ordered_vertices:
        j_index = vertices_to_idx[j_name]
        parents = list(sm.predecessors(j_name))
        if parents:
            # need to deal with indices not columns here (x_mat is a np array)
            parent_indices = [vertices_to_idx[p] for p in parents]
            eta = x_mat[:, parent_indices].dot(
                w_mat[parent_indices, j_index]
            )  # [n_samples,]
        else:
            eta = 0

        if sem_type == "linear-gauss":
            x_mat[:, j_index] = eta + np.random.normal(
                scale=noise_scale, size=n_samples
            )
        elif sem_type == "linear-exp":
            x_mat[:, j_index] = eta + np.random.exponential(
                scale=noise_scale, size=n_samples
            )
        elif sem_type == "linear-gumbel":
            x_mat[:, j_index] = eta + np.random.gumbel(
                scale=noise_scale, size=n_samples
            )
        else:
            raise ValueError("unknown sem type")
    return x_mat


def generate_continuous_dataframe(
    sm: StructureModel,
    n_samples: int,
    sem_type: str = "linear-gauss",
    noise_scale: float = 1.0,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.
    Args:
        sm: weigthed DAG - nodes must be zero-indexed
        n_samples: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM
        seed: Random state
    Returns:
        Dataframe with the node names as column names
    Raises:
        ValueError: if sem_type isn't linear-gauss/linear_exp/linear-gumbel
    """
    x_mat = generate_continuous_data(
        sm=sm,
        n_samples=n_samples,
        sem_type=sem_type,
        noise_scale=noise_scale,
        seed=seed,
    )

    return pd.DataFrame(x_mat, columns=sm.nodes)
