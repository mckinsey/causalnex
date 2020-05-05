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

import networkx as nx
import numpy as np
import pytest
from networkx.algorithms.dag import is_directed_acyclic_graph
from scipy.stats import anderson

from causalnex.structure.data_generators import (
    generate_continuous_data,
    generate_structure,
)
from causalnex.structure.structuremodel import StructureModel


@pytest.fixture
def graph():
    graph = StructureModel()
    edges = [(n, n + 1, 1) for n in range(5)]
    graph.add_weighted_edges_from(edges)
    return graph


class TestGenerateStructure:
    @pytest.mark.parametrize("graph_type", ["erdos-renyi", "barabasi-albert", "full"])
    def test_is_dag_graph_type(self, graph_type):
        """ Tests that the generated graph is a dag for all graph types. """
        degree, d_nodes = 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        assert is_directed_acyclic_graph(sm)

    @pytest.mark.parametrize("num_nodes,degree", [(5, 2), (10, 3), (15, 5)])
    def test_is_dag_nodes_degrees(self, num_nodes, degree):
        """ Tests that generated graph is dag for different numbers of nodes and degrees
        """
        sm = generate_structure(num_nodes, degree)
        assert nx.is_directed_acyclic_graph(sm)

    def test_bad_graph_type(self):
        """ Test that a value other than "erdos-renyi", "barabasi-albert", "full" throws ValueError """
        graph_type = "invalid"
        degree, d_nodes = 4, 10
        with pytest.raises(ValueError, match="unknown graph type"):
            generate_structure(d_nodes, degree, graph_type)

    @pytest.mark.parametrize("num_nodes,degree", [(5, 2), (10, 3), (15, 5)])
    def test_expected_num_nodes(self, num_nodes, degree):
        """ Test that generated structure has expected number of nodes = num_nodes """
        sm = generate_structure(num_nodes, degree)
        assert len(sm.nodes) == num_nodes

    @pytest.mark.parametrize(
        "num_nodes,degree,w_range",
        [(5, 2, (1, 2)), (10, 3, (100, 200)), (15, 5, (1.0, 1.0))],
    )
    def test_weight_range(self, num_nodes, degree, w_range):
        """ Test that w_range is respected in output """
        w_min = w_range[0]
        w_max = w_range[1]
        sm = generate_structure(num_nodes, degree, w_min=w_min, w_max=w_max)
        assert all(abs(sm[u][v]["weight"]) >= w_min for u, v in sm.edges)
        assert all(abs(sm[u][v]["weight"]) <= w_max for u, v in sm.edges)

    @pytest.mark.parametrize("num_nodes", [-1, 0, 1])
    def test_num_nodes_exception(self, num_nodes):
        """ Check a single node graph can't be generated """
        with pytest.raises(ValueError, match="DAG must have at least 2 nodes"):
            generate_structure(num_nodes, 1)

    def test_min_max_weights_exception(self):
        """ Check that w_range is valid """
        with pytest.raises(
            ValueError,
            match="Absolute minimum weight must be less than or equal to maximum weight:",
        ):
            generate_structure(4, 1, w_min=0.5, w_max=0)

    def test_min_max_weights_equal(self):
        """ If w_range (w, w) has w=w, check abs value of all weights respect this """
        w = 1
        sm = generate_structure(4, 1, w_min=w, w_max=w)
        w_mat = nx.to_numpy_array(sm)
        assert np.all((w_mat == 0) | (w_mat == w) | (w_mat == -w))

    def test_erdos_renyi_degree_increases_edges(self):
        """ Erdos-Renyi degree increases edges """
        edge_counts = [
            max(
                [
                    len(generate_structure(100, degree, "erdos-renyi").edges)
                    for _ in range(10)
                ]
            )
            for degree in [10, 90]
        ]

        assert edge_counts == sorted(edge_counts)

    def test_barabasi_albert_degree_increases_edges(self):
        """ Barabasi-Albert degree increases edges """
        edge_counts = [
            max(
                [
                    len(generate_structure(100, degree, "barabasi-albert").edges)
                    for _ in range(10)
                ]
            )
            for degree in [10, 90]
        ]

        assert edge_counts == sorted(edge_counts)

    def test_full_network(self):
        """ Fully connected network has expected edge counts """
        sm = generate_structure(40, degree=0, graph_type="full")

        assert len(sm.edges) == (40 * 39) / 2


class TestGenerateContinuousData:
    @pytest.mark.parametrize(
        "sem_type", ["linear-gauss", "linear-exp", "linear-gumbel"]
    )
    def test_returns_ndarray(self, sem_type):
        """ Return value is an ndarray - test over all sem_types """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        ndarray = generate_continuous_data(sm, sem_type=sem_type, n_samples=10)
        assert isinstance(ndarray, np.ndarray)

    def test_bad_sem_type(self):
        """ Test that invalid sem-type other than "linear-gauss", "linear-exp", "linear-gumbel" is not accepted """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        with pytest.raises(ValueError, match="unknown sem type"):
            generate_continuous_data(sm, sem_type="invalid", n_samples=10, seed=10)

    @pytest.mark.parametrize("num_nodes", [5, 10, 15])
    def test_number_of_nodes(self, num_nodes):
        """ Length of each row in generated data equals num_nodes """
        graph = StructureModel()
        edges = [(n, n + 1, 1) for n in range(num_nodes - 1)]
        graph.add_weighted_edges_from(edges)

        data = generate_continuous_data(graph, 100, seed=10)
        assert all(len(sample) == num_nodes for sample in data)

    @pytest.mark.parametrize("num_samples", [5, 10, 15])
    def test_number_of_samples(self, num_samples, graph):
        """ Assert number of samples generated (rows) = num_samples """
        data = generate_continuous_data(graph, num_samples, "linear-gauss", 1, seed=10)
        assert len(data) == num_samples

    def test_linear_gauss_parent_dist(self, graph):
        """ Anderson-Darling test for data coming from a particular distribution, for linear-gauss."""
        data = generate_continuous_data(graph, 1000000, "linear-gauss", 1, seed=10)

        stat, crit, sig = anderson(data[:, 0], "norm")
        assert stat < crit[list(sig).index(5)]

    def test_linear_exp_parent_dist(self, graph):
        """ Anderson-Darling test for data coming from a particular distribution, for linear-exp """
        data = generate_continuous_data(graph, 1000000, "linear-exp", 1, seed=10)

        stat, crit, sig = anderson(data[:, 0], "expon")
        assert stat < crit[list(sig).index(5)]

    def test_linear_gumbel_parent_dist(self, graph):
        """ Anderson-Darling test for data coming from a particular distribution, for linear-exp """
        data = generate_continuous_data(graph, 1000000, "linear-gumbel", 1, seed=10)

        stat, crit, sig = anderson(data[:, 0], "gumbel_r")
        assert stat < crit[list(sig).index(5)]

    @pytest.mark.parametrize("num_nodes", (10, 20, 30))
    @pytest.mark.parametrize("seed", (10, 20, 30))
    def test_order_is_correct(self, num_nodes, seed):
        """
        Check if the order of the nodes is the same order as `sm.nodes`, which in turn is the same order as the
        adjacency matrix.

        To do so, we create graphs with degree in {0,1} by doing permutations on identity.
        The edge values are always 100 and the noise is 1, so we expect `edge_from` < `edge_to` in absolute value
        almost every time.
        """
        np.random.seed(seed)
        perms = np.tril(np.random.permutation(np.eye(num_nodes, num_nodes)) * 100, -1)
        perms = np.array(perms).T
        edges_from, edges_to = np.where(perms)
        sm = StructureModel(perms)
        data = generate_continuous_data(sm, 10, "linear-gauss", 1, seed=10)
        for edge_from, edge_to in zip(edges_from, edges_to):
            assert np.all(np.abs(data[:, edge_from]) < np.abs(data[:, edge_to]))
