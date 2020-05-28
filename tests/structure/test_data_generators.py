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
import operator
import string
from itertools import product
from typing import Hashable, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from networkx.algorithms.dag import is_directed_acyclic_graph
from scipy.stats import anderson, stats

from causalnex.structure.data_generators import (
    generate_binary_data,
    generate_binary_dataframe,
    generate_categorical_dataframe,
    generate_continuous_data,
    generate_continuous_dataframe,
    generate_structure,
    sem_generator,
)
from causalnex.structure.structuremodel import StructureModel


@pytest.fixture
def graph():
    graph = StructureModel()
    edges = [(n, n + 1, 1) for n in range(5)]
    graph.add_weighted_edges_from(edges)
    return graph


@pytest.fixture
def schema():
    # use the default schema for 3
    schema = {
        0: "binary",
        1: "categorical:3",
        2: "binary",
        4: "continuous",
        5: "categorical:5",
    }
    return schema


@pytest.fixture()
def graph_gen():
    def generator(num_nodes, seed, weight=None):
        np.random.seed(seed)

        sm = StructureModel()
        nodes = list(
            "".join(x) for x in product(string.ascii_lowercase, string.ascii_lowercase)
        )[:num_nodes]
        np.random.shuffle(nodes)
        sm.add_nodes_from(nodes)

        # one edge:
        sm.add_weighted_edges_from([("aa", "ab", weight)])
        return sm

    return generator


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
            match="Absolute minimum weight must be less than or equal to maximum weight",
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
        "distribution", ["gaussian", "normal", "student-t", "exponential", "gumbel"]
    )
    def test_returns_ndarray(self, distribution):
        """ Return value is an ndarray - test over all sem_types """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        ndarray = generate_continuous_data(sm, distribution=distribution, n_samples=10)
        assert isinstance(ndarray, np.ndarray)

    def test_bad_distribution_type(self):
        """ Test that invalid sem-type other than "gaussian", "normal", "student-t",
        "exponential", "gumbel" is not accepted """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        with pytest.raises(ValueError, match="Unknown continuous distribution"):
            generate_continuous_data(sm, distribution="invalid", n_samples=10, seed=10)

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
        data = generate_continuous_data(graph, num_samples, "gaussian", 1, seed=10)
        assert len(data) == num_samples

    def test_linear_gauss_parent_dist(self, graph):
        """ Anderson-Darling test for data coming from a particular distribution, for gaussian."""
        data = generate_continuous_data(graph, 1000000, "gaussian", 1, seed=10)

        stat, crit, sig = anderson(data[:, 0], "norm")
        assert stat < crit[list(sig).index(5)]

    def test_linear_normal_parent_dist(self, graph):
        """ Anderson-Darling test for data coming from a particular distribution, for normal."""
        data = generate_continuous_data(
            graph, distribution="normal", n_samples=1000000, noise_scale=1, seed=10
        )

        stat, crit, sig = anderson(data[:, 0], "norm")
        assert stat < crit[list(sig).index(5)]

    def test_linear_studentt_parent_dist(self, graph):
        """
        Kolmogorov-Smirnov test for data coming from a student-t (degree of freedom = 3).
        """
        np.random.seed(10)

        data = generate_continuous_data(
            graph, distribution="student-t", noise_scale=1, n_samples=100000, seed=10
        )

        x = data[:, 0]
        _, p_val = stats.kstest(x, "t", args=[3])
        assert p_val < 0.01

    def test_linear_exp_parent_dist(self, graph):
        """ Anderson-Darling test for data coming from a particular distribution, for exponential."""
        data = generate_continuous_data(
            graph, distribution="exponential", noise_scale=1, n_samples=100000, seed=10
        )

        stat, crit, sig = anderson(data[:, 0], "expon")
        assert stat < crit[list(sig).index(5)]

    def test_linear_gumbel_parent_dist(self, graph):
        """ Anderson-Darling test for data coming from a particular distribution, for gumbel."""
        data = generate_continuous_data(
            graph, distribution="gumbel", noise_scale=1, n_samples=100000, seed=10
        )

        stat, crit, sig = anderson(data[:, 0], "gumbel_r")
        assert stat < crit[list(sig).index(5)]

    @pytest.mark.parametrize(
        "distribution", ["gaussian", "normal", "student-t", "exponential", "gumbel"]
    )
    def test_intercept(self, distribution):
        graph = StructureModel()
        graph.add_node("123")

        data_noint = generate_continuous_data(
            graph,
            n_samples=100000,
            distribution=distribution,
            noise_scale=0,
            seed=10,
            intercept=False,
        )
        data_intercept = generate_continuous_data(
            graph,
            n_samples=100000,
            distribution=distribution,
            noise_scale=0,
            seed=10,
            intercept=True,
        )
        assert not np.isclose(data_noint[:, 0].mean(), data_intercept[:, 0].mean())
        assert np.isclose(data_noint[:, 0].std(), data_intercept[:, 0].std())

    @pytest.mark.parametrize("num_nodes", (10, 20, 30))
    @pytest.mark.parametrize("seed", (10, 20, 30))
    def test_order_is_correct(self, graph_gen, num_nodes, seed):
        """
        Check if the order of the nodes is the same order as `sm.nodes`, which in turn is the same order as the
        adjacency matrix.

        To do so, we create graphs with degree in {0,1} by doing permutations on identity.
        The edge values are always 100 and the noise is 1, so we expect `edge_from` < `edge_to` in absolute value
        almost every time.
        """
        sm = graph_gen(num_nodes=num_nodes, seed=seed, weight=100)
        nodes = sm.nodes()
        node_seq = {node: ix for ix, node in enumerate(sm.nodes())}

        data = generate_continuous_data(
            sm,
            n_samples=10000,
            distribution="normal",
            seed=seed,
            noise_scale=1.0,
            intercept=False,
        )

        assert data[:, node_seq["aa"]].std() < data[:, node_seq["ab"]].std()

        tol = 0.15
        # for gaussian distribution: var=0 iff independent:
        for node in nodes:
            if node == "aa":
                continue
            if node == "ab":
                assert not np.isclose(
                    np.corrcoef(data[:, [node_seq["aa"], node_seq["ab"]]].T)[0, 1],
                    0,
                    atol=tol,
                )
            else:
                assert np.isclose(
                    np.corrcoef(data[:, [node_seq["aa"], node_seq[node]]].T)[0, 1],
                    0,
                    atol=tol,
                )

    @pytest.mark.parametrize(
        "distribution", ["gaussian", "normal", "student-t", "exponential", "gumbel"]
    )
    @pytest.mark.parametrize("noise_std", [0.1, 1, 2])
    @pytest.mark.parametrize("intercept", [True, False])
    @pytest.mark.parametrize("seed", [10, 12])
    def test_dataframe(self, graph, distribution, noise_std, intercept, seed):
        """
        Tests equivalence of dataframe wrapper
        """
        data = generate_continuous_data(
            graph,
            1000,
            distribution,
            noise_scale=noise_std,
            seed=seed,
            intercept=intercept,
        )
        df = generate_continuous_dataframe(
            graph,
            1000,
            distribution,
            noise_scale=noise_std,
            seed=seed,
            intercept=intercept,
        )

        assert np.array_equal(data, df[list(graph.nodes())].values)


class TestGenerateBinaryData:
    @pytest.mark.parametrize("distribution", ["probit", "normal", "logit"])
    def test_returns_ndarray(self, distribution):
        """ Return value is an ndarray - test over all sem_types """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        ndarray = generate_binary_data(sm, distribution=distribution, n_samples=10)
        assert isinstance(ndarray, np.ndarray)

    def test_bad_distribution_type(self):
        """ Test that invalid sem-type other than "probit", "normal", "logit" is not accepted """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        with pytest.raises(ValueError, match="Unknown binary distribution"):
            generate_binary_data(sm, distribution="invalid", n_samples=10, seed=10)

    @pytest.mark.parametrize("num_nodes", [5, 10, 15])
    def test_number_of_nodes(self, num_nodes):
        """ Length of each row in generated data equals num_nodes """
        graph = StructureModel()
        edges = [(n, n + 1, 1) for n in range(num_nodes - 1)]
        graph.add_weighted_edges_from(edges)

        data = generate_binary_data(graph, 100, seed=10)
        assert all(len(sample) == num_nodes for sample in data)

    @pytest.mark.parametrize("num_samples", [5, 10, 15])
    def test_number_of_samples(self, num_samples, graph):
        """ Assert number of samples generated (rows) = num_samples """
        data = generate_binary_data(graph, num_samples, "logit", 1, seed=10)
        assert len(data) == num_samples

    @pytest.mark.parametrize("distribution", ["logit", "probit", "normal"])
    def test_baseline_probability_probit(self, graph, distribution):
        """ Test whether probability centered around 50% if no intercept given"""
        graph = StructureModel()
        graph.add_nodes_from(["A"])
        data = generate_binary_data(
            graph,
            1000000,
            distribution=distribution,
            noise_scale=0.1,
            seed=10,
            intercept=False,
        )
        assert 0.45 < data[:, 0].mean() < 0.55

    @pytest.mark.parametrize("distribution", ["logit", "probit", "normal"])
    def test_intercept_probability_logit(self, graph, distribution):
        """ Test whether probability is not centered around 50% when using an intercept"""
        graph = StructureModel()
        graph.add_nodes_from(["A"])
        data = generate_binary_data(
            graph,
            1000000,
            distribution=distribution,
            noise_scale=0.1,
            seed=10,
            intercept=True,
        )
        mean_prob = data[:, 0].mean()
        assert not np.isclose(mean_prob, 0.5, atol=0.05)

    @pytest.mark.parametrize("distribution", ["logit", "probit", "normal"])
    def test_intercept(self, distribution):
        graph = StructureModel()
        graph.add_node("123")

        data_noint = generate_binary_data(
            graph, 100000, distribution, noise_scale=0, seed=10, intercept=False
        )
        data_intercept = generate_binary_data(
            graph, 100000, distribution, noise_scale=0, seed=10, intercept=True
        )
        assert not np.isclose(data_noint[:, 0].mean(), data_intercept[:, 0].mean())

    @pytest.mark.parametrize("num_nodes", (10, 20, 30))
    @pytest.mark.parametrize("seed", (10, 20, 30))
    def test_order_is_correct(self, graph_gen, num_nodes, seed):
        """
        Check if the order of the nodes is the same order as `sm.nodes`, which in turn is the same order as the
        adjacency matrix.

        To do so, we create graphs with degree in {0,1} by doing permutations on identity.
        The edge values are always 100 and the noise is 1, so we expect `edge_from` < `edge_to` in absolute value
        almost every time.
        """
        sm = graph_gen(num_nodes=num_nodes, seed=seed, weight=None)
        nodes = sm.nodes()
        node_seq = {node: ix for ix, node in enumerate(sm.nodes())}

        data = generate_binary_data(
            sm,
            n_samples=10000,
            distribution="normal",
            seed=seed,
            noise_scale=0.1,
            intercept=False,
        )
        tol = 0.15
        # since we dont have an intercept, the mean proba for the parent is 0.5,
        # which has the highest possible std for a binary feature (std= p(1-p)),
        # hence, any child necessarily has a lower probability.
        assert data[:, node_seq["aa"]].std() > data[:, node_seq["ab"]].std()

        for node in nodes:
            if node == "aa":
                continue
            joint_proba, factored_proba = calculate_proba(
                data, node_seq["aa"], node_seq[node]
            )
            if node == "ab":
                # this is the only link
                assert not np.isclose(joint_proba, factored_proba, rtol=tol, atol=0)
            else:
                assert np.isclose(joint_proba, factored_proba, rtol=tol, atol=0)

    @pytest.mark.parametrize("distribution", ["probit", "normal", "logit"])
    @pytest.mark.parametrize("noise_std", [0.1, 1, 2])
    @pytest.mark.parametrize("intercept", [True, False])
    @pytest.mark.parametrize("seed", [10, 12])
    def test_dataframe(self, graph, distribution, noise_std, intercept, seed):
        """
        Tests equivalence of dataframe wrapper
        """
        data = generate_binary_data(
            graph,
            100,
            distribution,
            noise_scale=noise_std,
            seed=seed,
            intercept=intercept,
        )
        df = generate_binary_dataframe(
            graph,
            100,
            distribution,
            noise_scale=noise_std,
            seed=seed,
            intercept=intercept,
        )

        assert np.array_equal(data, df[list(graph.nodes())].values)

    @pytest.mark.parametrize("num_nodes", (2, 3, 10, 20, 30))
    @pytest.mark.parametrize("seed", (10, 20, 30))
    def test_independence(self, graph_gen, seed, num_nodes):
        """
        test whether the relation is accurate, implicitely tests sequence of
        nodes.
        """

        sm = graph_gen(num_nodes=num_nodes, seed=seed, weight=None)
        nodes = sm.nodes()

        df = generate_binary_dataframe(
            sm,
            n_samples=100000,
            distribution="normal",
            seed=seed,
            noise_scale=0.5,
            intercept=False,
        )

        tol = 0.05

        for node in nodes:
            if node == "aa":
                continue
            joint_proba, factored_proba = calculate_proba(df, "aa", node)
            if node == "ab":
                # this is the only link
                assert not np.isclose(
                    joint_proba, factored_proba, atol=tol, rtol=0
                ), df.mean()
            else:
                assert np.isclose(joint_proba, factored_proba, atol=tol, rtol=0)


class TestGenerateCategoricalData:
    @pytest.mark.parametrize("distribution", ["probit", "normal", "logit", "gumbel"])
    def test_returns_dataframe(self, distribution):
        """ Return value is an ndarray - test over all sem_types """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        ndarray = generate_categorical_dataframe(
            sm, distribution=distribution, n_samples=10
        )
        assert isinstance(ndarray, pd.DataFrame)

    def test_bad_distribution_type(self):
        """ Test that invalid sem-type other than "probit", "normal", "logit", "gumbel" is not accepted """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        with pytest.raises(ValueError, match="Unknown categorical distribution"):
            generate_categorical_dataframe(
                sm, distribution="invalid", n_samples=10, seed=10
            )

    @pytest.mark.parametrize(
        "num_nodes,n_categories", list(product([5, 10, 15], [3, 5, 7]))
    )
    def test_number_of_columns(self, num_nodes, n_categories):
        """ Length of dataframe is in the correct shape"""
        graph = StructureModel()
        edges = [(n, n + 1, 1) for n in range(num_nodes - 1)]
        graph.add_weighted_edges_from(edges)

        data = generate_categorical_dataframe(
            graph, 100, seed=10, n_categories=n_categories
        )
        assert data.shape[1] == (num_nodes * n_categories)

    @pytest.mark.parametrize("num_samples", [5, 10, 15])
    def test_number_of_samples(self, num_samples, graph):
        """ Assert number of samples generated (rows) = num_samples """
        data = generate_categorical_dataframe(graph, num_samples, "logit", 1, seed=10)
        assert len(data) == num_samples

    @pytest.mark.parametrize(
        "distribution, n_categories",
        list(product(["logit", "probit", "normal", "gumbel"], [3, 5, 7])),
    )
    def test_baseline_probability(self, graph, distribution, n_categories):
        """ Test whether probability centered around 50% if no intercept given"""
        graph = StructureModel()
        graph.add_nodes_from(["A"])
        data = generate_categorical_dataframe(
            graph,
            10000,
            distribution=distribution,
            n_categories=n_categories,
            noise_scale=1.0,
            seed=10,
            intercept=False,
        )
        # without intercept, the probabilities should be fairly uniform
        assert np.allclose(data.mean(axis=0), 1 / n_categories, atol=0.01, rtol=0)

    @pytest.mark.parametrize(
        "distribution,n_categories",
        list(product(["logit", "probit", "normal", "gumbel"], [3, 5, 7])),
    )
    def test_intercept_probability(self, graph, distribution, n_categories):
        """ Test whether probability is not centered around 50% when using an intercept"""
        graph = StructureModel()
        graph.add_nodes_from(["A"])
        data = generate_categorical_dataframe(
            graph,
            1000000,
            distribution=distribution,
            n_categories=n_categories,
            noise_scale=0.1,
            seed=10,
            intercept=True,
        )
        assert not np.allclose(data.mean(axis=0), 1 / n_categories, atol=0.01, rtol=0)

    @pytest.mark.parametrize("n_categories", (2, 10,))
    @pytest.mark.parametrize("distribution", ["probit", "logit"])
    def test_intercept(self, distribution, n_categories):
        graph = StructureModel()
        graph.add_node("A")

        data_noint = generate_categorical_dataframe(
            graph,
            100000,
            distribution,
            noise_scale=0.1,
            n_categories=n_categories,
            seed=10,
            intercept=False,
        )
        data_intercept = generate_categorical_dataframe(
            graph,
            100000,
            distribution,
            noise_scale=0.1,
            n_categories=n_categories,
            seed=10,
            intercept=True,
        )

        assert np.all(
            ~np.isclose(
                data_intercept.mean(axis=0), data_noint.mean(axis=0), atol=0.05, rtol=0
            )
        )

    @pytest.mark.parametrize("num_nodes", (3, 6))
    @pytest.mark.parametrize("seed", (10, 20))
    @pytest.mark.parametrize("n_categories", (2, 6,))
    @pytest.mark.parametrize("distribution", ["probit", "logit"])
    def test_independence(self, graph_gen, seed, num_nodes, n_categories, distribution):
        """
        test whether the relation is accurate, implicitely tests sequence of
        nodes.
        """
        sm = graph_gen(num_nodes=num_nodes, seed=seed, weight=None)
        nodes = sm.nodes()

        df = generate_categorical_dataframe(
            sm,
            n_samples=100000,
            distribution=distribution,
            n_categories=n_categories,
            seed=seed,
            noise_scale=1,
            intercept=False,
        )

        tol = 0.05

        # independent links
        for node in nodes:
            if node == "aa":
                continue
            joint_proba, factored_proba = calculate_proba(df, "aa_0", node + "_0")
            if node == "ab":
                assert not np.isclose(joint_proba, factored_proba, rtol=tol, atol=0)
            else:
                assert np.isclose(joint_proba, factored_proba, rtol=tol, atol=0)


class TestMixedDataGen:
    def test_run(self, graph, schema):
        df = sem_generator(
            graph=graph,
            schema=schema,
            default_type="continuous",
            noise_std=1.0,
            n_samples=1000,
            intercept=False,
            seed=12,
        )

        # test binary:
        assert df[0].nunique() == 2
        assert df[0].nunique() == 2

        # test categorical:
        for col in ["1_{}".format(i) for i in range(3)]:
            assert df[col].nunique() == 2
        assert len([x for x in df.columns if isinstance(x, str) and "1_" in x]) == 3

        for col in ["5_{}".format(i) for i in range(5)]:
            assert df[col].nunique() == 2
        assert len([x for x in df.columns if isinstance(x, str) and "5_" in x]) == 5

        # test continuous
        assert df[3].nunique() == 1000
        assert df[4].nunique() == 1000

    def test_graph_not_a_dag(self):
        graph = StructureModel()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

        with pytest.raises(ValueError, match="Provided graph is not a DAG"):
            _ = sem_generator(graph=graph)

    def test_not_permissible_type(self, graph):
        schema = {
            0: "unknown data type",
        }
        with pytest.raises(ValueError, match="Unknown data type"):
            _ = sem_generator(
                graph=graph,
                schema=schema,
                default_type="continuous",
                noise_std=1.0,
                n_samples=1000,
                intercept=False,
                seed=12,
            )

    def test_missing_cardinality(self, graph):
        schema = {
            0: "categorical",
            1: "categorical:3",
            5: "categorical:5",
        }
        with pytest.raises(ValueError, match="Missing cardinality for categorical"):
            _ = sem_generator(
                graph=graph,
                schema=schema,
                default_type="continuous",
                noise_std=1.0,
                n_samples=1000,
                intercept=False,
                seed=12,
            )

    def test_missing_default_type(self, graph):
        with pytest.raises(ValueError, match="Unknown default data type"):
            _ = sem_generator(
                graph=graph,
                schema=schema,
                default_type="unknown",
                noise_std=1.0,
                n_samples=1000,
                intercept=False,
                seed=12,
            )

    def test_incorrect_weight_dist(self):
        sm = StructureModel()
        nodes = list(str(x) for x in range(6))
        np.random.shuffle(nodes)
        sm.add_nodes_from(nodes)

        sm.add_weighted_edges_from([("0", "1", None), ("2", "4", None)])

        with pytest.raises(ValueError, match="Unknown weight distribution"):
            _ = sem_generator(
                graph=sm,
                schema=None,
                default_type="continuous",
                distributions={"weight": "unknown"},
                noise_std=2.0,
                n_samples=1000,
                intercept=False,
                seed=10,
            )

    def test_incorrect_intercept_dist(self, graph):
        with pytest.raises(ValueError, match="Unknown intercept distribution"):
            _ = sem_generator(
                graph=graph,
                schema=None,
                default_type="continuous",
                distributions={"intercept": "unknown"},
                noise_std=2.0,
                n_samples=10,
                intercept=True,
                seed=10,
            )

    # def test_mixed_type_independence(self):
    @pytest.mark.parametrize("seed", (10, 20))
    @pytest.mark.parametrize("n_categories", (2, 5,))
    @pytest.mark.parametrize("weight_distribution", ["uniform", "gaussian"])
    @pytest.mark.parametrize("intercept_distribution", ["uniform", "gaussian"])
    def test_mixed_type_independence(
        self, seed, n_categories, weight_distribution, intercept_distribution
    ):
        """
        Test whether the relation is accurate, implicitly tests sequence of
        nodes.
        """
        np.random.seed(seed)

        sm = StructureModel()
        nodes = list(str(x) for x in range(6))
        np.random.shuffle(nodes)
        sm.add_nodes_from(nodes)
        # binary -> categorical
        sm.add_weighted_edges_from([("0", "1", 10)])
        # binary -> continuous
        sm.add_weighted_edges_from([("2", "4", None)])

        schema = {
            "0": "binary",
            "1": "categorical:{}".format(n_categories),
            "2": "binary",
            "4": "continuous",
            "5": "categorical:{}".format(n_categories),
        }

        df = sem_generator(
            graph=sm,
            schema=schema,
            default_type="continuous",
            distributions={
                "weight": weight_distribution,
                "intercept": intercept_distribution,
            },
            noise_std=2,
            n_samples=100000,
            intercept=True,
            seed=seed,
        )

        atol = 0.05  # 5% difference bewteen joint & factored!
        # 1. dependent links
        # 0 -> 1 (we look at the class with the highest deviation from uniform
        # to avoid small values)
        c, _ = max(
            [
                (c, np.abs(df["1_{}".format(c)].mean() - 1 / n_categories))
                for c in range(n_categories)
            ],
            key=operator.itemgetter(1),
        )
        joint_proba, factored_proba = calculate_proba(df, "0", "1_{}".format(c))
        assert not np.isclose(joint_proba, factored_proba, rtol=0, atol=atol)
        # 2 -> 4
        assert not np.isclose(
            df["4"].mean(), df["4"][df["2"] == 1].mean(), rtol=0, atol=atol
        )

        tol = 0.15  # relative tolerance of +- 15% of the
        # 2. independent links
        # categorical
        c, _ = max(
            [
                (c, np.abs(df["1_{}".format(c)].mean() - 1 / n_categories))
                for c in range(n_categories)
            ],
            key=operator.itemgetter(1),
        )
        joint_proba, factored_proba = calculate_proba(df, "0", "5_{}".format(c))
        assert np.isclose(joint_proba, factored_proba, rtol=tol, atol=0)

        # binary
        joint_proba, factored_proba = calculate_proba(df, "0", "2")
        assert np.isclose(joint_proba, factored_proba, rtol=tol, atol=0)

        # categorical
        c, _ = max(
            [
                (c, np.abs(df["1_{}".format(c)].mean() - 1 / n_categories))
                for c in range(n_categories)
            ],
            key=operator.itemgetter(1),
        )
        d, _ = max(
            [
                (d, np.abs(df["5_{}".format(d)].mean() - 1 / n_categories))
                for d in range(n_categories)
            ],
            key=operator.itemgetter(1),
        )
        joint_proba, factored_proba = calculate_proba(
            df, "1_{}".format(d), "5_{}".format(c)
        )
        assert np.isclose(joint_proba, factored_proba, rtol=tol, atol=0)

        # continuous
        # for gaussian distributions, zero variance is equivalent to independence
        assert np.isclose(df[["3", "4"]].corr().values[0, 1], 0, atol=tol)


def calculate_proba(
    df: Union[pd.DataFrame, np.ndarray], col_0: Hashable, col_1: Hashable
) -> Tuple[float, float]:
    if isinstance(df, pd.DataFrame):
        marginal_0 = df[col_0].mean()
        marginal_1 = df[col_1].mean()
        joint_proba = (df[col_0] * df[col_1]).mean()
    else:
        marginal_0 = df[:, col_0].mean()
        marginal_1 = df[:, col_1].mean()
        joint_proba = (df[:, col_0] * df[:, col_1]).mean()

    factored_proba = marginal_0 * marginal_1
    return joint_proba, factored_proba
