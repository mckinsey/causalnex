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
import re
import string
from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.stats import anderson, stats
from sklearn.gaussian_process.kernels import RBF

from causalnex.structure import StructureModel
from causalnex.structure.data_generators import (
    gen_stationary_dyn_net_and_df,
    generate_binary_data,
    generate_binary_dataframe,
    generate_categorical_dataframe,
    generate_continuous_data,
    generate_continuous_dataframe,
    generate_count_dataframe,
    generate_dataframe_dynamic,
    generate_structure,
    generate_structure_dynamic,
)
from tests.structure.data_generators.test_core import calculate_proba


@pytest.fixture
def graph():
    graph = StructureModel()
    edges = [(n, n + 1, 1) for n in range(5)]
    graph.add_weighted_edges_from(edges)
    return graph


@pytest.fixture
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
        """Test that invalid sem-type other than "gaussian", "normal", "student-t",
        "exponential", "gumbel" is not accepted"""
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
    @pytest.mark.parametrize("noise_scale", [0.0, 0.1])
    def test_intercept(self, distribution, noise_scale):
        graph = StructureModel()
        graph.add_node("123")

        data_noint = generate_continuous_data(
            graph,
            n_samples=100000,
            distribution=distribution,
            noise_scale=noise_scale,
            seed=10,
            intercept=False,
        )
        data_intercept = generate_continuous_data(
            graph,
            n_samples=100000,
            distribution=distribution,
            noise_scale=noise_scale,
            seed=10,
            intercept=True,
        )
        assert not np.isclose(data_noint[:, 0].mean(), data_intercept[:, 0].mean())
        assert np.isclose(data_noint[:, 0].std(), data_intercept[:, 0].std(), rtol=0.01)

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
    @pytest.mark.parametrize("kernel", [None, RBF(1)])
    def test_dataframe(self, graph, distribution, noise_std, intercept, seed, kernel):
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
            kernel=kernel,
        )
        df = generate_continuous_dataframe(
            graph,
            1000,
            distribution,
            noise_scale=noise_std,
            seed=seed,
            intercept=intercept,
            kernel=kernel,
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
    @pytest.mark.parametrize("noise_scale", [0.0, 0.1])
    def test_intercept(self, distribution, noise_scale):
        graph = StructureModel()
        graph.add_node("123")

        data_noint = generate_binary_data(
            graph,
            100000,
            distribution,
            noise_scale=noise_scale,
            seed=10,
            intercept=False,
        )
        data_intercept = generate_binary_data(
            graph,
            100000,
            distribution,
            noise_scale=noise_scale,
            seed=10,
            intercept=True,
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
    @pytest.mark.parametrize("kernel", [None, RBF(1)])
    def test_dataframe(self, graph, distribution, noise_std, intercept, seed, kernel):
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
            kernel=kernel,
        )
        df = generate_binary_dataframe(
            graph,
            100,
            distribution,
            noise_scale=noise_std,
            seed=seed,
            intercept=intercept,
            kernel=kernel,
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

    @pytest.mark.parametrize("distribution", ["logit", "probit", "normal", "gumbel"])
    @pytest.mark.parametrize("noise_std", [0.1, 1, 2])
    @pytest.mark.parametrize("intercept", [True, False])
    @pytest.mark.parametrize("seed", [10, 42])
    @pytest.mark.parametrize("kernel", [None, RBF(1)])
    @pytest.mark.parametrize(
        "n_categories",
        (
            2,
            10,
        ),
    )
    def test_dataframe(
        self, graph, distribution, noise_std, intercept, seed, kernel, n_categories
    ):
        """
        Tests equivalence of dataframe wrapper
        """
        data = generate_categorical_dataframe(
            graph,
            100,
            distribution,
            noise_scale=noise_std,
            seed=seed,
            intercept=intercept,
            kernel=kernel,
            n_categories=n_categories,
        )
        df = generate_categorical_dataframe(
            graph,
            100,
            distribution,
            noise_scale=noise_std,
            seed=seed,
            intercept=intercept,
            kernel=kernel,
            n_categories=n_categories,
        )

        cols = []
        for node in graph.nodes():
            for cat in range(n_categories):
                cols.append(f"{node}_{cat}")
        assert np.array_equal(data, df[cols].values)

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

    @pytest.mark.parametrize(
        "n_categories",
        (
            2,
            10,
        ),
    )
    @pytest.mark.parametrize("distribution", ["probit", "logit"])
    @pytest.mark.parametrize("noise_scale", [0.0, 0.1])
    def test_intercept(self, distribution, n_categories, noise_scale):
        graph = StructureModel()
        graph.add_node("A")

        data_noint = generate_categorical_dataframe(
            graph,
            100000,
            distribution,
            noise_scale=noise_scale,
            n_categories=n_categories,
            seed=10,
            intercept=False,
        )
        data_intercept = generate_categorical_dataframe(
            graph,
            100000,
            distribution,
            noise_scale=noise_scale,
            n_categories=n_categories,
            seed=10,
            intercept=True,
        )

        # NOTE: as n_categories increases, the probability that at least one category with
        # intercept=True will be the same as intercept=False -> 1.0
        num_similar = np.isclose(
            data_intercept.mean(axis=0), data_noint.mean(axis=0), atol=0.05, rtol=0
        ).sum()
        assert num_similar < n_categories / 2

    @pytest.mark.parametrize("num_nodes", (3, 6))
    @pytest.mark.parametrize("seed", (10, 20))
    @pytest.mark.parametrize(
        "n_categories",
        (
            2,
            6,
        ),
    )
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


class TestGenerateCountData:
    def test_zero_lambda(self):
        """
        A wrong initialisation could lead to counts always being zero if they dont
        have parents.
        """
        graph = StructureModel()
        graph.add_nodes_from(list(range(20)))
        df = generate_count_dataframe(graph, 10000)
        assert not np.any(df.mean() == 0)

    @pytest.mark.parametrize("intercept", [True, False])
    @pytest.mark.parametrize("seed", [10, 12])
    @pytest.mark.parametrize("kernel", [None, RBF(1)])
    @pytest.mark.parametrize(
        "zero_inflation_factor", [int(0), 0.0, 0.01, 0.1, 0.5, 1.0, int(1)]
    )
    def test_dataframe(self, graph, intercept, seed, kernel, zero_inflation_factor):
        """
        Tests equivalence of dataframe wrapper
        """
        data = generate_count_dataframe(
            graph,
            100,
            zero_inflation_factor=zero_inflation_factor,
            seed=seed,
            intercept=intercept,
            kernel=kernel,
        )
        df = generate_count_dataframe(
            graph,
            100,
            zero_inflation_factor=zero_inflation_factor,
            seed=seed,
            intercept=intercept,
            kernel=kernel,
        )

        assert np.array_equal(data, df[list(graph.nodes())].values)


class TestGenerateStructureDynamic:
    @pytest.mark.parametrize("num_nodes", (10, 20))
    @pytest.mark.parametrize("p", [1, 10])
    @pytest.mark.parametrize("degree_intra, degree_inter", [(3, 0), (0, 3), (1, 1)])
    def test_all_nodes_in_structure(self, num_nodes, p, degree_intra, degree_inter):
        """both intra- and iter-slice nodes should be in the structure"""
        g = generate_structure_dynamic(num_nodes, p, degree_intra, degree_inter)
        assert np.all(
            [
                f"{var}_lag{l_val}" in g.nodes
                for l_val in range(p + 1)
                for var in range(num_nodes)
            ]
        )

    def test_naming_nodes(self):
        """Nodes should have the format {var}_lag{l}"""
        g = generate_structure_dynamic(5, 3, 3, 4)
        pattern = re.compile(r"[0-5]_lag[0-3]")
        for node in g.nodes:
            match = pattern.match(node)
            assert match and (match.group() == node)

    def test_degree_zero_implies_no_edges(self):
        """If the degree is zero, zero edges are generated.
        We test this is true for intra edges (ending in 'lag0') and inter edges
        """
        g = generate_structure_dynamic(15, 3, 0, 4)  # No intra edges
        lags = [(u.split("_lag")[1], v.split("_lag")[1]) for u, v in g.edges]
        assert np.all([el[0] != "0" for el in lags])
        g = generate_structure_dynamic(15, 3, 4, 0)
        lags = [(u.split("_lag")[1], v.split("_lag")[1]) for u, v in g.edges]
        assert np.all([el == ("0", "0") for el in lags])  # only Intra edges
        g = generate_structure_dynamic(15, 3, 0, 0)  # no edges
        assert len(g.edges) == 0

    def test_edges_have_weights(self):
        """all edges must have weight values as floats or int"""
        g = generate_structure_dynamic(10, 3, 4, 4)  # No intra edges
        ws = [w for _, _, w in g.edges(data="weight")]
        assert np.all([isinstance(w, (float, int)) for w in ws])

    def test_raise_error_if_wrong_graph_type(self):
        """if the graph_type chosen is not among the options available, raise error"""
        with pytest.raises(
            ValueError,
            match=r"Unknown graph type some_type\. "
            r"Available types are \['erdos-renyi', 'barabasi-albert', 'full'\]",
        ):
            generate_structure_dynamic(10, 10, 10, 10, graph_type_intra="some_type")
        with pytest.raises(
            ValueError,
            match=r"Unknown inter-slice graph type `some_type`\. "
            "Valid types are 'erdos-renyi' and 'full'",
        ):
            generate_structure_dynamic(10, 10, 10, 10, graph_type_inter="some_type")

    def test_raise_error_if_min_greater_than_max(self):
        """if min > max,raise error"""
        with pytest.raises(
            ValueError,
            match="Absolute minimum weight must be "
            r"less than or equal to maximum weight\: 3 \> 2",
        ):
            generate_structure_dynamic(10, 10, 10, 10, w_min_inter=3, w_max_inter=2)

    @pytest.mark.parametrize("num_nodes", (10, 20))
    @pytest.mark.parametrize("p", [1, 10])
    def test_full_graph_type(self, num_nodes, p):
        """all the connections from past variables to current variables should be there if using `full` graph_type"""
        g = generate_structure_dynamic(num_nodes, p, 4, 4, graph_type_inter="full")
        lagged_edges = sorted((u, v) for u, v in g.edges if int(u.split("_lag")[1]) > 0)
        assert lagged_edges == sorted(
            (f"{v_f}_lag{l_}", f"{v_t}_lag0")
            for l_ in range(1, p + 1)
            for v_f in range(num_nodes)  # var from
            for v_t in range(num_nodes)  # var to
        )


class TestGenerateDataframeDynamic:
    @pytest.mark.parametrize(
        "sem_type", ["linear-gauss", "linear-exp", "linear-gumbel"]
    )
    def test_returns_dateframe(self, sem_type):
        """ Return value is an ndarray - test over all sem_types """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure_dynamic(d_nodes, 2, degree, degree, graph_type)
        data = generate_dataframe_dynamic(sm, sem_type=sem_type, n_samples=10)
        assert isinstance(data, pd.DataFrame)

    def test_bad_sem_type(self):
        """ Test that invalid sem-type other than "linear-gauss", "linear-exp", "linear-gumbel" is not accepted """
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure_dynamic(d_nodes, 2, degree, degree, graph_type)
        with pytest.raises(
            ValueError,
            match="unknown sem type invalid. Available types are:"
            r" \('linear-gauss', 'linear-exp', 'linear-gumbel'\)",
        ):
            generate_dataframe_dynamic(sm, sem_type="invalid", n_samples=10)

    @pytest.mark.parametrize("p", [0, 1, 2])
    def test_labels_correct(self, p):
        graph_type, degree, d_nodes = "erdos-renyi", 4, 10
        sm = generate_structure_dynamic(d_nodes, p, degree, degree, graph_type)
        data = generate_dataframe_dynamic(sm, sem_type="linear-gauss", n_samples=10)
        intra_nodes = sorted([el for el in sm.nodes if "_lag0" in el])
        inter_nodes = sorted([el for el in sm.nodes if "_lag0" not in el])
        assert sorted(data.columns) == sorted(list(inter_nodes) + list(intra_nodes))


class TestGenerateStationaryDynamicStructureAndSamples:
    def test_wta(self):
        with pytest.warns(
            UserWarning, match="Could not simulate data, returning constant dataframe"
        ):
            gen_stationary_dyn_net_and_df(
                w_min_inter=1, w_max_inter=2, max_data_gen_trials=2
            )

    @pytest.mark.parametrize("seed", [2, 3, 5])
    def test_seems_stationary(self, seed):
        np.random.seed(seed)
        _, df, _, _ = gen_stationary_dyn_net_and_df(
            w_min_inter=0.1, w_max_inter=0.2, max_data_gen_trials=2
        )
        assert np.all(df.max() - df.min() < 10)

    def test_error_if_wmin_less_wmax(self):
        with pytest.raises(
            ValueError,
            match="Absolute minimum weight must be less than or equal to maximum weight: 2 > 1",
        ):
            gen_stationary_dyn_net_and_df(
                w_min_inter=2, w_max_inter=1, max_data_gen_trials=2
            )

    def test_dense_networks(self):
        """dense network are more likely to be non stationary. we check that the simulator is still able to provide a
        stationary time-deries in that case.

        If df contain only ones it means that the generator failed to obtain a stationary structure"""
        np.random.seed(4)
        _, df, _, _ = gen_stationary_dyn_net_and_df(
            n_samples=1000,
            p=1,
            w_min_inter=0.2,
            w_max_inter=0.5,
            max_data_gen_trials=10,
            degree_intra=4,
            degree_inter=7,
        )
        assert np.any(np.ones(df.shape) != df)

    def test_fail_to_find_stationary_network(self):
        """if fails to find suitable network, returns dataset of ones"""
        np.random.seed(5)
        _, df, _, _ = gen_stationary_dyn_net_and_df(
            n_samples=1000,
            p=1,
            w_min_inter=0.6,
            w_max_inter=0.6,
            max_data_gen_trials=20,
            degree_intra=4,
            degree_inter=7,
        )
        assert np.any(np.ones(df.shape) == df)
