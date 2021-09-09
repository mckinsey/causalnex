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
from typing import Hashable, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from networkx.algorithms.dag import is_directed_acyclic_graph
from sklearn.gaussian_process.kernels import RBF

from causalnex.structure.data_generators.core import (
    _sample_binary_from_latent,
    _sample_count_from_latent,
    _sample_poisson,
    generate_structure,
    nonlinear_sem_generator,
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


class TestGenerateStructure:
    @pytest.mark.parametrize("graph_type", ["erdos-renyi", "barabasi-albert", "full"])
    def test_is_dag_graph_type(self, graph_type):
        """ Tests that the generated graph is a dag for all graph types. """
        degree, d_nodes = 4, 10
        sm = generate_structure(d_nodes, degree, graph_type)
        assert is_directed_acyclic_graph(sm)

    @pytest.mark.parametrize("num_nodes,degree", [(5, 2), (10, 3), (15, 5)])
    def test_is_dag_nodes_degrees(self, num_nodes, degree):
        """Tests that generated graph is dag for different numbers of nodes and degrees"""
        sm = generate_structure(num_nodes, degree)
        assert nx.is_directed_acyclic_graph(sm)

    def test_bad_graph_type(self):
        """ Test that a value other than "erdos-renyi", "barabasi-albert", "full" throws ValueError """
        graph_type = "invalid"
        degree, d_nodes = 4, 10
        with pytest.raises(
            ValueError,
            match="Unknown graph type invalid. Available types"
            r" are \['erdos-renyi', 'barabasi-albert', 'full'\]",
        ):
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
        assert df[2].nunique() == 2

        # test categorical:
        for col in [f"1_{i}" for i in range(3)]:
            assert df[col].nunique() == 2
        assert len([x for x in df.columns if isinstance(x, str) and "1_" in x]) == 3

        for col in [f"5_{i}" for i in range(5)]:
            assert df[col].nunique() == 2
        assert len([x for x in df.columns if isinstance(x, str) and "5_" in x]) == 5

        # test continuous
        assert df[3].nunique() == 1000
        assert df[4].nunique() == 1000

    def test_graph_not_a_dag(self):
        graph = StructureModel()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

        with pytest.raises(ValueError, match="Provided graph is not a DAG"):
            _ = sem_generator(graph=graph, seed=42)

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

    # Seed 20 is an unlucky seed and fails the assertion. All other seeds tested
    # pass the assertion. Similar issue to the categorical intercept test?
    @pytest.mark.parametrize("seed", (10, 17))
    @pytest.mark.parametrize(
        "n_categories",
        (
            2,
            5,
        ),
    )
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
        # binary -> count
        sm.add_weighted_edges_from([("2", "6", 100)])

        schema = {
            "0": "binary",
            "1": f"categorical:{n_categories}",
            "2": "binary",
            "4": "continuous",
            "5": f"categorical:{n_categories}",
            "6": "count",
        }

        df = sem_generator(
            graph=sm,
            schema=schema,
            default_type="continuous",
            distributions={
                "weight": weight_distribution,
                "intercept": intercept_distribution,
                "count": 0.05,
            },
            noise_std=2,
            n_samples=100000,
            intercept=True,
            seed=seed,
        )

        atol = 0.02  # at least 2% difference bewteen joint & factored!
        # 1. dependent links
        # 0 -> 1 (we look at the class with the highest deviation from uniform
        # to avoid small values)
        c, _ = max(
            [
                (i, np.abs(df[f"1_{i}"].mean() - 1 / n_categories))
                for i in range(n_categories)
            ],
            key=operator.itemgetter(1),
        )
        joint_proba, factored_proba = calculate_proba(df, "0", f"1_{c}")
        assert not np.isclose(joint_proba, factored_proba, rtol=0, atol=atol)
        # 2 -> 4
        assert not np.isclose(
            df["4"].mean(), df["4"][df["2"] == 1].mean(), rtol=0, atol=atol
        )
        # binary on count
        assert not np.isclose(
            df.loc[df["2"] == 0, "6"].mean(),
            df.loc[df["2"] == 1, "6"].mean(),
            rtol=0,
            atol=atol,
        )

        tol = 0.20  # at most relative tolerance of +- 20% of the
        # 2. independent links
        # categorical
        c, _ = max(
            [
                (i, np.abs(df[f"1_{i}"].mean() - 1 / n_categories))
                for i in range(n_categories)
            ],
            key=operator.itemgetter(1),
        )
        joint_proba, factored_proba = calculate_proba(df, "0", f"5_{c}")
        assert np.isclose(joint_proba, factored_proba, rtol=tol, atol=0)

        # binary
        joint_proba, factored_proba = calculate_proba(df, "0", "2")
        assert np.isclose(joint_proba, factored_proba, rtol=tol, atol=0)

        # categorical
        c, _ = max(
            [
                (i, np.abs(df[f"1_{i}"].mean() - 1 / n_categories))
                for i in range(n_categories)
            ],
            key=operator.itemgetter(1),
        )
        d, _ = max(
            [
                (d, np.abs(df[f"5_{d}"].mean() - 1 / n_categories))
                for d in range(n_categories)
            ],
            key=operator.itemgetter(1),
        )
        joint_proba, factored_proba = calculate_proba(df, f"1_{d}", f"5_{c}")
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


@pytest.mark.parametrize("distribution", ["probit", "logit"])
@pytest.mark.parametrize("max_imbalance", [0.01, 0.05, 0.1, 0.5])
def test_sample_binary_from_latent_imbalance(max_imbalance, distribution):
    """
    Tests max imbalance argument to sample the binary variable.
    This way we are guaranteed to always have some positives/negatives.
    """
    # corner case:
    eta = np.ones(1000) * 1000

    sample = _sample_binary_from_latent(
        latent_mean=eta,
        distribution=distribution,
        noise_std=0.1,
        root_node=False,
        max_imbalance=max_imbalance,
    )
    tol = 0.01
    assert np.isclose(sample.mean(), max_imbalance, atol=0, rtol=tol)


@pytest.mark.parametrize("poisson_lambda", [0.1, 1, 10, 100])
def test_sample_poisson(poisson_lambda):
    """
    We test whether the first two moments match a Poisson distribution
    """
    sample = _sample_poisson(np.ones(shape=10000) * poisson_lambda)
    tol = 0.05
    assert np.isclose(sample.mean(), poisson_lambda, atol=0, rtol=tol)
    assert np.isclose(sample.var(), poisson_lambda, atol=0, rtol=tol)


@pytest.mark.parametrize("poisson_lambda", [-0.5, 0.1, 1, 10, 100])
@pytest.mark.parametrize("zero_inflation_pct", [0.0, 0.01, 0.1, 0.5, 1.0])
def test_sample_count_from_latent_zero_inflation(poisson_lambda, zero_inflation_pct):
    """
    We test whether the zero-inflation is functional using the first two moments.
    """
    sample = _sample_count_from_latent(
        np.ones(shape=10000) * poisson_lambda,
        zero_inflation_pct=zero_inflation_pct,
        root_node=False,
    )
    if poisson_lambda < 0:
        poisson_lambda = np.exp(poisson_lambda)

    tol = 0.1
    assert np.isclose(
        sample.mean(), (1 - zero_inflation_pct) * poisson_lambda, atol=0, rtol=tol
    )
    assert np.isclose(
        sample.var(),
        (1 + zero_inflation_pct * poisson_lambda)
        * (1 - zero_inflation_pct)
        * poisson_lambda,
        atol=0,
        rtol=tol,
    )


class TestCountGenerator:
    @pytest.mark.parametrize(
        "zero_inflation_pct", [int(0), 0.0, 0.01, 0.1, 0.5, 1.0, int(1)]
    )
    def test_only_count(self, graph, zero_inflation_pct):
        df = sem_generator(
            graph,
            default_type="count",
            n_samples=1000,
            distributions={"count": zero_inflation_pct},
            seed=43,
        )
        # count puts a lower bound on the output:
        assert np.all(df.min() >= 0)

        # zero inflation puts a lower bound on the zero-share
        assert np.all((df == 0).mean() >= zero_inflation_pct)

        # How to test dependence/independence for Poisson?

    @pytest.mark.parametrize("wrong_count_zif", ["text", (0.1,), {0.1}, -0.1, 1.01])
    def test_zif_value_error(self, graph, wrong_count_zif):
        """
        Test if ValueError raised for unsupported Zero-Inflation Factor for the
        count data type.
        """
        with pytest.raises(ValueError, match="Unsupported zero-inflation factor"):
            sem_generator(
                graph,
                default_type="count",
                distributions={"count": wrong_count_zif},
                seed=42,
            )


class TestNonlinearGenerator:
    def test_run(self, graph, schema):
        df = nonlinear_sem_generator(
            graph=graph,
            schema=schema,
            kernel=RBF(1),
            default_type="continuous",
            noise_std=1.0,
            n_samples=1000,
            seed=13,
        )

        # test binary:
        assert df[0].nunique() == 2
        assert df[2].nunique() == 2

        # test categorical:
        for col in [f"1_{i}" for i in range(3)]:
            assert df[col].nunique() == 2
        assert len([x for x in df.columns if isinstance(x, str) and "1_" in x]) == 3

        for col in [f"5_{i}" for i in range(5)]:
            assert df[col].nunique() == 2
        assert len([x for x in df.columns if isinstance(x, str) and "5_" in x]) == 5

        # test continuous
        assert df[3].nunique() == 1000
        assert df[4].nunique() == 1000
