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

import logging

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.optimize as sopt
from mock import patch

from causalnex.structure import StructureModel
from causalnex.structure.data_generators import (
    generate_binary_data,
    generate_binary_dataframe,
    generate_continuous_dataframe,
    generate_count_dataframe,
    generate_structure,
)
from causalnex.structure.pytorch.notears import from_numpy, from_pandas


class TestFromPandas:
    """Test behaviour of the from_pandas method"""

    def test_all_columns_in_structure(self, train_data_idx):
        """Every columns that is in the data should become a node in the learned structure"""

        g = from_pandas(train_data_idx)
        assert len(g.nodes) == len(train_data_idx.columns)

    def test_isolated_nodes_exist(self, train_data_idx):
        """Isolated nodes should still be in the learned structure"""

        g = from_pandas(train_data_idx, w_threshold=1.0)
        assert len(g.nodes) == len(train_data_idx.columns)

    def test_expected_structure_learned(self, train_data_idx, train_model):
        """Given a small data set that can be examined by hand, the structure should be deterministic"""

        g = from_pandas(train_data_idx, w_threshold=0.16)
        assert set(g.edges) == set(train_model.edges)

    def test_empty_data_raises_error(self):
        """
        Providing an empty data set should result in a Value Error explaining that data must not be empty.
        This error is useful to catch and handle gracefully, because otherwise the user would experience
        misleading division by zero, or unpacking errors.
        """

        with pytest.raises(ValueError):
            from_pandas(pd.DataFrame(data=[], columns=["a"]))

    def test_non_numeric_data_raises_error(self):
        """Only numeric data frames should be supported"""

        with pytest.raises(ValueError, match="All columns must have numeric data.*"):
            from_pandas(pd.DataFrame(data=["x"], columns=["a"]))

    def test_single_iter_gets_converged_fail_warnings(self, caplog, train_data_idx):
        """
        With a single iteration on this dataset, learn_structure fails to converge and should give warnings.
        """

        with caplog.at_level(logging.WARNING):
            from_numpy(train_data_idx.values, max_iter=1)
        assert "Failed to converge. Consider increasing max_iter." in caplog.text

    def test_certain_relationships_get_near_certain_weight(self):
        """If observations reliably show a==b and !a==!b then the relationship from a->b should be certain"""

        data = pd.DataFrame([[0, 1] for _ in range(10)], columns=["a", "b"])
        g = from_pandas(data)
        assert all(
            0.99 <= weight <= 1
            for u, v, weight in g.edges(data="weight")
            if u == 0 and v == 1
        )

    def test_inverse_relationships_get_negative_weight(self):
        """If observations indicate a==!b and b==!a then the weight of the relationship from a-> should be negative"""

        data = pd.DataFrame([[0, 1] for _ in range(10)], columns=["a", "b"])
        data.append(pd.DataFrame([[1, 0] for _ in range(10)], columns=["a", "b"]))
        g = from_pandas(data)
        assert all(
            weight < 0
            for u, v, weight in g.edges(data="mean_effect")
            if u == 0 and v == 1
        )

    def test_no_cycles(self, train_data_idx):
        """
        The learned structure should be acyclic
        """

        g = from_pandas(train_data_idx, w_threshold=0.25)
        assert nx.algorithms.is_directed_acyclic_graph(g)

    def test_tabu_expected_edges(self, train_data_idx):
        """Tabu edges should not exist in the network"""

        tabu_e = [("d", "a"), ("b", "c")]
        g = from_pandas(train_data_idx, tabu_edges=tabu_e)
        assert [e not in g.edges for e in tabu_e]

    def test_tabu_expected_parent_nodes(self, train_data_idx):
        """Tabu parent nodes should not have any outgoing edges"""

        tabu_p = ["a", "d", "b"]
        g = from_pandas(train_data_idx, tabu_parent_nodes=tabu_p)
        assert [p not in [e[0] for e in g.edges] for p in tabu_p]

    def test_tabu_expected_child_nodes(self, train_data_idx):
        """Tabu child nodes should not have any ingoing edges"""

        tabu_c = ["a", "d", "b"]
        g = from_pandas(train_data_idx, tabu_child_nodes=tabu_c)
        assert [c not in [e[1] for e in g.edges] for c in tabu_c]

    def test_multiple_tabu(self, train_data_idx):
        """Any edge related to tabu edges/parent nodes/child nodes should not exist in the network"""

        tabu_e = [("d", "a"), ("b", "c")]
        tabu_p = ["b"]
        tabu_c = ["a", "d"]
        g = from_pandas(
            train_data_idx,
            tabu_edges=tabu_e,
            tabu_parent_nodes=tabu_p,
            tabu_child_nodes=tabu_c,
        )
        assert [e not in g.edges for e in tabu_e]
        assert [p not in [e[0] for e in g.edges] for p in tabu_p]
        assert [c not in [e[1] for e in g.edges] for c in tabu_c]

    def test_sparsity(self, train_data_idx):
        """Structure learnt from larger lambda should be sparser than smaller lambda"""

        g1 = from_pandas(train_data_idx, lasso_beta=10.0, w_threshold=0.25)
        g2 = from_pandas(train_data_idx, lasso_beta=1e-6, w_threshold=0.25)
        assert len(g1.edges) < len(g2.edges)

    def test_sparsity_against_without_reg(self, train_data_idx):
        """Structure learnt from regularisation should be sparser than the one without"""

        g1 = from_pandas(train_data_idx, lasso_beta=10.0, w_threshold=0.25)
        g2 = from_pandas(train_data_idx, w_threshold=0.25)
        assert len(g1.edges) < len(g2.edges)

    def test_f1_score_fixed(self, train_data_idx, train_model):
        """Structure learnt from regularisation should have very high f1 score relative to the ground truth"""

        g = from_pandas(train_data_idx, lasso_beta=0.01, w_threshold=0.25)

        n_predictions_made = len(g.edges)
        n_correct_predictions = len(set(g.edges).intersection(set(train_model.edges)))
        n_relevant_predictions = len(train_model.edges)

        precision = n_correct_predictions / n_predictions_made
        recall = n_correct_predictions / n_relevant_predictions
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert f1_score > 0.8

    def test_f1score_generated(self, adjacency_mat_num_stability):
        """Structure learnt from regularisation should have very high f1 score relative to the ground truth"""
        df = pd.DataFrame(
            adjacency_mat_num_stability,
            columns=["a", "b", "c", "d", "e"],
            index=["a", "b", "c", "d", "e"],
        )
        train_model = StructureModel(df)
        X = generate_continuous_dataframe(train_model, 50, noise_scale=1, seed=1)
        g = from_pandas(X, lasso_beta=0.1, w_threshold=0.25)
        right_edges = train_model.edges

        n_predictions_made = len(g.edges)
        n_correct_predictions = len(set(g.edges).intersection(set(right_edges)))
        n_relevant_predictions = len(right_edges)

        precision = n_correct_predictions / n_predictions_made
        recall = n_correct_predictions / n_relevant_predictions
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert f1_score > 0.85

    @pytest.mark.parametrize("data", [[np.nan, 0], [np.inf, 0]])
    def test_check_array(self, data):
        """
        Providing a data set including nan or inf should result in a Value Error explaining that data contains nan.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match="Input contains NaN, infinity or a value too large for dtype*",
        ):
            from_pandas(pd.DataFrame(data=data, columns=["a"]))

    def test_f1score_generated_binary(self):
        """Binary strucutre learned should have good f1 score"""
        np.random.seed(10)
        sm = generate_structure(5, 2.0)
        df = generate_binary_dataframe(
            sm, 1000, intercept=False, noise_scale=0.1, seed=10
        )

        dist_type_schema = {i: "bin" for i in range(df.shape[1])}
        sm_fitted = from_pandas(
            df,
            dist_type_schema=dist_type_schema,
            lasso_beta=0.1,
            ridge_beta=0.0,
            w_threshold=0.1,
            use_bias=False,
        )

        right_edges = sm.edges
        n_predictions_made = len(sm_fitted.edges)
        n_correct_predictions = len(set(sm_fitted.edges).intersection(set(right_edges)))
        n_relevant_predictions = len(right_edges)

        precision = n_correct_predictions / n_predictions_made
        recall = n_correct_predictions / n_relevant_predictions
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert f1_score > 0.8

    def test_f1score_generated_poisson(self):
        """Poisson strucutre learned should have good f1 score"""
        np.random.seed(10)
        sm = generate_structure(5, 3.0)
        df = generate_count_dataframe(
            sm, 1000, intercept=False, zero_inflation_factor=0.0, seed=10
        )

        dist_type_schema = {i: "poiss" for i in range(df.shape[1])}
        sm_fitted = from_pandas(
            df,
            dist_type_schema=dist_type_schema,
            lasso_beta=0.1,
            ridge_beta=0.0,
            w_threshold=0.1,
            use_bias=False,
        )

        right_edges = sm.edges
        n_predictions_made = len(sm_fitted.edges)
        n_correct_predictions = len(set(sm_fitted.edges).intersection(set(right_edges)))
        n_relevant_predictions = len(right_edges)

        precision = n_correct_predictions / n_predictions_made
        recall = n_correct_predictions / n_relevant_predictions
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert f1_score > 0.7


class TestFromNumpy:
    """Test behaviour of the from_numpy_lasso method"""

    def test_all_columns_in_structure(self, train_data_idx):
        """Every columns that is in the data should become a node in the learned structure"""

        g = from_numpy(train_data_idx.values)
        assert (len(g.nodes)) == len(train_data_idx.columns)

    def test_isolated_nodes_exist(self, train_data_idx):
        """Isolated nodes should still be in the learned structure"""

        g = from_numpy(train_data_idx.values, w_threshold=1.0)
        assert len(g.nodes) == len(train_data_idx.columns)

    def test_expected_structure_learned(self, train_data_idx, train_model_idx):
        """Given a small data set that can be examined by hand, the structure should be deterministic"""

        g = from_numpy(train_data_idx.values, w_threshold=0.16)
        assert set(g.edges) == set(train_model_idx.edges)

    def test_empty_data_raises_error(self):
        """
        Providing an empty data set should result in a Value Error explaining that data must not be empty.
        This error is useful to catch and handle gracefully, because otherwise the user would experience
        misleading division by zero, or unpacking errors.
        """

        with pytest.raises(ValueError):
            from_numpy(np.empty([0, 5]))

    def test_single_iter_gets_converged_fail_warnings(self, caplog, train_data_idx):
        """
        With a single iteration on this dataset, learn_structure fails to converge and should give warnings.
        """
        with caplog.at_level(logging.WARNING):
            from_numpy(train_data_idx.values, max_iter=1)
        assert "Failed to converge. Consider increasing max_iter." in caplog.text

    def test_certain_relationships_get_near_certain_weight(self):
        """If observations reliably show a==b and !a==!b then the relationship from a->b should be certain"""

        data = pd.DataFrame([[1, 2] for _ in range(10)], columns=["a", "b"])
        g = from_numpy(data.values, w_threshold=0.25)
        assert set(g.edges) == {(0, 1)}
        assert 1.9 <= g.get_edge_data(0, 1)["weight"] <= 2

    def test_inverse_relationships_get_negative_weight(self):
        """If observations indicate a==!b and b==!a then the weight of the relationship from a-> should be negative"""

        data = pd.DataFrame([[1, -2] for _ in range(10)], columns=["a", "b"])
        data.append(pd.DataFrame([[-1, 2] for _ in range(10)], columns=["a", "b"]))
        g = from_numpy(data.values, w_threshold=0.25)
        assert set(g.edges) == {(0, 1)}
        assert -2 <= g.get_edge_data(0, 1)["mean_effect"] <= -1.9

    def test_no_cycles(self, train_data_idx):
        """
        The learned structure should be acyclic
        """

        g = from_numpy(train_data_idx.values, w_threshold=0.25)
        assert nx.algorithms.is_directed_acyclic_graph(g)

    def test_tabu_expected_edges(self, train_data_idx):
        """Tabu edges should not exist in the network"""

        tabu_e = [(3, 0), (1, 2)]
        g = from_numpy(train_data_idx.values, tabu_edges=tabu_e)
        assert [e not in g.edges for e in tabu_e]

    def test_tabu_expected_parent_nodes(self, train_data_idx):
        """Tabu parent nodes should not have any outgoing edges"""

        tabu_p = [0, 3, 1]
        g = from_numpy(train_data_idx.values, tabu_parent_nodes=tabu_p)
        assert [p not in [e[0] for e in g.edges] for p in tabu_p]

    def test_tabu_expected_child_nodes(self, train_data_idx):
        """Tabu child nodes should not have any ingoing edges"""

        tabu_c = [0, 3, 1]
        g = from_numpy(train_data_idx.values, tabu_child_nodes=tabu_c)
        assert [c not in [e[1] for e in g.edges] for c in tabu_c]

    def test_multiple_tabu(self, train_data_idx):
        """Any edge related to tabu edges/parent nodes/child nodes should not exist in the network"""

        tabu_e = [(3, 0), (1, 2)]
        tabu_p = [1]
        tabu_c = [0, 3]
        g = from_numpy(
            train_data_idx.values,
            tabu_edges=tabu_e,
            tabu_parent_nodes=tabu_p,
            tabu_child_nodes=tabu_c,
        )
        assert [e not in g.edges for e in tabu_e]
        assert [p not in [e[0] for e in g.edges] for p in tabu_p]
        assert [c not in [e[1] for e in g.edges] for c in tabu_c]

    def test_sparsity(self, train_data_idx):
        """Structure learnt from larger lambda should be sparser than smaller lambda"""

        g1 = from_numpy(train_data_idx.values, lasso_beta=10.0, w_threshold=0.25)
        g2 = from_numpy(train_data_idx.values, lasso_beta=1e-6, w_threshold=0.25)
        assert len(g1.edges) < len(g2.edges)

    def test_sparsity_against_without_reg(self, train_data_idx):
        """Structure learnt from regularisation should be sparser than the one without"""

        g1 = from_numpy(train_data_idx.values, lasso_beta=10.0, w_threshold=0.25)
        g2 = from_numpy(train_data_idx.values, w_threshold=0.25)
        assert len(g1.edges) < len(g2.edges)

    def test_f1_score_fixed(self, train_data_idx, train_model_idx):
        """Structure learnt from regularisation should have very high f1 score relative to the ground truth"""

        g = from_numpy(train_data_idx.values, lasso_beta=0.01, w_threshold=0.25)

        n_predictions_made = len(g.edges)
        n_correct_predictions = len(
            set(g.edges).intersection(set(train_model_idx.edges))
        )
        n_relevant_predictions = len(train_model_idx.edges)

        precision = n_correct_predictions / n_predictions_made
        recall = n_correct_predictions / n_relevant_predictions
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert f1_score > 0.8

    def test_f1score_generated(self, adjacency_mat_num_stability):
        """Structure learnt from regularisation should have very high f1 score relative to the ground truth"""
        df = pd.DataFrame(
            adjacency_mat_num_stability,
            columns=["a", "b", "c", "d", "e"],
            index=["a", "b", "c", "d", "e"],
        )
        train_model = StructureModel(df.values)
        X = generate_continuous_dataframe(StructureModel(df), 50, noise_scale=1, seed=1)
        g = from_numpy(
            X[["a", "b", "c", "d", "e"]].values, lasso_beta=0.1, w_threshold=0.25
        )
        right_edges = train_model.edges

        n_predictions_made = len(g.edges)
        n_correct_predictions = len(set(g.edges).intersection(set(right_edges)))
        n_relevant_predictions = len(right_edges)
        precision = n_correct_predictions / n_predictions_made
        recall = n_correct_predictions / n_relevant_predictions
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert f1_score > 0.85

    def test_non_negativity_constraint(self, train_data_idx):
        """
        The optimisation in notears lasso involves reshaping the initial similarity matrix
        into two strictly positive matrixes (w+ and w-) and imposing a non negativity constraint
        to the solver. We test here if these two contraints are imposed.

        We check if:
        (1) bounds impose non negativity constraint
        (2) initial guess obeys non negativity constraint
        (3) most importantly: output of sopt obeys the constraint
        """
        # using `wraps` to **spy** on the function
        with patch(
            "causalnex.structure.pytorch.core.sopt.minimize",
            wraps=sopt.minimize,
        ) as mocked:
            from_numpy(train_data_idx.values, lasso_beta=0.1, w_threshold=0.25)
            # We iterate over each time `sopt.minimize` was called
            for called_arguments in list(mocked.call_args_list):
                # These are the arguments with which the `sopt.minimize` was called
                func_ = called_arguments[0][0]  # positional arg
                w_est = called_arguments[0][1]  # positional arg
                keyword_args = called_arguments[1]

                # check 1:
                assert [
                    (len(el) == 2) and (el[0] == 0) for el in keyword_args["bounds"]
                ]
                # check 2:
                assert [el >= 0 for el in w_est]
                # check 3
                sol = sopt.minimize(func_, w_est, **keyword_args)
                assert [el >= 0 for el in sol.x]

    @pytest.mark.parametrize("data", [[np.nan, 0], [np.inf, 0]])
    def test_check_array(self, data):
        """
        Providing a data set including nan or inf should result in a Value Error explaining that data contains nan.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match="Input contains NaN, infinity or a value too large for dtype*",
        ):
            from_numpy(np.array([data]))

    def test_f1score_generated_binary(self):
        """Binary structure learned should have good f1 score"""
        np.random.seed(10)
        sm = generate_structure(5, 2.0)
        df = generate_binary_data(sm, 1000, intercept=False, noise_scale=0.1, seed=10)

        dist_type_schema = {i: "bin" for i in range(df.shape[1])}
        sm_fitted = from_numpy(
            df,
            dist_type_schema=dist_type_schema,
            lasso_beta=0.1,
            ridge_beta=0.0,
            w_threshold=0.1,
            use_bias=False,
        )

        right_edges = sm.edges
        n_predictions_made = len(sm_fitted.edges)
        n_correct_predictions = len(set(sm_fitted.edges).intersection(set(right_edges)))
        n_relevant_predictions = len(right_edges)

        precision = n_correct_predictions / n_predictions_made
        recall = n_correct_predictions / n_relevant_predictions
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert f1_score > 0.8

    def test_f1score_generated_poisson(self):
        """Poisson structure learned should have good f1 score"""
        np.random.seed(10)
        sm = generate_structure(5, 3.0)
        df = generate_count_dataframe(
            sm, 1000, intercept=False, zero_inflation_factor=0.0, seed=10
        )
        df = np.asarray(df)

        dist_type_schema = {i: "poiss" for i in range(df.shape[1])}
        sm_fitted = from_numpy(
            df,
            dist_type_schema=dist_type_schema,
            lasso_beta=0.1,
            ridge_beta=0.0,
            w_threshold=0.1,
            use_bias=False,
        )

        right_edges = sm.edges
        n_predictions_made = len(sm_fitted.edges)
        n_correct_predictions = len(set(sm_fitted.edges).intersection(set(right_edges)))
        n_relevant_predictions = len(right_edges)

        precision = n_correct_predictions / n_predictions_made
        recall = n_correct_predictions / n_relevant_predictions
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert f1_score > 0.7
