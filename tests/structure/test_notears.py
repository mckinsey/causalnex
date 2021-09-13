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
import pandas as pd
import pytest
import scipy.optimize as sopt
from mock import patch

from causalnex.structure import StructureModel
from causalnex.structure.data_generators import generate_continuous_dataframe
from causalnex.structure.notears import (
    from_numpy,
    from_numpy_lasso,
    from_pandas,
    from_pandas_lasso,
)


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

        g = from_pandas(train_data_idx, w_threshold=0.3)
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

        with pytest.raises(ValueError, match=r"All columns must have numeric data.*"):
            from_pandas(pd.DataFrame(data=["x"], columns=["a"]))

    def test_array_with_nan_raises_error(self):
        """
        Providing a data set including nan should result in a Value Error explaining that data contains nan.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match=r"Input contains NaN, infinity or a value too large for *",
        ):
            from_pandas(pd.DataFrame(data=[np.nan, 0], columns=["a"]))

    def test_array_with_inf_raises_error(self):
        """
        Providing a data set including infinite values should result in a Value Error explaining that data
        contains infinite values.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match=r"Input contains NaN, infinity or a value too large for *",
        ):
            from_pandas(pd.DataFrame(data=[np.inf, 0], columns=["a"]))

    def test_single_iter_gets_converged_fail_warnings(self, train_data_idx):
        """
        With a single iteration on this dataset, learn_structure fails to converge and should give warnings.
        """

        with pytest.warns(
            UserWarning, match="Failed to converge. Consider increasing max_iter."
        ):
            from_pandas(train_data_idx, max_iter=1)

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
            weight < 0 for u, v, weight in g.edges(data="weight") if u == 0 and v == 1
        )

    def test_no_cycles(self, train_data_idx):
        """
        The learned structure should be acyclic
        """

        g = from_pandas(train_data_idx, w_threshold=0.3)
        assert nx.algorithms.is_directed_acyclic_graph(g)

    def test_tabu_edges_on_non_existing_edges_do_nothing(self, train_data_idx):
        """If tabu edges do not exist in the original unconstrained network then nothing changes"""

        g1 = from_pandas(train_data_idx, w_threshold=0.3)
        g2 = from_pandas(
            train_data_idx, w_threshold=0.3, tabu_edges=[("a", "d"), ("e", "a")]
        )
        assert set(g1.edges) == set(g2.edges)

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


class TestFromPandasLasso:
    """Test behaviour of the from_pandas_lasso method"""

    def test_all_columns_in_structure(self, train_data_idx):
        """Every columns that is in the data should become a node in the learned structure"""

        g = from_pandas_lasso(train_data_idx, 0.1)
        assert len(g.nodes) == len(train_data_idx.columns)

    def test_isolated_nodes_exist(self, train_data_idx):
        """Isolated nodes should still be in the learned structure"""

        g = from_pandas_lasso(train_data_idx, 0.1, w_threshold=1.0)
        assert len(g.nodes) == len(train_data_idx.columns)

    def test_expected_structure_learned(self, train_data_idx, train_model):
        """Given an extremely small alpha and small data set that can be examined by hand,
        the structure should be deterministic"""

        g = from_pandas_lasso(train_data_idx, 1e-8, w_threshold=0.3)
        assert set(g.edges) == set(train_model.edges)

    def test_empty_data_raises_error(self):
        """
        Providing an empty data set should result in a Value Error explaining that data must not be empty.
        This error is useful to catch and handle gracefully, because otherwise the user would experience
        misleading division by zero, or unpacking errors.
        """

        with pytest.raises(ValueError):
            from_pandas_lasso(pd.DataFrame(data=[], columns=["a"]), 0.1)

    def test_non_numeric_data_raises_error(self):
        """Only numeric data frames should be supported"""

        with pytest.raises(ValueError, match=r"All columns must have numeric data.*"):
            from_pandas_lasso(pd.DataFrame(data=["x"], columns=["a"]), 0.1)

    def test_array_with_nan_raises_error(self):
        """
        Providing a data set including nan should result in a Value Error explaining that data contains nan.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match=r"Input contains NaN, infinity or a value too large for *",
        ):
            from_pandas_lasso(pd.DataFrame(data=[np.nan, 0], columns=["a"]), 0.1)

    def test_array_with_inf_raises_error(self):
        """
        Providing a data set including infinite values should result in a Value Error explaining that data
        contains infinite values.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match=r"Input contains NaN, infinity or a value too large for *",
        ):
            from_pandas_lasso(pd.DataFrame(data=[np.inf, 0], columns=["a"]), 0.1)

    def test_single_iter_gets_converged_fail_warnings(self, train_data_idx):
        """
        With a single iteration on this dataset, learn_structure fails to converge and should give warnings.
        """

        with pytest.warns(
            UserWarning, match="Failed to converge. Consider increasing max_iter."
        ):
            from_pandas_lasso(train_data_idx, 0.1, max_iter=1)

    def test_certain_relationships_get_near_certain_weight(self):
        """If observations reliably show a==b and !a==!b then the relationship from a->b should be certain"""

        data = pd.DataFrame([[0, 1] for _ in range(10)], columns=["a", "b"])
        g = from_pandas_lasso(data, 0.1)
        assert all(
            0.99 <= weight <= 1
            for u, v, weight in g.edges(data="weight")
            if u == 0 and v == 1
        )

    def test_inverse_relationships_get_negative_weight(self):
        """If observations indicate a==!b and b==!a then the weight of the relationship from a-> should be negative"""

        data = pd.DataFrame([[0, 1] for _ in range(10)], columns=["a", "b"])
        data.append(pd.DataFrame([[1, 0] for _ in range(10)], columns=["a", "b"]))
        g = from_pandas_lasso(data, 0.1)
        assert all(
            weight < 0 for u, v, weight in g.edges(data="weight") if u == 0 and v == 1
        )

    def test_no_cycles(self, train_data_idx):
        """
        The learned structure should be acyclic
        """

        g = from_pandas_lasso(train_data_idx, 0.1, w_threshold=0.3)
        assert nx.algorithms.is_directed_acyclic_graph(g)

    def test_tabu_expected_edges(self, train_data_idx):
        """Tabu edges should not exist in the network"""

        tabu_e = [("d", "a"), ("b", "c")]
        g = from_pandas_lasso(train_data_idx, 0.1, tabu_edges=tabu_e)
        assert [e not in g.edges for e in tabu_e]

    def test_tabu_expected_parent_nodes(self, train_data_idx):
        """Tabu parent nodes should not have any outgoing edges"""

        tabu_p = ["a", "d", "b"]
        g = from_pandas_lasso(train_data_idx, 0.1, tabu_parent_nodes=tabu_p)
        assert [p not in [e[0] for e in g.edges] for p in tabu_p]

    def test_tabu_expected_child_nodes(self, train_data_idx):
        """Tabu child nodes should not have any ingoing edges"""

        tabu_c = ["a", "d", "b"]
        g = from_pandas_lasso(train_data_idx, 0.1, tabu_child_nodes=tabu_c)
        assert [c not in [e[1] for e in g.edges] for c in tabu_c]

    def test_multiple_tabu(self, train_data_idx):
        """Any edge related to tabu edges/parent nodes/child nodes should not exist in the network"""

        tabu_e = [("d", "a"), ("b", "c")]
        tabu_p = ["b"]
        tabu_c = ["a", "d"]
        g = from_pandas_lasso(
            train_data_idx,
            0.1,
            tabu_edges=tabu_e,
            tabu_parent_nodes=tabu_p,
            tabu_child_nodes=tabu_c,
        )
        assert [e not in g.edges for e in tabu_e]
        assert [p not in [e[0] for e in g.edges] for p in tabu_p]
        assert [c not in [e[1] for e in g.edges] for c in tabu_c]

    def test_sparsity(self, train_data_idx):
        """Structure learnt from larger lambda should be sparser than smaller lambda"""

        g1 = from_pandas_lasso(train_data_idx, 0.1, w_threshold=0.3)
        g2 = from_pandas_lasso(train_data_idx, 1e-6, w_threshold=0.3)
        assert len(g1.edges) < len(g2.edges)

    def test_sparsity_against_without_reg(self, train_data_idx):
        """Structure learnt from regularisation should be sparser than the one without"""

        g1 = from_pandas_lasso(train_data_idx, 0.1, w_threshold=0.3)
        g2 = from_pandas(train_data_idx, w_threshold=0.3)
        assert len(g1.edges) < len(g2.edges)

    def test_f1_score_fixed(self, train_data_idx, train_model):
        """Structure learnt from regularisation should have very high f1 score relative to the ground truth"""
        g = from_pandas_lasso(train_data_idx, 0.1, w_threshold=0.3)

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
            10 * adjacency_mat_num_stability,
            columns=["a", "b", "c", "d", "e"],
            index=["a", "b", "c", "d", "e"],
        )
        train_model = StructureModel(df)
        X = generate_continuous_dataframe(train_model, 50, noise_scale=1, seed=20)
        g = from_pandas_lasso(X, 0.1, w_threshold=1)
        right_edges = train_model.edges

        n_predictions_made = len(g.edges)
        n_correct_predictions = len(set(g.edges).intersection(set(right_edges)))
        n_relevant_predictions = len(right_edges)

        precision = n_correct_predictions / n_predictions_made
        recall = n_correct_predictions / n_relevant_predictions
        f1_score = 2 * (precision * recall) / (precision + recall)

        assert f1_score > 0.85


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

        g = from_numpy(train_data_idx.values, w_threshold=0.3)
        assert set(g.edges) == set(train_model_idx.edges)

    def test_empty_data_raises_error(self):
        """
        Providing an empty data set should result in a Value Error explaining that data must not be empty.
        This error is useful to catch and handle gracefully, because otherwise the user would experience
        misleading division by zero, or unpacking errors.
        """

        with pytest.raises(ValueError):
            from_numpy(np.empty([0, 5]))

    def test_array_with_nan_raises_error(self):
        """
        Providing a data set including nan should result in a Value Error explaining that data contains nan.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match=r"Input contains NaN, infinity or a value too large for *",
        ):
            from_numpy(np.array([[0, np.nan]]))

    def test_array_with_inf_raises_error(self):
        """
        Providing a data set including infinite values should result in a Value Error explaining that data
        contains infinite values.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match=r"Input contains NaN, infinity or a value too large for *",
        ):
            from_numpy(np.array([[0, np.inf]]))

    def test_single_iter_gets_converged_fail_warnings(self, train_data_idx):
        """
        With a single iteration on this dataset, learn_structure fails to converge and should give warnings.
        """

        with pytest.warns(
            UserWarning, match="Failed to converge. Consider increasing max_iter."
        ):
            from_numpy(train_data_idx.values, max_iter=1)

    def test_certain_relationships_get_near_certain_weight(self):
        """If observations reliably show a==b and !a==!b then the relationship from a->b should be certain"""

        data = pd.DataFrame([[0, 1] for _ in range(10)], columns=["a", "b"])
        g = from_numpy(data.values)
        assert all(
            0.99 <= weight <= 1
            for u, v, weight in g.edges(data="weight")
            if u == 0 and v == 1
        )

    def test_inverse_relationships_get_negative_weight(self):
        """If observations indicate a==!b and b==!a then the weight of the relationship from a-> should be negative"""

        data = pd.DataFrame([[0, 1] for _ in range(10)], columns=["a", "b"])
        data.append(pd.DataFrame([[1, 0] for _ in range(10)], columns=["a", "b"]))
        g = from_numpy(data.values)
        assert all(
            weight < 0 for u, v, weight in g.edges(data="weight") if u == 0 and v == 1
        )

    def test_no_cycles(self, train_data_idx):
        """
        The learned structure should be acyclic
        """

        g = from_numpy(train_data_idx.values, w_threshold=0.3)
        assert nx.algorithms.is_directed_acyclic_graph(g)

    def test_tabu_edges_on_non_existing_edges_do_nothing(self, train_data_idx):
        """If tabu edges do not exist in the original unconstrained network then nothing changes"""

        g1 = from_numpy(train_data_idx.values, w_threshold=0.3)
        g2 = from_numpy(
            train_data_idx.values, w_threshold=0.3, tabu_edges=[(0, 3), (4, 0), (1, 6)]
        )
        assert set(g1.edges) == set(g2.edges)

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


class TestFromNumpyLasso:
    """Test behaviour of the from_numpy_lasso method"""

    def test_all_columns_in_structure(self, train_data_idx):
        """Every columns that is in the data should become a node in the learned structure"""

        g = from_numpy_lasso(train_data_idx.values, 0.1)
        assert len(g.nodes) == len(train_data_idx.columns)

    def test_isolated_nodes_exist(self, train_data_idx):
        """Isolated nodes should still be in the learned structure"""

        g = from_numpy_lasso(train_data_idx.values, 0.1, w_threshold=1.0)
        assert len(g.nodes) == len(train_data_idx.columns)

    def test_expected_structure_learned(self, train_data_idx, train_model_idx):
        """Given an extremely small lambda_lasso and small data set that can be examined by hand,
        the structure should be deterministic"""

        g = from_numpy_lasso(train_data_idx.values, 1e-8, w_threshold=0.3)
        assert set(g.edges) == set(train_model_idx.edges)

    def test_empty_data_raises_error(self):
        """
        Providing an empty data set should result in a Value Error explaining that data must not be empty.
        This error is useful to catch and handle gracefully, because otherwise the user would experience
        misleading division by zero, or unpacking errors.
        """

        with pytest.raises(ValueError):
            from_numpy_lasso(np.empty([0, 5]), 0.1)

    def test_array_with_nan_raises_error(self):
        """
        Providing a data set including nan should result in a Value Error explaining that data contains nan.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match=r"Input contains NaN, infinity or a value too large for *",
        ):
            from_numpy_lasso(np.array([[3, np.nan]]), 0.1)

    def test_array_with_inf_raises_error(self):
        """
        Providing a data set including infinite values should result in a Value Error explaining that data
        contains infinite values.
        This error is useful to catch and handle gracefully, because otherwise the user would have empty structures.
        """
        with pytest.raises(
            ValueError,
            match=r"Input contains NaN, infinity or a value too large for *",
        ):
            from_numpy_lasso(np.array([[3, np.inf]]), 0.1)

    def test_single_iter_gets_converged_fail_warnings(self, train_data_idx):
        """
        With a single iteration on this dataset, learn_structure fails to converge and should give warnings.
        """

        with pytest.warns(
            UserWarning, match="Failed to converge. Consider increasing max_iter."
        ):
            from_numpy_lasso(train_data_idx.values, 0.1, max_iter=1)

    def test_certain_relationships_get_near_certain_weight(self):
        """If observations reliably show a==b and !a==!b then the relationship from a->b should be certain"""

        data = pd.DataFrame([[0, 1] for _ in range(10)], columns=["a", "b"])
        g = from_numpy_lasso(data.values, 0.1)
        assert all(
            0.99 <= weight <= 1
            for u, v, weight in g.edges(data="weight")
            if u == 0 and v == 1
        )

    def test_inverse_relationships_get_negative_weight(self):
        """If observations indicate a==!b and b==!a then the weight of the relationship from a-> should be negative"""

        data = pd.DataFrame([[0, 1] for _ in range(10)], columns=["a", "b"])
        data.append(pd.DataFrame([[1, 0] for _ in range(10)], columns=["a", "b"]))
        g = from_numpy_lasso(data.values, 0.1)
        assert all(
            weight < 0 for u, v, weight in g.edges(data="weight") if u == 0 and v == 1
        )

    def test_no_cycles(self, train_data_idx):
        """
        The learned structure should be acyclic
        """

        g = from_numpy_lasso(train_data_idx.values, 0.1, w_threshold=0.3)
        assert nx.algorithms.is_directed_acyclic_graph(g)

    def test_tabu_expected_edges(self, train_data_idx):
        """Tabu edges should not exist in the network"""

        tabu_e = [(0, 1), (2, 3)]
        g = from_numpy_lasso(train_data_idx.values, 0.1, tabu_edges=tabu_e)
        assert [e not in g.edges for e in tabu_e]

    def test_tabu_expected_parent_nodes(self, train_data_idx):
        """Tabu parent nodes should not have any outgoing edges"""

        tabu_p = [0, 1, 2]
        g = from_numpy_lasso(train_data_idx.values, 0.1, tabu_parent_nodes=tabu_p)
        assert [p not in [e[0] for e in g.edges] for p in tabu_p]

    def test_tabu_expected_child_nodes(self, train_data_idx):
        """Tabu child nodes should not have any ingoing edges"""

        tabu_c = [0, 1, 2]
        g = from_numpy_lasso(train_data_idx.values, 0.1, tabu_child_nodes=tabu_c)
        assert [c not in [e[1] for e in g.edges] for c in tabu_c]

    def test_multiple_tabu(self, train_data_idx):
        """Any edge related to tabu edges/parent nodes/child nodes should not exist in the network"""

        tabu_e = [(3, 0), (1, 2)]
        tabu_p = [1]
        tabu_c = [0, 3]
        g = from_numpy_lasso(
            train_data_idx.values,
            0.1,
            tabu_edges=tabu_e,
            tabu_parent_nodes=tabu_p,
            tabu_child_nodes=tabu_c,
        )
        assert [e not in g.edges for e in tabu_e]
        assert [p not in [e[0] for e in g.edges] for p in tabu_p]
        assert [c not in [e[1] for e in g.edges] for c in tabu_c]

    def test_sparsity(self, train_data_idx):
        """Structure learnt from larger lambda should be sparser than smaller lambda"""

        g1 = from_numpy_lasso(train_data_idx.values, 0.1, w_threshold=0.3)
        g2 = from_numpy_lasso(train_data_idx.values, 1e-6, w_threshold=0.3)
        assert len(g1.edges) < len(g2.edges)

    def test_sparsity_against_without_reg(self, train_data_idx):
        """Structure learnt from regularisation should be sparser than the one without"""

        g1 = from_numpy_lasso(train_data_idx.values, 0.1, w_threshold=0.3)
        g2 = from_numpy(train_data_idx.values, w_threshold=0.3)
        assert len(g1.edges) < len(g2.edges)

    def test_f1_score_fixed(self, train_data_idx, train_model_idx):
        """Structure learnt from regularisation should have very high f1 score relative to the ground truth"""
        g = from_numpy_lasso(train_data_idx.values, 0.1, w_threshold=0.3)

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
            10 * adjacency_mat_num_stability,
            columns=["a", "b", "c", "d", "e"],
            index=["a", "b", "c", "d", "e"],
        )
        train_model = StructureModel(df.values)
        X = generate_continuous_dataframe(
            StructureModel(df), 100, noise_scale=1, seed=20
        )
        g = from_numpy_lasso(X[["a", "b", "c", "d", "e"]].values, 0.1, w_threshold=0.1)
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
            "causalnex.structure.notears.sopt.minimize", wraps=sopt.minimize
        ) as mocked:
            from_numpy_lasso(train_data_idx.values, 0.1, w_threshold=0.3)
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
