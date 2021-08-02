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

import math

import numpy as np
import pandas as pd
import pytest

from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
from causalnex.utils.network_utils import get_markov_blanket


class TestFitNodeStates:
    """Test behaviour of fit node states method"""

    @pytest.mark.parametrize(
        "weighted_edges, data",
        [
            ([("a", "b", 1)], pd.DataFrame([[1, 1]], columns=["a", "b"])),
            (
                [("a", "b", 1)],
                pd.DataFrame([[1, 1, 1, 1]], columns=["a", "b", "c", "d"]),
            ),
            # c and d are isolated nodes in the data
        ],
    )
    def test_all_nodes_included(self, weighted_edges, data):
        """No errors if all the nodes can be found in the columns of training data"""
        cg = StructureModel()
        cg.add_weighted_edges_from(weighted_edges)
        bn = BayesianNetwork(cg).fit_node_states(data)
        assert all(node in data.columns for node in bn.node_states.keys())

    def test_all_states_included(self):
        """All states in a node should be included"""
        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "b", 1)])
        bn = BayesianNetwork(cg).fit_node_states(
            pd.DataFrame([[i, i] for i in range(10)], columns=["a", "b"])
        )
        assert all(v in bn.node_states["a"] for v in range(10))

    def test_fit_with_null_states_raises_error(self):
        """An error should be raised if fit is called with null data"""
        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "b", 1)])
        with pytest.raises(ValueError, match="node '.*' contains None state"):
            BayesianNetwork(cg).fit_node_states(
                pd.DataFrame([[None, 1]], columns=["a", "b"])
            )

    def test_fit_with_missing_feature_in_data(self):
        """An error should be raised if fit is called with missing feature in data"""
        cg = StructureModel()

        cg.add_weighted_edges_from([("a", "e", 1)])
        with pytest.raises(
            KeyError,
            match="The data does not cover all the features found in the Bayesian Network. "
            "Please check the following features: {'e'}",
        ):
            BayesianNetwork(cg).fit_node_states(
                pd.DataFrame([[1, 1, 1, 1]], columns=["a", "b", "c", "d"])
            )


class TestFitCPDSErrors:
    """Test errors for fit CPDs method"""

    def test_invalid_method(self, bn, train_data_discrete):
        """a value error should be raised in an invalid method is provided"""

        with pytest.raises(ValueError, match=r"unrecognised method.*"):
            bn.fit_cpds(train_data_discrete, method="INVALID")

    def test_invalid_prior(self, bn, train_data_discrete):
        """a value error should be raised in an invalid prior is provided"""

        with pytest.raises(ValueError, match=r"unrecognised bayes_prior.*"):
            bn.fit_cpds(
                train_data_discrete, method="BayesianEstimator", bayes_prior="INVALID"
            )


class TestFitCPDsMaximumLikelihoodEstimator:
    """Test behaviour of fit_cpds using MLE"""

    def test_cause_only_node(self, bn, train_data_discrete, train_data_discrete_cpds):
        """Test that probabilities are fit correctly to nodes which are not caused by other nodes"""

        bn.fit_cpds(train_data_discrete)
        cpds = bn.cpds

        assert (
            np.mean(
                np.abs(
                    cpds["d"].values.reshape(2)
                    - train_data_discrete_cpds["d"].reshape(2)
                )
            )
            < 1e-7
        )
        assert (
            np.mean(
                np.abs(
                    cpds["e"].values.reshape(2)
                    - train_data_discrete_cpds["e"].reshape(2)
                )
            )
            < 1e-7
        )

    def test_dependent_node(self, bn, train_data_discrete, train_data_discrete_cpds):
        """Test that probabilities are fit correctly to nodes that are caused by other nodes"""

        bn.fit_cpds(train_data_discrete)
        cpds = bn.cpds

        assert (
            np.mean(
                np.abs(
                    cpds["a"].values.reshape(24)
                    - train_data_discrete_cpds["a"].reshape(24)
                )
            )
            < 1e-7
        )
        assert (
            np.mean(
                np.abs(
                    cpds["b"].values.reshape(12)
                    - train_data_discrete_cpds["b"].reshape(12)
                )
            )
            < 1e-7
        )
        assert (
            np.mean(
                np.abs(
                    cpds["c"].values.reshape(60)
                    - train_data_discrete_cpds["c"].reshape(60)
                )
            )
            < 1e-7
        )

    def test_fit_missing_states(self):
        """test issues/15: should be possible to fit with missing states"""

        sm = StructureModel([("a", "b"), ("c", "b")])
        bn = BayesianNetwork(sm)

        train = pd.DataFrame(
            data=[[0, 0, 1], [1, 0, 1], [1, 1, 1]], columns=["a", "b", "c"]
        )
        test = pd.DataFrame(
            data=[[0, 0, 1], [1, 0, 1], [1, 1, 2]], columns=["a", "b", "c"]
        )
        data = pd.concat([train, test])

        bn.fit_node_states(data)
        bn.fit_cpds(train)

        assert bn.cpds["c"].loc[1][0] == 1
        assert bn.cpds["c"].loc[2][0] == 0


class TestFitBayesianEstimator:
    """Test behaviour of fit_cpds using BE"""

    def test_cause_only_node_bdeu(
        self, bn, train_data_discrete, train_data_discrete_cpds
    ):
        """Test that probabilities are fit correctly to nodes which are not caused by other nodes"""

        bn.fit_cpds(
            train_data_discrete,
            method="BayesianEstimator",
            bayes_prior="BDeu",
            equivalent_sample_size=5,
        )
        cpds = bn.cpds

        assert (
            np.mean(
                np.abs(
                    cpds["d"].values.reshape(2)
                    - train_data_discrete_cpds["d"].reshape(2)
                )
            )
            < 0.02
        )
        assert (
            np.mean(
                np.abs(
                    cpds["e"].values.reshape(2)
                    - train_data_discrete_cpds["e"].reshape(2)
                )
            )
            < 0.02
        )

    def test_cause_only_node_k2(
        self, bn, train_data_discrete, train_data_discrete_cpds
    ):
        """Test that probabilities are fit correctly to nodes which are not caused by other nodes"""

        bn.fit_cpds(train_data_discrete, method="BayesianEstimator", bayes_prior="K2")
        cpds = bn.cpds

        assert (
            np.mean(
                np.abs(
                    cpds["d"].values.reshape(2)
                    - train_data_discrete_cpds["d"].reshape(2)
                )
            )
            < 0.02
        )
        assert (
            np.mean(
                np.abs(
                    cpds["e"].values.reshape(2)
                    - train_data_discrete_cpds["e"].reshape(2)
                )
            )
            < 0.02
        )

    def test_dependent_node_bdeu(
        self, bn, train_data_discrete, train_data_discrete_cpds
    ):
        """Test that probabilities are fit correctly to nodes that are caused by other nodes"""

        bn.fit_cpds(
            train_data_discrete,
            method="BayesianEstimator",
            bayes_prior="BDeu",
            equivalent_sample_size=1,
        )
        cpds = bn.cpds

        assert (
            np.mean(
                np.abs(
                    cpds["a"].values.reshape(24)
                    - train_data_discrete_cpds["a"].reshape(24)
                )
            )
            < 0.02
        )
        assert (
            np.mean(
                np.abs(
                    cpds["b"].values.reshape(12)
                    - train_data_discrete_cpds["b"].reshape(12)
                )
            )
            < 0.02
        )
        assert (
            np.mean(
                np.abs(
                    cpds["c"].values.reshape(60)
                    - train_data_discrete_cpds["c"].reshape(60)
                )
            )
            < 0.02
        )

    def test_dependent_node_k2(
        self, bn, train_data_discrete, train_data_discrete_cpds_k2
    ):
        """Test that probabilities are fit correctly to nodes that are caused by other nodes"""

        bn.fit_cpds(train_data_discrete, method="BayesianEstimator", bayes_prior="K2")
        cpds = bn.cpds

        assert (
            np.mean(
                np.abs(
                    cpds["a"].values.reshape(24)
                    - train_data_discrete_cpds_k2["a"].reshape(24)
                )
            )
            < 1e-7
        )
        assert (
            np.mean(
                np.abs(
                    cpds["b"].values.reshape(12)
                    - train_data_discrete_cpds_k2["b"].reshape(12)
                )
            )
            < 1e-7
        )
        assert (
            np.mean(
                np.abs(
                    cpds["c"].values.reshape(60)
                    - train_data_discrete_cpds_k2["c"].reshape(60)
                )
            )
            < 1e-7
        )

    def test_fit_missing_states(self):
        """test issues/15: should be possible to fit with missing states"""

        sm = StructureModel([("a", "b"), ("c", "b")])
        bn = BayesianNetwork(sm)

        train = pd.DataFrame(
            data=[[0, 0, 1], [1, 0, 1], [1, 1, 1]], columns=["a", "b", "c"]
        )
        test = pd.DataFrame(
            data=[[0, 0, 1], [1, 0, 1], [1, 1, 2]], columns=["a", "b", "c"]
        )
        data = pd.concat([train, test])

        bn.fit_node_states(data)
        bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

        assert bn.cpds["c"].loc[1][0] == 0.8
        assert bn.cpds["c"].loc[2][0] == 0.2


class TestPredictMaximumLikelihoodEstimator:
    """Test behaviour of predict using MLE"""

    def test_predictions_are_based_on_probabilities(
        self, bn, train_data_discrete, test_data_c_discrete
    ):
        """Predictions made using the model should be based on the probabilities that are in the model"""

        bn.fit_cpds(train_data_discrete)
        predictions = bn.predict(test_data_c_discrete, "c")
        assert np.all(
            predictions.values.reshape(len(predictions.values))
            == test_data_c_discrete["c"].values
        )

    def test_prediction_node_suffixed_as_prediction(
        self, bn, train_data_discrete, test_data_c_discrete
    ):
        """The column that contains the values of the predicted node should be named node_prediction"""

        bn.fit_cpds(train_data_discrete)
        predictions = bn.predict(test_data_c_discrete, "c")
        assert "c_prediction" in predictions.columns

    def test_only_predicted_column_returned(
        self, bn, train_data_discrete, test_data_c_discrete
    ):
        """The returned df should not contain any of the input data columns"""

        bn.fit_cpds(train_data_discrete)
        predictions = bn.predict(test_data_c_discrete, "c")
        assert len(predictions.columns) == 1

    def test_predictions_are_not_appended_to_input_df(
        self, bn, train_data_discrete, test_data_c_discrete
    ):
        """The predictions should not be appended to the input df"""

        expected_cols = test_data_c_discrete.columns
        bn.fit_cpds(train_data_discrete)
        bn.predict(test_data_c_discrete, "c")
        assert np.array_equal(test_data_c_discrete.columns, expected_cols)

    def test_missing_parent(self, bn, train_data_discrete, test_data_c_discrete):
        """Predictions made when parents are missing should still be reasonably accurate"""

        bn.fit_cpds(train_data_discrete)
        predictions = bn.predict(test_data_c_discrete[["a", "b", "c", "d"]], "c")

        n = len(test_data_c_discrete)

        accuracy = (
            1
            - np.count_nonzero(
                predictions.values.reshape(len(predictions.values))
                - test_data_c_discrete["c"].values
            )
            / n
        )

        assert accuracy > 0.9

    def test_missing_non_parent(self, bn, train_data_discrete, test_data_c_discrete):
        """It should be possible to make predictions with non-parent nodes missing"""

        bn.fit_cpds(train_data_discrete)
        predictions = bn.predict(test_data_c_discrete[["b", "c", "d", "e"]], "c")
        assert np.all(
            predictions.values.reshape(len(predictions.values))
            == test_data_c_discrete["c"].values
        )


class TestPredictBayesianEstimator:
    """Test behaviour of predict using BE"""

    def test_predictions_are_based_on_probabilities_dbeu(
        self, bn, train_data_discrete, test_data_c_discrete
    ):
        """Predictions made using the model should be based on the probabilities that are in the model"""

        bn.fit_cpds(
            train_data_discrete,
            method="BayesianEstimator",
            bayes_prior="BDeu",
            equivalent_sample_size=5,
        )
        predictions = bn.predict(test_data_c_discrete, "c")
        assert np.all(
            predictions.values.reshape(len(predictions.values))
            == test_data_c_discrete["c"].values
        )

    def test_predictions_are_based_on_probabilities_k2(
        self, bn, train_data_discrete, test_data_c_discrete
    ):
        """Predictions made using the model should be based on the probabilities that are in the model"""

        bn.fit_cpds(
            train_data_discrete,
            method="BayesianEstimator",
            bayes_prior="K2",
            equivalent_sample_size=5,
        )
        predictions = bn.predict(test_data_c_discrete, "c")
        assert np.all(
            predictions.values.reshape(len(predictions.values))
            == test_data_c_discrete["c"].values
        )


class TestPredictProbabilityMaximumLikelihoodEstimator:
    """Test behaviour of predict_probability using MLE"""

    def test_expected_probabilities_are_predicted(
        self, bn, train_data_discrete, test_data_c_discrete, test_data_c_likelihood
    ):
        """Probabilities should return exactly correct on a hand computable scenario"""
        bn.fit_cpds(train_data_discrete)
        probability = bn.predict_probability(test_data_c_discrete, "c")

        assert all(
            np.isclose(
                probability.values.flatten(), test_data_c_likelihood.values.flatten()
            )
        )

    def test_missing_parent(
        self, bn, train_data_discrete, test_data_c_discrete, test_data_c_likelihood
    ):
        """Probabilities made when parents are missing should still be reasonably accurate"""

        bn.fit_cpds(train_data_discrete)
        probability = bn.predict_probability(
            test_data_c_discrete[["a", "b", "c", "d"]], "c"
        )

        n = len(probability.values.flatten())

        accuracy = (
            np.count_nonzero(
                [
                    1 if math.isclose(a, b, abs_tol=0.15) else 0
                    for a, b in zip(
                        probability.values.flatten(),
                        test_data_c_likelihood.values.flatten(),
                    )
                ]
            )
            / n
        )

        assert accuracy > 0.8

    def test_missing_non_parent(
        self, bn, train_data_discrete, test_data_c_discrete, test_data_c_likelihood
    ):
        """It should be possible to make predictions with non-parent nodes missing"""

        bn.fit_cpds(train_data_discrete)
        probability = bn.predict_probability(
            test_data_c_discrete[["b", "c", "d", "e"]], "c"
        )
        assert all(
            np.isclose(
                probability.values.flatten(), test_data_c_likelihood.values.flatten()
            )
        )


class TestPredictProbabilityBayesianEstimator:
    """Test behaviour of predict_probability using BayesianEstimator"""

    def test_expected_probabilities_are_predicted(
        self, bn, train_data_discrete, test_data_c_discrete, test_data_c_likelihood
    ):
        """Probabilities should return exactly correct on a hand computable scenario"""

        bn.fit_cpds(
            train_data_discrete,
            method="BayesianEstimator",
            bayes_prior="BDeu",
            equivalent_sample_size=1,
        )
        probability = bn.predict_probability(test_data_c_discrete, "c")
        assert all(
            np.isclose(
                probability.values.flatten(),
                test_data_c_likelihood.values.flatten(),
                atol=0.1,
            )
        )


class TestFitNodesStatesAndCPDs:
    """Test behaviour of helper function"""

    def test_behaves_same_as_seperate_calls(self, train_data_idx, train_data_discrete):
        bn1 = BayesianNetwork(from_pandas(train_data_idx, w_threshold=0.3))
        bn2 = BayesianNetwork(from_pandas(train_data_idx, w_threshold=0.3))

        bn1.fit_node_states(train_data_discrete).fit_cpds(train_data_discrete)
        bn2.fit_node_states_and_cpds(train_data_discrete)

        assert bn1.edges == bn2.edges
        assert bn1.node_states == bn2.node_states

        cpds1 = bn1.cpds
        cpds2 = bn2.cpds

        assert cpds1.keys() == cpds2.keys()

        for k, df in cpds1.items():
            assert df.equals(cpds2[k])


class TestCPDsProperty:
    """Test behaviour of the CPDs property"""

    def test_row_index_of_state_values(self, bn):
        """CPDs should have row index set to values of all possible states of the node"""

        assert bn.cpds["a"].index.tolist() == sorted(list(bn.node_states["a"]))

    def test_col_index_of_parent_state_combinations(self, bn):
        """CPDs should have a column multi-index of parent state permutations"""

        assert bn.cpds["a"].columns.names == ["b", "d"]


class TestInit:
    """Test behaviour when constructing a BayesianNetwork"""

    def test_cycles_in_structure(self):
        """An error should be raised if cycles are present"""

        with pytest.raises(
            ValueError,
            match=r"The given structure is not acyclic\. "
            r"Please review the following cycle\.*",
        ):
            BayesianNetwork(StructureModel([(0, 1), (1, 2), (2, 0)]))

    @pytest.mark.parametrize(
        "test_input,n_components",
        [([(0, 1), (1, 2), (3, 4), (4, 6)], 2), ([(0, 1), (1, 2), (3, 4), (5, 6)], 3)],
    )
    def test_disconnected_components(self, test_input, n_components):
        """An error should be raised if there is more than one graph component"""

        with pytest.raises(
            ValueError,
            match=r"The given structure has "
            + str(n_components)
            + r" separated graph components\. "
            r"Please make sure it has only one\.",
        ):
            BayesianNetwork(StructureModel(test_input))


class TestStructure:
    """Test behaviour of the property structure"""

    def test_get_structure(self):
        """The structure retrieved should be the same"""

        sm = StructureModel()

        sm.add_weighted_edges_from([(1, 2, 2.0)], origin="unknown")
        sm.add_weighted_edges_from([(1, 3, 1.0)], origin="learned")
        sm.add_weighted_edges_from([(3, 5, 0.7)], origin="expert")

        bn = BayesianNetwork(sm)

        sm_from_bn = bn.structure

        assert set(sm.edges.data("origin")) == set(sm_from_bn.edges.data("origin"))
        assert set(sm.edges.data("weight")) == set(sm_from_bn.edges.data("weight"))

        assert set(sm.nodes) == set(sm_from_bn.nodes)

    def test_set_structure(self):
        """An error should be raised if setting the structure"""

        sm = StructureModel()
        sm.add_weighted_edges_from([(1, 2, 2.0)], origin="unknown")
        sm.add_weighted_edges_from([(1, 3, 1.0)], origin="learned")
        sm.add_weighted_edges_from([(3, 5, 0.7)], origin="expert")

        bn = BayesianNetwork(sm)

        new_sm = StructureModel()
        sm.add_weighted_edges_from([(2, 5, 3.0)], origin="unknown")
        sm.add_weighted_edges_from([(2, 3, 2.0)], origin="learned")
        sm.add_weighted_edges_from([(3, 4, 1.7)], origin="expert")

        with pytest.raises(AttributeError, match=r"can't set attribute"):
            bn.structure = new_sm


class TestMarkovBlanket:
    """Test behavior of Markov Blanket """

    def test_elements(self, bn_train_model):
        """ check if all elements are included"""
        blanket = get_markov_blanket(bn_train_model, "a")
        parents_of_node = {"b", "d"}
        children_of_node = {"f"}
        parents_of_children = {"e"}

        assert parents_of_node <= set(blanket.nodes)
        assert children_of_node <= set(blanket.nodes)
        assert parents_of_children <= set(blanket.nodes)

    def test_connection(self, bn_train_model):
        """ Check if edges are correct """
        blanket = get_markov_blanket(bn_train_model, "a")
        assert blanket.structure.has_edge("b", "a")
        assert blanket.structure.has_edge("d", "a")
        assert blanket.structure.has_edge("a", "f")
        assert blanket.structure.has_edge("e", "f")
        assert blanket.structure.has_edge("e", "b")

    def test_invalid_node(self, bn_train_model):
        with pytest.raises(
            KeyError,
            match="is not found in the network",
        ):
            get_markov_blanket(bn_train_model, "invalid")
