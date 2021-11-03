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

from causalnex.evaluation import classification_report, roc_auc
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas

from .estimator.test_em import naive_bayes_plus_parents


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

    def test_behaves_same_as_separate_calls(self, train_data_idx, train_data_discrete):
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


class TestLatentVariable:
    @staticmethod
    def mean_absolute_error(cpds_a, cpds_b):
        """Compute the absolute error among each single parameter and average them out"""

        mae = 0
        n_param = 0

        for node in cpds_a.keys():
            err = np.abs(cpds_a[node] - cpds_b[node]).values
            mae += np.sum(err)
            n_param += err.shape[0] * err.shape[1]

        return mae / n_param

    def test_em_algorithm(self):  # pylint: disable=too-many-locals
        """
        Test if `BayesianNetwork` works with EM algorithm.
        We use a naive bayes + parents + an extra node not related to the latent variable.
        """

        # p0   p1  p2
        #   \  |  /
        #      z
        #   /  |  \
        # c0  c1  c2
        # |
        # cc0
        np.random.seed(22)

        data, sm, _, true_lv_values = naive_bayes_plus_parents(
            percentage_not_missing=0.1,
            samples=1000,
            p_z=0.7,
            p_c=0.7,
        )
        data["cc_0"] = np.where(
            np.random.random(len(data)) < 0.5, data["c_0"], (data["c_0"] + 1) % 3
        )
        data.drop(columns=["z"], inplace=True)

        complete_data = data.copy(deep=True)
        complete_data["z"] = true_lv_values

        # Baseline model: the structure of the figure trained with complete data. We try to reproduce it
        complete_bn = BayesianNetwork(
            StructureModel(list(sm.edges) + [("c_0", "cc_0")])
        )
        complete_bn.fit_node_states_and_cpds(complete_data)

        # BN without latent variable: All `p`s are connected to all `c`s + `c0` ->`cc0`
        sm_no_lv = StructureModel(
            [(f"p_{p}", f"c_{c}") for p in range(3) for c in range(3)]
            + [("c_0", "cc_0")]
        )
        bn = BayesianNetwork(sm_no_lv)
        bn.fit_node_states(data)
        bn.fit_cpds(data)

        # TEST 1: cc_0 does not depend on the latent variable so:
        assert np.all(bn.cpds["cc_0"] == complete_bn.cpds["cc_0"])

        # BN with latent variable
        # When we add the latent variable, we add the edges in the image above
        # and remove the connection among `p`s and `c`s
        edges_to_add = list(sm.edges)
        edges_to_remove = [(f"p_{p}", f"c_{c}") for p in range(3) for c in range(3)]
        bn.add_node("z", edges_to_add, edges_to_remove)
        bn.fit_latent_cpds("z", [0, 1, 2], data, stopping_delta=0.001)

        # TEST 2: cc_0 CPD should remain untouched by the EM algorithm
        assert np.all(bn.cpds["cc_0"] == complete_bn.cpds["cc_0"])

        # TEST 3: We should recover the correct CPDs quite accurately
        assert bn.cpds.keys() == complete_bn.cpds.keys()
        assert self.mean_absolute_error(bn.cpds, complete_bn.cpds) < 0.01

        # TEST 4: Inference over recovered CPDs should be also accurate
        eng = InferenceEngine(bn)
        query = eng.query()
        n_rows = complete_data.shape[0]

        for node in query:
            assert (
                np.abs(query[node][0] - sum(complete_data[node] == 0) / n_rows) < 1e-2
            )
            assert (
                np.abs(query[node][1] - sum(complete_data[node] == 1) / n_rows) < 1e-2
            )

        # TEST 5: Inference using predict and predict_probability functions
        report = classification_report(bn, complete_data, "z")
        _, auc = roc_auc(bn, complete_data, "z")
        complete_report = classification_report(complete_bn, complete_data, "z")
        _, complete_auc = roc_auc(complete_bn, complete_data, "z")

        for category, metrics in report.items():
            if isinstance(metrics, dict):
                for key, val in metrics.items():
                    assert np.abs(val - complete_report[category][key]) < 1e-2
            else:
                assert np.abs(metrics - complete_report[category]) < 1e-2

        assert np.abs(auc - complete_auc) < 1e-2


class TestAddNode:
    def test_add_node_not_in_edges_to_add(self):
        """An error should be raised if the latent variable is NOT part of the edges to add"""

        with pytest.raises(
            ValueError,
            match="Should only add edges containing node 'd'",
        ):
            _, sm, _, _ = naive_bayes_plus_parents()
            sm = StructureModel(list(sm.edges))
            bn = BayesianNetwork(sm)
            bn.add_node("d", [("a", "z"), ("b", "z")], [])

    def test_add_node_in_edges_to_remove(self):
        """An error should be raised if the latent variable is part of the edges to remove"""

        with pytest.raises(
            ValueError,
            match="Should only remove edges NOT containing node 'd'",
        ):
            _, sm, _, _ = naive_bayes_plus_parents()
            sm = StructureModel(list(sm.edges))
            bn = BayesianNetwork(sm)
            bn.add_node("d", [], [("a", "d"), ("b", "d")])


class TestFitLatentCPDs:
    @pytest.mark.parametrize("lv_name", [None, [], set(), {}, tuple(), 123, {}])
    def test_fit_invalid_lv_name(self, lv_name):
        """An error should be raised if the latent variable is of an invalid type"""

        with pytest.raises(
            ValueError,
            match=r"Invalid latent variable name *",
        ):
            df, sm, _, _ = naive_bayes_plus_parents()
            sm = StructureModel(list(sm.edges))
            bn = BayesianNetwork(sm)
            bn.fit_latent_cpds(lv_name, [0, 1, 2], df)

    def test_fit_lv_not_added(self):
        """An error should be raised if the latent variable is not added to the network yet"""

        with pytest.raises(
            ValueError,
            match=r"Latent variable 'd' not added to the network",
        ):
            df, sm, _, _ = naive_bayes_plus_parents()
            sm = StructureModel(list(sm.edges))
            bn = BayesianNetwork(sm)
            bn.fit_latent_cpds("d", [0, 1, 2], df)

    @pytest.mark.parametrize("lv_states", [None, [], set(), {}])
    def test_fit_invalid_lv_states(self, lv_states):
        """An error should be raised if the latent variable has invalid states"""

        with pytest.raises(
            ValueError,
            match="Latent variable 'd' contains no states",
        ):
            df, sm, _, _ = naive_bayes_plus_parents()
            sm = StructureModel(list(sm.edges))
            bn = BayesianNetwork(sm)
            bn.add_node("d", [("z", "d")], [])
            bn.fit_latent_cpds("d", lv_states, df)


class TestSetCPD:
    """Test behaviour of adding a self-defined cpd"""

    def test_set_cpd(self, bn, good_cpd):
        """The CPD of the target node should be the same as the self-defined table after adding"""

        bn.set_cpd("b", good_cpd)
        assert bn.cpds["b"].values.tolist() == good_cpd.values.tolist()

    def test_set_other_cpd(self, bn, good_cpd):
        """The CPD of nodes other than the target node should not be affected"""

        cpd = bn.cpds["a"].values.tolist()
        bn.set_cpd("b", good_cpd)
        cpd_after_adding = bn.cpds["a"].values.tolist()

        assert all(
            val == val_after_adding
            for val, val_after_adding in zip(*(cpd, cpd_after_adding))
        )

    def test_set_cpd_to_non_existent_node(self, bn, good_cpd):
        """Should raise error if adding a cpd to a non-existing node in Bayesian Network"""

        with pytest.raises(
            ValueError,
            match=r'Non-existing node "test"',
        ):
            bn.set_cpd("test", good_cpd)

    def test_set_bad_cpd(self, bn, bad_cpd):
        """Should raise error if it the prpbability values do not sum up to 1 in the table"""

        with pytest.raises(
            ValueError,
            match=r"Sum or integral of conditional probabilites for node b is not equal to 1.",
        ):
            bn.set_cpd("b", bad_cpd)

    def test_no_overwritten_after_setting_bad_cpd(self, bn, bad_cpd):
        """The cpd of bn won't be overwritten if adding a bad cpd"""

        original_cpd = bn.cpds["b"].values.tolist()

        try:
            bn.set_cpd("b", bad_cpd)
        except ValueError:
            assert bn.cpds["b"].values.tolist() == original_cpd

    def test_bad_node_index(self, bn, good_cpd):
        """Should raise an error when setting bad node index"""

        bad_cpd = good_cpd
        bad_cpd.index.name = "test"

        with pytest.raises(
            IndexError,
            match=r"Wrong index values. Please check your indices",
        ):
            bn.set_cpd("b", bad_cpd)

    def test_bad_node_states_index(self, bn, good_cpd):
        """Should raise an error when setting bad node states index"""

        bad_cpd = good_cpd.reindex([1, 2, 3])

        with pytest.raises(
            IndexError,
            match=r"Wrong index values. Please check your indices",
        ):
            bn.set_cpd("b", bad_cpd)

    def test_bad_parent_node_index(self, bn, good_cpd):
        """Should raise an error when setting bad parent node index"""

        bad_cpd = good_cpd
        bad_cpd.columns = bad_cpd.columns.rename("test", level=1)

        with pytest.raises(
            IndexError,
            match=r"Wrong index values. Please check your indices",
        ):
            bn.set_cpd("b", bad_cpd)

    def test_bad_parent_node_states_index(self, bn, good_cpd):
        """Should raise an error when setting bad parent node states index"""

        bad_cpd = good_cpd
        bad_cpd.columns.set_levels(["test1", "test2"], level=0, inplace=True)

        with pytest.raises(
            IndexError,
            match=r"Wrong index values. Please check your indices",
        ):
            bn.set_cpd("b", bad_cpd)


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
