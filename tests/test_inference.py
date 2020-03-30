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

import pytest

from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas


class TestInferenceEngineIdx:
    def test_create_inference_from_bn(self, train_model, train_data_idx):
        """It should be possible to create a new Inference object from an existing pgmpy model"""

        bn = BayesianNetwork(train_model).fit_node_states(train_data_idx)
        bn.fit_cpds(train_data_idx)
        InferenceEngine(bn)

    def test_create_inference_with_bad_variable_names_fails(
        self, train_model, train_data_idx
    ):

        model = StructureModel()
        model.add_edges_from(
            [
                (str(u).replace("a", "$a"), str(v).replace("a", "$a"))
                for u, v in train_model.edges
            ]
        )

        train_data_idx.rename(columns={"a": "$a"}, inplace=True)

        bn = BayesianNetwork(model).fit_node_states(train_data_idx)
        bn.fit_cpds(train_data_idx)

        with pytest.raises(ValueError, match="Variable names must match.*"):
            InferenceEngine(bn)

    def test_empty_query_returns_marginals(
        self, train_model, train_data_idx, train_data_idx_marginals
    ):
        """An empty query should return all the marginal probabilities of the model's distribution"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)
        dist = ie.query({})

        for node, states in dist.items():
            for state, p in states.items():
                assert math.isclose(
                    train_data_idx_marginals[node][state], p, abs_tol=0.05
                )

    def test_observations_affect_marginals(self, train_model, train_data_idx):
        """Observing the state of a node should affect the marginals of dependent nodes"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)

        m1 = ie.query({})
        m2 = ie.query({"d": 1})

        assert m2["d"][0] == 0
        assert m2["d"][1] == 1
        assert not math.isclose(m2["b"][1], m1["b"][1], abs_tol=0.01)

    def test_observations_does_not_affect_marginals_of_independent_nodes(
        self, train_model, train_data_idx
    ):
        """Observing the state of a node should not affect the marginal probability of an independent node"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)

        m1 = ie.query({})
        m2 = ie.query({"d": 1})

        assert m2["d"][0] == 0
        assert m2["d"][1] == 1
        assert math.isclose(m2["e"][1], m1["e"][1], abs_tol=0.05)

    def test_do_sets_state_probability_to_one(self, train_model, train_data_idx):
        """Do should update the probability of the given observation=state to 1"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)
        ie.do_intervention("d", 1)
        assert math.isclose(ie.query()["d"][1], 1)

    def test_do_on_node_with_no_effects_not_allowed(self, train_model, train_data_idx):
        """It should not be possible to create an isolated node in the network"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)

        with pytest.raises(
            ValueError,
            match="Do calculus cannot be applied because it would result in an isolate",
        ):
            ie.do_intervention("a", 1)

    def test_do_sets_other_state_probabilitys_to_zero(
        self, train_model, train_data_idx
    ):
        """Do should update the probability of every other state for the observation to zero"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)
        ie.do_intervention("d", 1)
        assert ie.query()["d"][0] == 0

    def test_do_accepts_all_state_probabilities(self, train_model, train_data_idx):
        """Do should accept a map of state->p and update p accordingly"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)
        ie.do_intervention("d", {0: 0.7, 1: 0.3})
        assert math.isclose(ie.query()["d"][0], 0.7)
        assert math.isclose(ie.query()["d"][1], 0.3)

    def test_do_expects_all_state_probabilities_sum_to_one(
        self, train_model, train_data_idx
    ):
        """Do should accept only state probabilities where the full distribution is provided"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)

        with pytest.raises(
            ValueError, match="The cpd for the provided observation must sum to 1"
        ):
            ie.do_intervention("d", {0: 0.7, 1: 0.4})

    def test_do_expects_all_state_probabilities_within_0_and_1(
        self, train_model, train_data_idx
    ):
        """Do should accept only state probabilities where the full distribution is provided"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)

        with pytest.raises(
            ValueError,
            match="The cpd for the provided observation must be between 0 and 1",
        ):
            ie.do_intervention("d", {0: -1.0, 1: 2.0})

    def test_do_expects_all_states_have_a_probability(
        self, train_model, train_data_idx
    ):
        """Do should accept only state probabilities where all states in the original cpds are present"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)

        with pytest.raises(
            ValueError, match="The cpd states do not match expected states*"
        ):
            ie.do_intervention("d", {1: 1})

    def test_do_prevents_new_states_being_added(self, train_model, train_data_idx):
        """Do should not allow the introduction of new states"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)

        with pytest.raises(
            ValueError, match="The cpd states do not match expected states*"
        ):
            ie.do_intervention("d", {0: 0.7, 1: 0.3, 2: 0.0})

    def test_do_reflected_in_query(self, train_model, train_data_idx):
        """Do should adjust marginals returned by query when given a different observation"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)

        assert ie.query({"a": 1})["d"][1] != 1
        ie.do_intervention("d", 1)
        assert ie.query({"a": 1})["d"][1] == 1

    def test_reset_do_sets_probabilities_back_to_initial_state(
        self, train_model, train_data_idx, train_data_idx_marginals
    ):
        """Resetting Do operator should re-introduce the original conditional dependencies"""

        bn = BayesianNetwork(train_model)
        bn.fit_node_states(train_data_idx).fit_cpds(train_data_idx)

        ie = InferenceEngine(bn)
        ie.do_intervention("d", {0: 0.7, 1: 0.3})
        ie.reset_do("d")

        assert math.isclose(ie.query()["d"][0], train_data_idx_marginals["d"][0])
        assert math.isclose(ie.query()["d"][1], train_data_idx_marginals["d"][1])


class TestInferenceEngineDiscrete:
    """Test behaviour of query and interventions"""

    def test_query_when_cpds_not_fit(self, train_data_idx, train_data_discrete):
        """An error should be raised if query before CPDs are fit"""

        bn = BayesianNetwork(
            from_pandas(train_data_idx, w_threshold=0.3)
        ).fit_node_states(train_data_discrete)

        with pytest.raises(
            ValueError, match=r"Bayesian Network does not contain any CPDs.*"
        ):
            InferenceEngine(bn)

    def test_empty_query_returns_marginals(self, bn, train_data_discrete_marginals):
        """An empty query should return all the marginal probabilities of the model's distribution"""

        ie = InferenceEngine(bn)
        dist = ie.query({})

        for node, states in dist.items():
            for state, p in states.items():
                assert math.isclose(
                    train_data_discrete_marginals[node][state], p, abs_tol=0.05
                )

    def test_observations_affect_marginals(self, bn):
        """Observing the state of a node should affect the marginals of dependent nodes"""

        ie = InferenceEngine(bn)

        m1 = ie.query({})
        m2 = ie.query({"d": True})

        assert m2["d"][False] == 0
        assert m2["d"][True] == 1
        assert not math.isclose(m2["b"]["x"], m1["b"]["x"], abs_tol=0.05)

    def test_observations_does_not_affect_marginals_of_independent_nodes(self, bn):
        """Observing the state of a node should not affect the marginal probability of an independent node"""

        ie = InferenceEngine(bn)

        m1 = ie.query({})
        m2 = ie.query({"d": True})

        assert m2["d"][False] == 0
        assert m2["d"][True] == 1
        assert math.isclose(m2["e"][True], m1["e"][True], abs_tol=0.05)

    def test_do_sets_state_probability_to_one(self, bn):
        """Do should update the probability of the given observation=state to 1"""

        ie = InferenceEngine(bn)
        ie.do_intervention("d", True)
        assert math.isclose(ie.query()["d"][True], 1)

    def test_do_on_node_with_no_effects_not_allowed(self, bn):
        """It should not be possible to create an isolated node in the network"""

        ie = InferenceEngine(bn)

        with pytest.raises(
            ValueError,
            match="Do calculus cannot be applied because it would result in an isolate",
        ):
            ie.do_intervention("a", "b")

    def test_do_sets_other_state_probabilitys_to_zero(self, bn):
        """Do should update the probability of every other state for the observation to zero"""

        ie = InferenceEngine(bn)
        ie.do_intervention("d", True)
        assert ie.query()["d"][False] == 0

    def test_do_accepts_all_state_probabilities(self, bn):
        """Do should accept a map of state->p and update p accordingly"""

        ie = InferenceEngine(bn)
        ie.do_intervention("d", {False: 0.7, True: 0.3})
        assert math.isclose(ie.query()["d"][False], 0.7)
        assert math.isclose(ie.query()["d"][True], 0.3)

    def test_do_expects_all_state_probabilities_sum_to_one(self, bn):
        """Do should accept only state probabilities where the full distribution is provided"""

        ie = InferenceEngine(bn)

        with pytest.raises(
            ValueError, match="The cpd for the provided observation must sum to 1"
        ):
            ie.do_intervention("d", {False: 0.7, True: 0.4})

    def test_do_expects_all_states_have_a_probability(self, bn):
        """Do should accept only state probabilities where all states in the original cpds are present"""

        ie = InferenceEngine(bn)

        with pytest.raises(
            ValueError, match="The cpd states do not match expected states*"
        ):
            ie.do_intervention("d", {False: 1})

    def test_do_prevents_new_states_being_added(self, bn):
        """Do should not allow the introduction of new states"""

        ie = InferenceEngine(bn)

        with pytest.raises(
            ValueError, match="The cpd states do not match expected states*"
        ):
            ie.do_intervention("d", {False: 0.7, True: 0.3, "other": 0.0})

    def test_do_reflected_in_query(self, bn):
        """Do should adjust marginals returned by query when given a different observation"""

        ie = InferenceEngine(bn)

        assert ie.query({"a": "b"})["d"][True] != 1
        ie.do_intervention("d", True)
        assert ie.query({"a": "b"})["d"][True] == 1

    def test_reset_do_sets_probabilities_back_to_initial_state(
        self, bn, train_data_discrete_marginals
    ):
        """Resetting Do operator should re-introduce the original conditional dependencies"""

        ie = InferenceEngine(bn)
        ie.do_intervention("d", {False: 0.7, True: 0.3})
        ie.reset_do("d")

        assert math.isclose(
            ie.query()["d"][False], train_data_discrete_marginals["d"][False]
        )
        assert math.isclose(
            ie.query()["d"][False], train_data_discrete_marginals["d"][False]
        )
