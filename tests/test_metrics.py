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
import random

import numpy as np
import pandas as pd

from causalnex.evaluation import classification_report, roc_auc
from causalnex.network import BayesianNetwork
from causalnex.structure.notears import from_pandas
from causalnex.structure.structuremodel import StructureModel


class TestROCAUCStates:
    """Test behaviour of the roc_auc_states metric"""

    def test_roc_of_incorrect_has_fpr_lt_tpr(self):
        """The ROC of incorrect predictions should have FPR < TPR"""

        # regardless of a or b, c=1 is always more likely to varying amounts (to create multiple threshold
        # points in roc curve)
        train = pd.DataFrame(
            [[a, b, 0] for a in range(3) for b in range(3) for _ in range(1)]
            + [
                [a, b, 1]
                for a in range(3)
                for b in range(3)
                for _ in range(a * 1000 + b * 1000 + 1000)
            ],
            columns=["a", "b", "c"],
        )

        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "c", 1), ("b", "c", 1)])

        bn = BayesianNetwork(cg)
        bn.fit_node_states(train)
        bn.fit_cpds(train)

        assert np.allclose(bn.cpds["c"].loc[1].values, 1, atol=0.02)

        # in test, c=0 is always more likely (opposite of train)
        test = pd.DataFrame(
            [[a, b, 0] for a in range(3) for b in range(3) for _ in range(1000)]
            + [[a, b, 1] for a in range(3) for b in range(3) for _ in range(1)],
            columns=["a", "b", "c"],
        )

        roc, _ = roc_auc(bn, test, "c")

        assert len(roc) > 3
        assert all(fpr > tpr for fpr, tpr in roc if tpr not in [0.0, 1.0])

    def test_auc_of_incorrect_close_to_zero(self):
        """The AUC of incorrect predictions should be close to zero"""

        # regardless of a or b, c=1 is always more likely to varying amounts (to create multiple threshold
        # points in roc curve)
        train = pd.DataFrame(
            [[a, b, 0] for a in range(3) for b in range(3) for _ in range(1)]
            + [
                [a, b, 1]
                for a in range(3)
                for b in range(3)
                for _ in range(a * 1000 + b * 1000 + 1000)
            ],
            columns=["a", "b", "c"],
        )

        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "c", 1), ("b", "c", 1)])

        bn = BayesianNetwork(cg)
        bn.fit_node_states(train)
        bn.fit_cpds(train)

        assert np.allclose(bn.cpds["c"].loc[1].values, 1, atol=0.02)

        # in test, c=0 is always more likely (opposite of train)
        test = pd.DataFrame(
            [[a, b, 0] for a in range(3) for b in range(3) for _ in range(1000)]
            + [[a, b, 1] for a in range(3) for b in range(3) for _ in range(1)],
            columns=["a", "b", "c"],
        )

        _, auc = roc_auc(bn, test, "c")

        assert math.isclose(auc, 0, abs_tol=0.001)

    def test_roc_of_random_has_unit_gradient(self):
        """The ROC curve for random predictions should be a line from (0,0) to (1,1)"""

        # regardless of a or b, c=1 is always more likely to varying amounts (to create multiple threshold
        # points in roc curve)
        train = pd.DataFrame(
            [[a, b, 0] for a in range(3) for b in range(3) for _ in range(1)]
            + [
                [a, b, 1]
                for a in range(3)
                for b in range(3)
                for _ in range(a * 1000 + b * 1000 + 1000)
            ],
            columns=["a", "b", "c"],
        )

        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "c", 1), ("b", "c", 1)])

        bn = BayesianNetwork(cg)
        bn.fit_node_states(train)
        bn.fit_cpds(train)

        assert np.allclose(bn.cpds["c"].loc[1].values, 1, atol=0.02)

        test = pd.DataFrame(
            [
                [a, b, random.randint(0, 1)]
                for a in range(3)
                for b in range(3)
                for _ in range(1000)
            ],
            columns=["a", "b", "c"],
        )

        roc, _ = roc_auc(bn, test, "c")

        assert len(roc) > 3
        assert all(math.isclose(a, b, abs_tol=0.03) for a, b in roc)

    def test_auc_of_random_is_half(self):
        """The AUC of random predictions should be 0.5"""

        # regardless of a or b, c=1 is always more likely to varying amounts (to create multiple threshold
        # points in roc curve)
        train = pd.DataFrame(
            [[a, b, 0] for _ in range(10) for a in range(3) for b in range(3)]
            + [
                [a, b, 1]
                for a in range(3)
                for b in range(3)
                for _ in range(a * 1000 + b * 1000 + 1000)
            ],
            columns=["a", "b", "c"],
        )

        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "c", 1), ("b", "c", 1)])

        bn = BayesianNetwork(cg)
        bn.fit_node_states(train)
        bn.fit_cpds(train)

        assert np.allclose(bn.cpds["c"].loc[1].values, 1, atol=0.02)

        test = pd.DataFrame(
            [
                [a, b, random.randint(0, 1)]
                for a in range(3)
                for b in range(3)
                for _ in range(1000)
            ],
            columns=["a", "b", "c"],
        )

        _, auc = roc_auc(bn, test, "c")

        assert math.isclose(auc, 0.5, abs_tol=0.03)

    def test_roc_of_accurate_predictions(self):
        """TPR should always be better than FPR for accurate predictions"""

        # equal class (c) weighting to guarantee high ROC expected
        train = pd.DataFrame(
            [[a, b, 0] for a in range(0, 2) for b in range(0, 2) for _ in range(10)]
            + [
                [a, b, 1]
                for a in range(0, 2)
                for b in range(0, 2)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [
                [a, b, 0]
                for a in range(2, 4)
                for b in range(2, 4)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [[a, b, 1] for a in range(2, 4) for b in range(2, 4) for _ in range(10)],
            columns=["a", "b", "c"],
        )

        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "c", 1), ("b", "c", 1)])

        bn = BayesianNetwork(cg)
        bn.fit_node_states(train)
        bn.fit_cpds(train)

        roc, _ = roc_auc(bn, train, "c")
        assert all(tpr > fpr for fpr, tpr in roc if tpr not in [0.0, 1.0])

    def test_auc_of_accurate_predictions(self):
        """AUC of accurate predictions should be 1"""

        # equal class (c) weighting to guarantee high ROC expected
        train = pd.DataFrame(
            [[a, b, 0] for a in range(0, 2) for b in range(0, 2) for _ in range(1)]
            + [
                [a, b, 1]
                for a in range(0, 2)
                for b in range(0, 2)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [
                [a, b, 0]
                for a in range(2, 4)
                for b in range(2, 4)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [[a, b, 1] for a in range(2, 4) for b in range(2, 4) for _ in range(1)],
            columns=["a", "b", "c"],
        )

        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "c", 1), ("b", "c", 1)])

        bn = BayesianNetwork(cg)
        bn.fit_node_states(train)
        bn.fit_cpds(train)

        _, auc = roc_auc(bn, train, "c")
        assert math.isclose(auc, 1, abs_tol=0.001)

    def test_auc_with_missing_state_in_test(self):
        """AUC should still be calculated correctly with states missing in test set"""

        # equal class (c) weighting to guarantee high ROC expected
        train = pd.DataFrame(
            [[a, b, 0] for a in range(0, 2) for b in range(0, 2) for _ in range(1)]
            + [
                [a, b, 1]
                for a in range(0, 2)
                for b in range(0, 2)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [
                [a, b, 0]
                for a in range(2, 4)
                for b in range(2, 4)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [[a, b, 1] for a in range(2, 4) for b in range(2, 4) for _ in range(1)],
            columns=["a", "b", "c"],
        )

        test = train[train["c"] == 1]
        assert len(test["c"].unique()) == 1

        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "c", 1), ("b", "c", 1)])

        bn = BayesianNetwork(cg)
        bn.fit_node_states(train)
        bn.fit_cpds(train)

        _, auc = roc_auc(bn, test, "c")
        assert math.isclose(auc, 1, abs_tol=0.01)

    def test_auc_node_with_no_parents(self):
        """Should be possible to compute auc for state with no parent nodes"""

        train = pd.DataFrame(
            [[a, b, 0] for a in range(0, 2) for b in range(0, 2) for _ in range(1)]
            + [
                [a, b, 1]
                for a in range(0, 2)
                for b in range(0, 2)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [
                [a, b, 0]
                for a in range(2, 4)
                for b in range(2, 4)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [[a, b, 1] for a in range(2, 4) for b in range(2, 4) for _ in range(1)],
            columns=["a", "b", "c"],
        )

        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "c", 1), ("b", "c", 1)])

        bn = BayesianNetwork(cg)
        bn.fit_node_states(train)
        bn.fit_cpds(train)

        _, auc = roc_auc(bn, train, "a")
        assert math.isclose(auc, 0.5, abs_tol=0.01)

    def test_auc_for_nonnumeric_features(self):
        """AUC of accurate predictions should be 1 even after remapping numbers to strings"""

        # equal class (c) weighting to guarantee high ROC expected
        train = pd.DataFrame(
            [[a, b, 0] for a in range(0, 2) for b in range(0, 2) for _ in range(1)]
            + [
                [a, b, 1]
                for a in range(0, 2)
                for b in range(0, 2)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [
                [a, b, 0]
                for a in range(2, 4)
                for b in range(2, 4)
                for _ in range(a * 10 + b * 10 + 1000)
            ]
            + [[a, b, 1] for a in range(2, 4) for b in range(2, 4) for _ in range(1)],
            columns=["a", "b", "c"],
        )

        # remap values in column c
        train["c"] = train["c"].map({0: "f", 1: "g"})

        cg = StructureModel()
        cg.add_weighted_edges_from([("a", "c", 1), ("b", "c", 1)])

        bn = BayesianNetwork(cg)
        bn.fit_node_states(train)
        bn.fit_cpds(train)

        _, auc = roc_auc(bn, train, "c")
        assert math.isclose(auc, 1, abs_tol=0.001)


class TestClassificationReport:
    """Test behaviour of classification_report"""

    def test_contains_all_class_data(
        self, test_data_c_discrete, bn, test_data_c_likelihood
    ):
        """Check that the report contains data on each possible class"""

        report = classification_report(bn, test_data_c_discrete, "c")

        assert (label in report for label in test_data_c_likelihood.columns)

    def test_report_ignores_unrequired_columns_in_data(
        self, train_data_idx, train_data_discrete, test_data_c_discrete
    ):
        """Classification report should ignore any columns that are no needed by predict"""

        bn = BayesianNetwork(
            from_pandas(train_data_idx, w_threshold=0.3)
        ).fit_node_states(train_data_discrete)
        train_data_discrete["NEW_COL"] = [1] * len(train_data_discrete)
        bn.fit_cpds(train_data_discrete)
        classification_report(bn, test_data_c_discrete, "c")

    def test_report_on_node_with_no_parents_based_on_modal_state(
        self, bn, train_data_discrete
    ):
        """Classification Report on a node with no parents should reflect that predictions are on modal state"""

        report = classification_report(bn, train_data_discrete, "d")

        assert report["d_False"]["recall"] == 1  # always predicts most likely class
        assert report["d_True"]["recall"] == 0
