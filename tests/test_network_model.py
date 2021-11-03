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
import pytest

from causalnex.network.sklearn import BayesianNetworkClassifier


class TestBayesianCPDs:
    def test_default_params(self):
        edge_list = [
            ("b", "a"),
            ("b", "c"),
            ("d", "a"),
            ("d", "c"),
            ("d", "b"),
            ("e", "c"),
            ("e", "b"),
        ]
        clf = BayesianNetworkClassifier(edge_list)
        params = clf.get_params()
        assert params["discretiser_alg"] == {}
        assert params["probability_kwargs"]["method"] == "BayesianEstimator"
        assert params["probability_kwargs"]["bayes_prior"] == "K2"
        assert params["discretiser_kwargs"] == {}

    def test_predict_quantile(self, iris_test_data, iris_edge_list):
        df = iris_test_data.copy()
        ground_truth = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
                [1, 1, 1, 2, 1, 2, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 1, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 1, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 1, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            ]
        )

        discretiser_params = {
            "sepal width (cm)": {"method": "quantile", "num_buckets": 3},
            "petal length (cm)": {"method": "quantile", "num_buckets": 3},
            "petal width (cm)": {"method": "quantile", "num_buckets": 3},
        }

        label = df["sepal length (cm)"]
        df.drop(["sepal length (cm)"], axis=1, inplace=True)
        clf = BayesianNetworkClassifier(
            iris_edge_list,
            discretiser_kwargs=discretiser_params,
            discretiser_alg={
                "sepal width (cm)": "unsupervised",
                "petal length (cm)": "unsupervised",
                "petal width (cm)": "unsupervised",
            },
        )
        clf.fit(df, label)
        output = clf.predict(df)
        assert np.array_equal(output.reshape(15, -1), ground_truth)

    def test_predict_fixed(self, iris_test_data, iris_edge_list):
        df = iris_test_data.copy()

        ground_truth = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 1, 1, 1, 2, 0, 1, 1],
                [0, 2, 1, 1, 0, 2, 2, 1, 1, 1],
                [2, 1, 1, 1, 1, 2, 1, 2, 1, 0],
                [1, 1, 1, 1, 2, 2, 2, 1, 2, 1],
                [1, 2, 1, 0, 1, 2, 1, 1, 0, 1],
                [2, 1, 2, 1, 2, 2, 1, 1, 1, 2],
                [2, 1, 2, 1, 1, 2, 2, 2, 1, 1],
                [2, 1, 1, 1, 2, 2, 1, 2, 1, 2],
                [1, 2, 1, 1, 1, 2, 2, 2, 2, 2],
                [2, 2, 1, 2, 2, 2, 1, 2, 2, 2],
            ]
        )

        discretiser_params = {
            "sepal width (cm)": {"method": "fixed", "numeric_split_points": [3]},
            "petal length (cm)": {"method": "fixed", "numeric_split_points": [3.7]},
            "petal width (cm)": {"method": "fixed", "numeric_split_points": [1.2]},
        }

        label = df["sepal length (cm)"]
        df.drop(["sepal length (cm)"], axis=1, inplace=True)
        clf = BayesianNetworkClassifier(
            iris_edge_list,
            discretiser_kwargs=discretiser_params,
            discretiser_alg={
                "sepal width (cm)": "unsupervised",
                "petal length (cm)": "unsupervised",
                "petal width (cm)": "unsupervised",
            },
        )
        clf.fit(df, label)
        output = clf.predict(df)
        assert np.array_equal(output.reshape(15, -1), ground_truth)

    def test_return_probability(self, iris_test_data, iris_edge_list):
        df = iris_test_data.copy()

        discretiser_params = {
            "sepal width (cm)": {"method": "fixed", "numeric_split_points": [3]},
            "petal length (cm)": {"method": "fixed", "numeric_split_points": [3.7]},
            "petal width (cm)": {"method": "fixed", "numeric_split_points": [1.2]},
        }

        label = df["sepal length (cm)"]
        df.drop(["sepal length (cm)"], axis=1, inplace=True)
        clf = BayesianNetworkClassifier(
            iris_edge_list,
            discretiser_kwargs=discretiser_params,
            discretiser_alg={
                "sepal width (cm)": "unsupervised",
                "petal length (cm)": "unsupervised",
                "petal width (cm)": "unsupervised",
            },
            return_prob=True,
        )
        clf.fit(df, label)
        output = clf.predict(df.iloc[0:1])
        assert len(list(output)) == 3
        assert math.isclose(
            output["sepal length (cm)_0"].values, 0.764706, abs_tol=1e-3
        )
        assert math.isclose(
            output["sepal length (cm)_1"].values, 0.215686, abs_tol=1e-3
        )

    def test_no_fit(self, iris_test_data, iris_edge_list):
        df = iris_test_data.copy()
        df.drop(["sepal length (cm)"], axis=1, inplace=True)
        clf = BayesianNetworkClassifier(iris_edge_list)
        with pytest.raises(
            ValueError,
            match="No CPDs found. The model has not been fitted",
        ):
            clf.predict(df)

    def test_dt_discretiser(self, iris_test_data, iris_edge_list):
        df = iris_test_data.copy()
        ground_truth = np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [2, 2, 2, 1, 1, 1, 2, 0, 1, 1],
                [0, 1, 1, 1, 1, 2, 1, 1, 1, 1],
                [2, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                [2, 1, 2, 2, 2, 2, 1, 2, 2, 2],
                [2, 2, 2, 1, 1, 2, 2, 2, 2, 1],
                [2, 1, 2, 1, 2, 2, 1, 1, 2, 2],
                [2, 2, 2, 1, 2, 2, 2, 2, 1, 2],
                [2, 2, 1, 2, 2, 2, 1, 2, 2, 1],
            ]
        )
        supervised_param = {
            "sepal width (cm)": {"max_depth": 2, "random_state": 2020},
            "petal length (cm)": {"max_depth": 2, "random_state": 2020},
            "petal width (cm)": {"max_depth": 2, "random_state": 2020},
        }

        label = df["sepal length (cm)"]
        df.drop(["sepal length (cm)"], axis=1, inplace=True)
        clf = BayesianNetworkClassifier(
            iris_edge_list,
            discretiser_kwargs=supervised_param,
            discretiser_alg={
                "sepal width (cm)": "tree",
                "petal length (cm)": "tree",
                "petal width (cm)": "tree",
            },
        )
        clf.fit(df, label)
        output = clf.predict(df)
        assert np.array_equal(output.reshape(15, -1), ground_truth)

    def test_mdlp_discretiser(self, iris_test_data, iris_edge_list):
        df = iris_test_data.copy()
        ground_truth = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 1, 1, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0, 2, 1, 1, 1, 0],
                [2, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 2, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                [2, 2, 2, 2, 2, 2, 1, 2, 2, 2],
                [2, 2, 2, 1, 2, 2, 2, 2, 2, 1],
                [2, 1, 2, 1, 2, 2, 1, 1, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 1, 2],
                [2, 2, 2, 2, 2, 2, 1, 2, 2, 2],
            ]
        )
        supervised_param = {
            "sepal width (cm)": {"min_depth": 0, "random_state": 2020},
            "petal length (cm)": {"min_depth": 0, "random_state": 2020},
            "petal width (cm)": {"min_depth": 0, "random_state": 2020},
        }
        label = df["sepal length (cm)"]
        df.drop(["sepal length (cm)"], axis=1, inplace=True)
        clf = BayesianNetworkClassifier(
            iris_edge_list,
            discretiser_alg={
                "sepal width (cm)": "mdlp",
                "petal length (cm)": "mdlp",
                "petal width (cm)": "mdlp",
            },
            discretiser_kwargs=supervised_param,
        )
        clf.fit(df, label)
        output = clf.predict(df)
        assert np.array_equal(output.reshape(15, -1), ground_truth)

    def test_invalid_algorithm(self, iris_edge_list):

        with pytest.raises(
            KeyError, match="Some discretiser algorithms are not supported"
        ):
            BayesianNetworkClassifier(
                iris_edge_list,
                discretiser_alg={
                    "sepal width (cm)": "invalid",
                    "petal length (cm)": "invalid",
                    "petal width (cm)": "mdlp",
                },
            )

    def test_missing_kwargs(self, iris_edge_list):
        supervised_param = {
            "sepal width (cm)": {"min_depth": 0, "random_state": 2020},
            "petal length (cm)": {"min_depth": 0, "random_state": 2020},
        }
        discretiser_alg = {
            "sepal width (cm)": "tree",
            "petal length (cm)": "tree",
            "petal width (cm)": "mdlp",
        }
        with pytest.raises(
            ValueError,
            match="discretiser_alg and discretiser_kwargs should have the same keys",
        ):
            BayesianNetworkClassifier(
                iris_edge_list,
                discretiser_alg=discretiser_alg,
                discretiser_kwargs=supervised_param,
            )

    def test_shuffled_data(self, iris_test_data, iris_edge_list):
        df = iris_test_data.copy()
        df = df.sample(frac=0.5, random_state=2020)
        ground_truth = np.array(
            [
                [2, 0, 1, 2, 2, 1, 2, 0, 0, 0, 2, 1, 0, 2, 2],
                [0, 1, 2, 2, 0, 0, 1, 2, 0, 2, 1, 1, 2, 0, 0],
                [2, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 2, 1, 0, 2],
                [2, 1, 2, 2, 0, 0, 1, 1, 0, 2, 1, 2, 2, 1, 1],
                [0, 0, 0, 0, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 0],
            ]
        )
        supervised_param = {
            "sepal width (cm)": {"max_depth": 2, "random_state": 2020},
            "petal length (cm)": {"max_depth": 2, "random_state": 2020},
            "petal width (cm)": {"max_depth": 2, "random_state": 2020},
        }

        label = df["sepal length (cm)"]
        df.drop(["sepal length (cm)"], axis=1, inplace=True)
        clf = BayesianNetworkClassifier(
            iris_edge_list,
            discretiser_kwargs=supervised_param,
            discretiser_alg={
                "sepal width (cm)": "tree",
                "petal length (cm)": "tree",
                "petal width (cm)": "tree",
            },
        )
        clf.fit(df, label)
        output = clf.predict(df)
        assert np.isnan(output).sum() == 0
        assert (ground_truth == output.reshape(5, 15)).all()
