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
from IPython.display import Image
from mock import patch
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold, cross_val_score

from causalnex.structure import DAGClassifier, DAGRegressor
from causalnex.structure import data_generators as dg


class TestDAGSklearn:
    """ Tests aspects common to both DAGRegressor and DAGClassifier """

    @pytest.mark.parametrize("model", [DAGRegressor, DAGClassifier])
    @pytest.mark.parametrize(
        "val, msg, error",
        [
            ({"alpha": "0.0"}, "alpha should be numeric", TypeError),
            ({"beta": "0.0"}, "beta should be numeric", TypeError),
            ({"fit_intercept": 0}, "fit_intercept should be a bool", TypeError),
            ({"threshold": "0.0"}, "threshold should be numeric", TypeError),
        ],
    )
    def test_input_type_assertion(self, val, msg, error, model):
        with pytest.raises(error, match=msg):
            model(**val)

    @pytest.mark.parametrize("model", [DAGRegressor, DAGClassifier])
    def test_notfitted_error(self, model):
        m = model()
        X = np.random.normal(size=(100, 2))
        with pytest.raises(NotFittedError):
            m.predict(X)

    @pytest.mark.parametrize(
        "model, y",
        [
            (DAGRegressor, np.random.normal(size=(100,))),
            (DAGClassifier, np.random.randint(2, size=(100,))),
        ],
    )
    def test_tabu_parent_nodes(self, model, y):
        X = np.random.normal(size=(100, 2))
        X, y = pd.DataFrame(X), pd.Series(y, name="test")

        m = model(dependent_target=True, tabu_parent_nodes=["test"])
        assert "test" in m.tabu_parent_nodes

        m = model(dependent_target=True, tabu_parent_nodes=[])
        m.fit(X, y)
        assert "test" not in m.tabu_parent_nodes

    @pytest.mark.parametrize(
        "model, y",
        [
            (DAGRegressor, np.random.normal(size=(100,))),
            (DAGClassifier, np.random.randint(2, size=(100,))),
        ],
    )
    def test_numpy_fit(self, model, y):
        m = model()
        X = np.random.normal(size=(100, 2))
        m.fit(X, y)

    @pytest.mark.parametrize(
        "model, y",
        [
            (DAGRegressor, np.random.normal(size=(100,))),
            (DAGClassifier, np.random.randint(2, size=(100,))),
        ],
    )
    def test_pandas_fit(self, model, y):
        m = model()
        X = np.random.normal(size=(100, 2))
        X, y = pd.DataFrame(X), pd.Series(y)
        m.fit(X, y)

    @pytest.mark.parametrize(
        "model, y",
        [
            (DAGRegressor, np.random.normal(size=(100,))),
            (DAGClassifier, np.random.randint(2, size=(100,))),
        ],
    )
    @pytest.mark.parametrize("enforce_dag", [True, False])
    def test_plot_dag(self, enforce_dag, model, y):
        m = model()
        X = np.random.normal(size=(100, 2))
        m.fit(X, y)
        image = m.plot_dag(enforce_dag=enforce_dag)
        assert isinstance(image, Image)

    @pytest.mark.parametrize(
        "model, y",
        [
            (DAGRegressor, np.random.normal(size=(100,))),
            (DAGClassifier, np.random.randint(2, size=(100,))),
        ],
    )
    def test_plot_dag_importerror(self, model, y):
        with patch.dict("sys.modules", {"IPython.display": None}):
            m = model()
            X = np.random.normal(size=(100, 2))
            m.fit(X, y)

            with pytest.raises(
                ImportError,
                match=r"plot_dag method requires IPython installed.",
            ):
                m.plot_dag()

    @pytest.mark.parametrize(
        "model, y",
        [
            (DAGRegressor, np.random.normal(size=(100,))),
            (DAGClassifier, np.random.randint(2, size=(100,))),
        ],
    )
    @pytest.mark.parametrize(
        "hidden_layer_units", [None, [], [0], [1], (0,), (1,), [1, 1], (1, 1)]
    )
    def test_hidden_layer_units(self, hidden_layer_units, model, y):
        m = model(hidden_layer_units=hidden_layer_units)
        X = np.random.normal(size=(100, 2))
        m.fit(X, y)

    @pytest.mark.parametrize(
        "model, y",
        [
            (DAGRegressor, np.random.normal(size=(100,))),
            (DAGClassifier, np.random.randint(2, size=(100,))),
        ],
    )
    def test_enforce_dag(self, model, y):
        m = model(enforce_dag=True)
        X = np.random.normal(size=(100, 2))
        X, y = pd.DataFrame(X), pd.Series(y)
        m.fit(X, y)
        assert nx.algorithms.is_directed_acyclic_graph(m.graph_)

    @pytest.mark.parametrize(
        "model, y",
        [
            (DAGRegressor, np.random.normal(size=(100,))),
            (DAGClassifier, np.random.randint(2, size=(100,))),
        ],
    )
    def test_container_predict_type(self, model, y):
        m = model()
        X = np.random.normal(size=(100, 2))
        m.fit(X, y)
        assert isinstance(m.predict(X), np.ndarray)
        m = model()
        X = np.random.normal(size=(100, 2))
        X, y = pd.DataFrame(X), pd.Series(y)
        m.fit(X, y)
        assert isinstance(m.predict(X), np.ndarray)


class TestDAGRegressor:
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_intercept(self, fit_intercept):
        model, y = DAGRegressor, np.random.normal(size=(100,))
        m = model(fit_intercept=fit_intercept)
        X = np.random.normal(size=(100, 2))
        X, y = pd.DataFrame(X), pd.Series(y)
        m.fit(X, y)
        # intercept should return zero when fit_intercept == False
        assert (m.intercept_ == 0) is not fit_intercept
        assert isinstance(m.intercept_, float)

    @pytest.mark.parametrize("hidden_layer_units", [None, [2], [2, 2]])
    def test_coef(self, hidden_layer_units):
        reg = DAGRegressor(hidden_layer_units=hidden_layer_units)
        X, y = (
            pd.DataFrame(np.random.normal(size=(100, 1))),
            pd.Series(np.random.normal(size=(100,))),
        )
        X["true_feat"] = y * -3
        reg.fit(X, y)
        assert isinstance(reg.coef_, np.ndarray)
        coef_ = pd.Series(reg.coef_, index=X.columns)
        # assert that the sign of the coefficient is correct for both nonlinear and linear cases
        assert coef_["true_feat"] < 0

    @pytest.mark.parametrize("hidden_layer_units", [None, [2], [2, 2]])
    def test_feature_importances(self, hidden_layer_units):
        reg = DAGRegressor(hidden_layer_units=hidden_layer_units)
        X, y = (
            pd.DataFrame(np.random.normal(size=(100, 1))),
            pd.Series(np.random.normal(size=(100,))),
        )
        X["true_feat"] = y * -3
        reg.fit(X, y)
        assert isinstance(reg.feature_importances_, np.ndarray)
        coef_ = pd.Series(reg.feature_importances_, index=X.columns)
        # assert that the sign of the coefficient is positive for both nonlinear and linear cases
        assert coef_["true_feat"] > 0

    @pytest.mark.parametrize("standardize", [True, False])
    def test_nonlinear_performance(self, standardize):
        np.random.seed(42)
        sm = dg.generate_structure(num_nodes=10, degree=3)
        sm.threshold_till_dag()
        data = dg.generate_continuous_dataframe(
            sm, n_samples=1000, intercept=True, seed=42, noise_scale=0.1, kernel=RBF(1)
        )
        node = 1
        y = data.iloc[:, node]
        X = data.drop(node, axis=1)

        reg = DAGRegressor(
            alpha=0.0,
            fit_intercept=True,
            dependent_target=True,
            hidden_layer_units=[0],
            standardize=standardize,
        )
        linear_score = cross_val_score(
            reg, X, y, cv=KFold(shuffle=True, random_state=42)
        ).mean()

        reg = DAGRegressor(
            alpha=0.1,
            fit_intercept=True,
            hidden_layer_units=[2],
            standardize=standardize,
        )
        small_nl_score = cross_val_score(
            reg, X, y, cv=KFold(shuffle=True, random_state=42)
        ).mean()

        reg = DAGRegressor(
            alpha=0.1,
            fit_intercept=True,
            hidden_layer_units=[4],
            standardize=standardize,
        )
        medium_nl_score = cross_val_score(
            reg, X, y, cv=KFold(shuffle=True, random_state=42)
        ).mean()

        assert small_nl_score > linear_score
        assert medium_nl_score > small_nl_score


class TestDAGClassifier:
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_intercept_binary(self, fit_intercept):
        model, y = DAGClassifier, np.random.randint(2, size=(100,))
        m = model(fit_intercept=fit_intercept)
        X = np.random.normal(size=(100, 2))
        X, y = pd.DataFrame(X), pd.Series(y)
        m.fit(X, y)
        # intercept should return zero when fit_intercept == False
        assert (m.intercept_[0] == 0) is not fit_intercept
        assert isinstance(m.intercept_, np.ndarray)
        assert len(m.intercept_) == 1

    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_intercept_categorical(self, fit_intercept):
        model, y = DAGClassifier, np.random.randint(3, size=(100,))
        m = model(fit_intercept=fit_intercept)
        X = np.random.normal(size=(100, 2))
        X, y = pd.DataFrame(X), pd.Series(y)
        m.fit(X, y)
        # intercept should return zero when fit_intercept == False
        for intercept in m.intercept_:
            assert (intercept == 0) is not fit_intercept
        assert isinstance(m.intercept_, np.ndarray)
        assert len(m.intercept_) == 3

    @pytest.mark.parametrize("hidden_layer_units", [None, [2], [2, 2]])
    def test_coef_binary(self, hidden_layer_units):
        clf = DAGClassifier(alpha=0.1, hidden_layer_units=hidden_layer_units)
        X, y = (
            pd.DataFrame(np.random.normal(size=(100, 2))),
            pd.Series(np.zeros(shape=(100,), dtype=int)),
        )
        y[X[0] < 0] = 1
        clf.fit(X, y)

        assert isinstance(clf.coef_, np.ndarray)
        coef_ = pd.Series(clf.coef_, index=X.columns)
        # assert that the sign of the coefficient is correct for both nonlinear and linear cases
        assert coef_[0] < 0

    @pytest.mark.parametrize("hidden_layer_units", [None, [2], [2, 2]])
    def test_coef_categorical(self, hidden_layer_units):
        clf = DAGClassifier(alpha=0.1, hidden_layer_units=hidden_layer_units)
        X, y = (
            pd.DataFrame(np.random.normal(size=(100, 2))),
            pd.Series(np.zeros(shape=(100,), dtype=int)),
        )
        y[X[0] < -0.1] = 1
        y[X[0] > 0.1] = 2
        clf.fit(X, y)

        assert isinstance(clf.coef_, np.ndarray)
        assert clf.coef_.shape == (3, 2)
        coef_ = pd.DataFrame(clf.coef_, columns=X.columns)
        # second category is made likely by negative X
        assert coef_.iloc[1, 0] < 0
        # third category is made likely by positive X
        assert coef_.iloc[2, 0] > 0

    @pytest.mark.parametrize("hidden_layer_units", [None, [2], [2, 2]])
    def test_feature_importances_binary(self, hidden_layer_units):
        clf = DAGClassifier(alpha=0.1, hidden_layer_units=hidden_layer_units)
        X, y = (
            pd.DataFrame(np.random.normal(size=(100, 2))),
            pd.Series(np.zeros(shape=(100,), dtype=int)),
        )
        y[X[0] < 0] = 1
        clf.fit(X, y)

        assert isinstance(clf.feature_importances_, np.ndarray)
        coef_ = pd.DataFrame(clf.feature_importances_, columns=X.columns)
        # assert that the sign of the coefficient is positive for both nonlinear and linear cases
        assert coef_.iloc[0, 0] > 0

    @pytest.mark.parametrize("hidden_layer_units", [None, [2], [2, 2]])
    def test_feature_importances_categorical(self, hidden_layer_units):
        clf = DAGClassifier(alpha=0.1, hidden_layer_units=hidden_layer_units)
        X, y = (
            pd.DataFrame(np.random.normal(size=(100, 2))),
            pd.Series(np.zeros(shape=(100,), dtype=int)),
        )
        y[X[0] < -0.1] = 1
        y[X[0] > 0.1] = 2
        clf.fit(X, y)

        assert isinstance(clf.feature_importances_, np.ndarray)
        assert clf.feature_importances_.shape == (3, 2)
        feature_importances_ = pd.DataFrame(clf.feature_importances_, columns=X.columns)
        # assert that the sign of the coefficient is positive for both nonlinear and linear cases
        assert feature_importances_.iloc[1, 0] > 0
        assert feature_importances_.iloc[2, 0] > 0

    @pytest.mark.parametrize("y_type", [float, str, np.int32, np.int64, np.float32])
    def test_value_predict_type_binary(self, y_type):
        clf = DAGClassifier(alpha=0.1)
        X, y = (
            pd.DataFrame(np.random.normal(size=(100, 2))),
            pd.Series(np.zeros(shape=(100,), dtype=y_type)),
        )
        y[X[0] < 0] = y_type(1)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert isinstance(y_pred[0], y_type)
        y_pred_proba = clf.predict_proba(X)
        assert isinstance(y_pred_proba[0, 0], np.float64)
        assert len(y_pred_proba.shape) == 2

    @pytest.mark.parametrize("y_type", [float, str, np.int32, np.int64, np.float32])
    def test_value_predict_type_categorical(self, y_type):
        clf = DAGClassifier(alpha=0.1)
        X, y = (
            pd.DataFrame(np.random.normal(size=(100, 2))),
            pd.Series(np.zeros(shape=(100,), dtype=y_type)),
        )
        y[X[0] < -0.1] = y_type(1)
        y[X[0] > 0.1] = y_type(2)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert isinstance(y_pred[0], y_type)
        y_pred_proba = clf.predict_proba(X)
        assert isinstance(y_pred_proba[0, 0], np.float64)
        assert len(y_pred_proba.shape) == 2

    @pytest.mark.parametrize("y", [np.random.randint(1, size=(100,))])
    def test_class_number_error(self, y):
        clf = DAGClassifier(alpha=0.1)
        X = pd.DataFrame(np.random.normal(size=(100, 2)))
        with pytest.raises(
            ValueError,
            match="This solver needs samples of at least 2 classes"
            " in the data, but the data contains only one"
            " class: 0",
        ):
            clf.fit(X, y)
