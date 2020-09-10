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

from causalnex.structure import data_generators as dg
from causalnex.structure.sklearn import DAGRegressor


class TestStructureModel:
    @pytest.mark.parametrize(
        "val, msg, error",
        [
            ({"alpha": "0.0"}, "alpha should be numeric", TypeError),
            ({"beta": "0.0"}, "beta should be numeric", TypeError),
            ({"fit_intercept": 0}, "fit_intercept should be a bool", TypeError),
            ({"threshold": "0.0"}, "threshold should be numeric", TypeError),
        ],
    )
    def test_input_type_assertion(self, val, msg, error):
        with pytest.raises(error, match=msg):
            DAGRegressor(**val)

    def test_pandas_fit(self):
        reg = DAGRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        X, y = pd.DataFrame(X), pd.Series(y)
        reg.fit(X, y)

    def test_numpy_fit(self):
        reg = DAGRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        reg.fit(X, y)

    def test_predict_type(self):
        reg = DAGRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        reg.fit(X, y)
        assert isinstance(reg.predict(X), np.ndarray)
        reg = DAGRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        X, y = pd.DataFrame(X), pd.Series(y)
        reg.fit(X, y)
        assert isinstance(reg.predict(X), np.ndarray)

    def test_notfitted_error(self):
        reg = DAGRegressor()
        X = np.random.normal(size=(100, 2))
        with pytest.raises(NotFittedError):
            reg.predict(X)

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

    def test_tabu_parent_nodes(self):
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        X, y = pd.DataFrame(X), pd.Series(y, name="test")

        reg = DAGRegressor(dependent_target=True, tabu_parent_nodes=["test"])
        assert "test" in reg.tabu_parent_nodes

        reg = DAGRegressor(dependent_target=True, tabu_parent_nodes=[])
        reg.fit(X, y)
        assert "test" not in reg.tabu_parent_nodes

    @pytest.mark.parametrize(
        "fit_intercept, equals_zero", [(True, False), (False, True)]
    )
    def test_intercept(self, fit_intercept, equals_zero):
        reg = DAGRegressor(fit_intercept=fit_intercept)
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        X, y = pd.DataFrame(X), pd.Series(y)
        reg.fit(X, y)
        # intercept should return zero when fit_intercept == False
        assert (reg.intercept_ == 0) is equals_zero
        assert isinstance(reg.intercept_, float)

    @pytest.mark.parametrize("enforce_dag", [True, False])
    def test_plot_dag(self, enforce_dag):
        reg = DAGRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        reg.fit(X, y)
        image = reg.plot_dag(enforce_dag=enforce_dag)
        assert isinstance(image, Image)

    def test_plot_dag_importerror(self):
        with patch.dict("sys.modules", {"IPython.display": None}):
            reg = DAGRegressor()
            X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
            reg.fit(X, y)

            with pytest.raises(
                ImportError,
                match=r"DAGRegressor\.plot_dag method requires IPython installed\.",
            ):
                reg.plot_dag()

    @pytest.mark.parametrize(
        "hidden_layer_units", [None, [], [0], [1], (0,), (1,), [1, 1], (1, 1)]
    )
    def test_hidden_layer_units(self, hidden_layer_units):
        reg = DAGRegressor(hidden_layer_units=hidden_layer_units)
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        reg.fit(X, y)

    def test_enforce_dag(self):
        reg = DAGRegressor(enforce_dag=True)
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        X, y = pd.DataFrame(X), pd.Series(y)
        reg.fit(X, y)
        assert nx.algorithms.is_directed_acyclic_graph(reg.graph_)

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
            l1_ratio=0.0,
            fit_intercept=True,
            dependent_target=True,
            enforce_dag=False,
            hidden_layer_units=[0],
            standardize=standardize,
        )
        linear_score = cross_val_score(
            reg, X, y, cv=KFold(shuffle=True, random_state=42)
        ).mean()

        reg = DAGRegressor(
            alpha=0.1,
            l1_ratio=1.0,
            fit_intercept=True,
            enforce_dag=False,
            hidden_layer_units=[2],
            standardize=standardize,
        )
        small_nl_score = cross_val_score(
            reg, X, y, cv=KFold(shuffle=True, random_state=42)
        ).mean()

        reg = DAGRegressor(
            alpha=0.1,
            l1_ratio=1.0,
            fit_intercept=True,
            enforce_dag=False,
            hidden_layer_units=[4],
            standardize=standardize,
        )
        medium_nl_score = cross_val_score(
            reg, X, y, cv=KFold(shuffle=True, random_state=42)
        ).mean()

        assert small_nl_score > linear_score
        assert medium_nl_score > small_nl_score
