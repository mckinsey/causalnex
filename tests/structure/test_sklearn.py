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

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from causalnex.structure.sklearn import StructureModelRegressor


class TestStructureModel:
    @pytest.mark.parametrize(
        "val, msg",
        [
            ({"alpha": "0.0"}, "alpha should be a float"),
            ({"fit_intercept": 0}, "fit_intercept should be a bool"),
            ({"threshold": "0.0"}, "threshold should be a float"),
        ],
    )
    def test_input_type_assertion(self, val, msg):
        with pytest.raises(TypeError, match=msg):
            StructureModelRegressor(**val)

    def test_pandas_fit(self):
        smr = StructureModelRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        X, y = pd.DataFrame(X), pd.Series(y)
        smr.fit(X, y)

    def test_numpy_fit(self):
        smr = StructureModelRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        smr.fit(X, y)

    def test_predict_type(self):
        smr = StructureModelRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        smr.fit(X, y)
        assert isinstance(smr.predict(X), np.ndarray)
        smr = StructureModelRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        X, y = pd.DataFrame(X), pd.Series(y)
        smr.fit(X, y)
        assert isinstance(smr.predict(X), np.ndarray)

    def test_notfitted_error(self):
        smr = StructureModelRegressor()
        X = np.random.normal(size=(100, 2))

        with pytest.raises(NotFittedError):
            smr.predict(X)

    def test_coef(self):
        smr = StructureModelRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        smr.fit(X, y)

        assert isinstance(smr.coef_, np.ndarray)

    def test_intercept(self):
        smr = StructureModelRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        smr.fit(X, y)

        assert isinstance(smr.intercept_, float)

    def test_feature_importances(self):
        smr = StructureModelRegressor()
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        smr.fit(X, y)

        assert isinstance(smr.feature_importances, np.ndarray)

    def test_tabu_parent_nodes(self):
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        X, y = pd.DataFrame(X), pd.Series(y, name="test")

        smr = StructureModelRegressor(dependent_target=True, tabu_parent_nodes=["test"])
        assert "test" in smr.tabu_parent_nodes

        smr = StructureModelRegressor(dependent_target=True, tabu_parent_nodes=[])
        smr.fit(X, y)
        assert "test" not in smr.tabu_parent_nodes

    def test_normalize(self):
        X, y = np.random.normal(size=(100, 2)), np.random.normal(size=(100,))
        smr = StructureModelRegressor(normalize=True)
        smr.fit(X, y)
        assert isinstance(smr.predict(X), np.ndarray)
