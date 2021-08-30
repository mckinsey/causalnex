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
"""
This module contains the implementation of ``DAGRegressor``.

``DAGRegressor`` is a class which wraps the StructureModel in an sklearn interface for regression.
"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from causalnex.structure.pytorch.sklearn._base import DAGBase


class DAGRegressor(RegressorMixin, DAGBase):
    """
    Regressor wrapper of the StructureModel.
    Implements the sklearn .fit and .predict interface.

    Example:
    ::
        >>> from causalnex.sklearn import DAGRegressor
        >>>
        >>> reg = DAGRegressor(threshold=0.1)
        >>> reg.fit(X_train, y_train)
        >>>
        >>> y_preds = reg.predict(X_test)
        >>> type(y_preds)
        np.ndarray
        >>>
        >>> type(reg.feature_importances_)
        np.ndarray
    ::

    Attributes:
        feature_importances_ (np.ndarray): An array of edge weights corresponding
        positionally to the feature X.

        coef_ (np.ndarray): An array of edge weights corresponding
        positionally to the feature X.

        intercept_ (float): The target node bias value.
    """

    _supported_types = ("cont", "poiss")

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "DAGRegressor":
        """
        Fits the sm model using the concat of X and y.

        Raises:
            NotImplementedError: If unsupported _target_dist_type provided.

        Returns:
            Instance of DAGRegressor.
        """

        # store the protected attr _target_dist_type
        if self.target_dist_type is None:
            self.target_dist_type = "cont"

        # fit the NOTEARS model
        super().fit(X, y)
        return self
