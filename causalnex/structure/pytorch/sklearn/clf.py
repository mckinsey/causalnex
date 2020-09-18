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
This module contains the implementation of ``DAGClassifier``.

``DAGClassifier`` is a class which wraps the StructureModel in an sklearn interface for classification.
"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets

from causalnex.structure.pytorch.sklearn._base import DAGBase


class DAGClassifier(ClassifierMixin, DAGBase):
    """
    Classifier wrapper of the StructureModel.
    Implements the sklearn .fit and .predict interface.

    Example:
    ::
        >>> from causalnex.sklearn import DAGRegressor
        >>>
        >>> clf = DAGClassifier(threshold=0.1)
        >>> clf.fit(X_train, y_train)
        >>>
        >>> y_preds = clf.predict(X_test)
        >>> type(y_preds)
        np.ndarray
        >>>
        >>> type(clf.feature_importances_)
        np.ndarray
    ::

    Attributes:
        feature_importances_ (np.ndarray): An array of edge weights corresponding
        positionally to the feature X.

        coef_ (np.ndarray): An array of edge weights corresponding
        positionally to the feature X.

        intercept_ (float): The target node bias value.
    """

    def _target_dist_type(self) -> str:
        return self.__target_dist_type

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "DAGClassifier":
        """
        Fits the sm model using the concat of X and y.
        """
        # clf target check
        check_classification_targets(y)

        # encode the categories to be numeric
        enc = LabelEncoder()
        y = y.copy()
        y[:] = enc.fit_transform(y)
        # store the classes from the LabelEncoder
        self.classes_ = enc.classes_

        # class number checks
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: {}".format(self.classes_[0])
            )
        if n_classes > 2:
            raise ValueError("This solver does not support more than 2 classes")

        # store the private attr __target_dist_type
        self.__target_dist_type = "bin"
        # fit the NOTEARS model
        super().fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Uses the fitted NOTEARS algorithm to reconstruct y from known X data.

        Returns:
            Predicted y values for each row of X.
        """
        probs = self.predict_proba(X)

        # get the class by rounding the (0, 1) bound probability
        indices = probs.round().astype(np.int64)

        return self.classes_[indices]

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Uses the fitted NOTEARS algorithm to reconstruct y from known X data.

        Returns:
            Predicted y class probabilities for each row of X.
        """
        return super().predict(X)
