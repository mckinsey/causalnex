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

from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets

from causalnex.structure.pytorch.dist_type.categorical import DistTypeCategorical
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

    _supported_types = ("ord", "bin", "cat")
    _default_type = "cont"

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "DAGClassifier":
        """
        Fits the sm model using the concat of X and y.

        Raises:
            NotImplementedError: If unsupported target_dist_type provided.
            ValueError: If less than 2 classes provided.

        Returns:
            Instance of DAGClassifier.
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
                f" class: {self.classes_[0]}"
            )

        # store the protected attr _target_dist_type
        if self.target_dist_type is None:
            self.target_dist_type = "cat" if n_classes > 2 else "bin"

        # fit the NOTEARS model
        super().fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Uses the fitted NOTEARS algorithm to reconstruct y from known X data.

        Returns:
            Predicted y values for each row of X.
        """
        # get the predicted probabilities
        probs = self.predict_proba(X)

        n_classes = len(self.classes_)
        if n_classes == 2:
            # get the class by rounding the (0, 1) bound probability
            # NOTE: probs is returned as a (n_samples, n_classes) array
            indices = probs[:, 1].round().astype(np.int64)
        else:
            # use max probability in columns to determine class
            indices = np.argmax(probs, axis=1)

        return self.classes_[indices]

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Uses the fitted NOTEARS algorithm to reconstruct y from known X data.

        Returns:
            Predicted y class probabilities for each row of X.
        """
        y_pred = super().predict(X)
        # binary predict returns a (n_samples,) array
        # sklearn interface requires (n_samples, n_classes)
        if len(y_pred.shape) == 1:
            y_pred = np.vstack([1 - y_pred, y_pred]).T
        return y_pred

    @property
    def _target_node_names(self) -> List[str]:
        # target node names are build according to
        target_nodes = []
        for catidx in range(len(self.classes_)):
            target_nodes.append(
                DistTypeCategorical.make_node_name(self._target, catidx)
            )
        return target_nodes

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Unsigned importances of the features wrt to the target.
        NOTE: these are used as the graph adjacency matrix.
        Returns:
            the L2 relationship between nodes.
            shape: (1, n_features) or (n_classes, n_features).
        """
        n_classes = len(self.classes_)
        # handle binary
        if n_classes == 2:
            return np.asarray(self.get_edges_to_node(self._target).values).reshape(
                1, -1
            )

        # handle categorical
        data = []
        for node in self._target_node_names:
            # stack into (n_classes, n_features) format
            data.append(np.asarray(self.get_edges_to_node(node).values).reshape(1, -1))
        return np.vstack(data)

    @property
    def coef_(self) -> np.ndarray:
        """
        Signed relationship between features and the target.
        For this linear case this equivalent to linear regression coefficients.
        Returns:
            the mean effect relationship between nodes.
            shape: (1, n_features) or (n_classes, n_features).
        """
        n_classes = len(self.classes_)
        # handle binary
        if n_classes == 2:
            return np.asarray(
                self.get_edges_to_node(self._target, data="mean_effect")
            ).reshape(1, -1)

        # handle categorical
        data = []
        for node in self._target_node_names:
            # get stack into (n_classes, n_features) format
            data.append(
                np.asarray(
                    self.get_edges_to_node(node, data="mean_effect").values
                ).reshape(1, -1)
            )
        return np.vstack(data)

    @property
    def intercept_(self) -> np.ndarray:
        """
        Returns:
            The bias term from the target node.
            shape: (1,) or (n_classes,).
        """
        n_classes = len(self.classes_)
        # handle binary
        if n_classes == 2:
            bias = self.graph_.nodes[self._target]["bias"]
            bias = 0.0 if bias is None else float(bias)
            return np.array([bias])

        # handle categorical
        biases = []
        for node in self._target_node_names:
            bias = self.graph_.nodes[node]["bias"]
            bias = 0.0 if bias is None else float(bias)
            biases.append(bias)
        return np.array(biases)
