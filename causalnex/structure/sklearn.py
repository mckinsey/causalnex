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
This module contains the implementation of ``StructureModelRegressor``.

``StructureModelRegressor`` is a class which wraps the StructureModel in an sklearn interface for regression.
"""

import copy
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from causalnex.structure.pytorch import notears


class StructureModelRegressor(
    BaseEstimator, RegressorMixin
):  # pylint: disable=too-many-instance-attributes
    """
    Regressor wrapper of the StructureModel.
    Implements the sklearn .fit and .predict interface.
    Currently only supports linear NOTEARS fitting by the DAG.

    Example:
    ::
        >>> from causalnex.XXX import StructureModelRegressor
        >>>
        >>> smr = StructureModelRegressor(threshold=0.1)
        >>> smr.fit(X_train, y_train)
        >>>
        >>> y_preds = smr.predict(X_test)
        >>> type(y_preds)
        np.array
        >>>
        >>> type(smr.feature_importances_)
        np.array
    ::

    Attributes:
        feature_importances_ (np.array): An array of edge weights corresponding
                                         positionally to the feature X.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        normalize: bool = False,
        threshold: float = 0.0,
        tabu_edges: List = None,
        tabu_parent_nodes: List = None,
        tabu_child_nodes: List = None,
        dependent_target: bool = True,
        enforce_dag: bool = True,
        **kwargs
    ):
        """
        Args:
            alpha: The l1 regularization applied to the NOTEARS algorithm.

            fit_intercept: Whether to fit an intercept in the structure model
                                  equation. Use this if variables are offset.

            normalize: Whether to normalize the data to have unit variance.
                              Currently assumes that the data is gaussian.

            threshold: The thresholding to apply to the DAG weights.
                               If 0.0, does not apply any threshold.

            tabu_edges: Tabu edges passed directly to the NOTEARS algorithm.

            tabu_parent_nodes: Tabu nodes passed directly to the NOTEARS algorithm.

            tabu_child_nodes: Tabu nodes passed directly to the NOTEARS algorithm.

            dependent_target: If True, constrains NOTEARS so that y can only
                                     be dependent (i.e. cannot have children) and
                                     imputes from parent nodes.

            enforce_dag: If True, thresholds the graph until it is a DAG.
                                NOTE: a properly trained model should be a DAG, and failure
                                indicates other issues. Use of this is only recommended if
                                features have similar units, otherwise comparing edge weight
                                magnatide has limited meaning.

            kwargs: Extra arguments passed to the NOTEARS from_pandas function.

        Raises:
            TypeError: if beta is not a float.
            TypeError: if fit_intercept is not a bool.
            TypeError: if threshold is not a float.
        """

        # core causalnex parameters
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.threshold = threshold
        self.tabu_edges = tabu_edges
        self.tabu_parent_nodes = tabu_parent_nodes
        self.tabu_child_nodes = tabu_child_nodes
        self.kwargs = kwargs

        if not isinstance(alpha, float):
            raise TypeError("alpha should be a float")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept should be a bool")
        if not isinstance(threshold, float):
            raise TypeError("threshold should be a float")

        # sklearn wrapper paramters
        self.dependent_target = dependent_target
        self.enforce_dag = enforce_dag

    def fit(
        self, X: Union[pd.DataFrame, np.array], y: Union[pd.Series, np.array]
    ) -> "StructureModelRegressor":
        """
        Fits the sm model using the concat of X and y.
        """

        # force as DataFrame and Series (for later calculations)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        # force name so that name /= None (errors in notears)
        if y.name is None:
            y.name = "__target"

        # normalize the data to have unit variance
        if self.normalize:
            X /= X.std()
            y /= y.std()

        # preserve the feature and target colnames
        self._features = tuple(X.columns)
        self._target = y.name

        # concat X and y along column axis
        X = pd.concat([X, y], axis=1)

        # make copy to prevent mutability
        tabu_parent_nodes = copy.deepcopy(self.tabu_parent_nodes)
        if self.dependent_target:
            if tabu_parent_nodes is None:
                tabu_parent_nodes = [self._target]
            elif self._target not in tabu_parent_nodes:
                tabu_parent_nodes.append(self._target)

        # fit the structured model
        self._graph = notears.from_pandas(
            X,
            beta=self.alpha,
            w_threshold=self.threshold,
            tabu_edges=self.tabu_edges,
            tabu_parent_nodes=tabu_parent_nodes,
            tabu_child_nodes=self.tabu_child_nodes,
            use_bias=self.fit_intercept,
            **self.kwargs
        )

        # keep thresholding until the DAG constraint is enforced
        if self.enforce_dag:
            self._graph.threshold_till_dag()

        return self

    def _predict_from_parents(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = np.zeros(X.shape[0])
        # multiply the sm weights into the X to get the y predictions
        for (i, j, edge_dict) in self._graph.edges(data=True):
            if j == self._target:
                y_pred += X.loc[:, i] * edge_dict["weight"]

        # add the intercept term
        y_pred += self._graph.nodes[self._target]["bias"]
        return y_pred

    def predict(self, X: Union[pd.DataFrame, np.array]) -> np.ndarray:
        """
        Get the predictions of the structured model.
        This is done by multiplying the edge weights with the feature i.e. X @ W
        """

        # force as DataFrame
        X = pd.DataFrame(X)

        # check that the model has been fit
        check_is_fitted(self, "_graph")

        return np.asarray(self._predict_from_parents(X))

    @property
    def feature_importances(self) -> np.array:
        """
        Getter method which returns the weights as a proxy for the feature importances.
        """
        # build base _feature_importances_
        feature_importances_ = pd.Series(index=self._features)

        # iterate over all edges
        for (i, j, w) in self._graph.edges(data="weight"):
            # for edges directed towards target
            if j == self._target:
                # insert the weight into the series
                feature_importances_[i] = w

        # feature importances are returned as NaNs if edges are removed
        feature_importances_ = feature_importances_.fillna(0)

        return feature_importances_.values

    @property
    def coef_(self) -> np.array:
        """ An alias for the feature_importances method """
        return self.feature_importances

    @property
    def intercept_(self) -> float:
        """ The bias term from the target node """
        return float(self._graph.nodes[self._target]["bias"])
