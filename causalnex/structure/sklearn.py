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

import copy
import warnings
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y

from causalnex.plots import EDGE_STYLE, NODE_STYLE, plot_structure
from causalnex.structure.pytorch import notears


class DAGRegressor(
    BaseEstimator, RegressorMixin
):  # pylint: disable=too-many-instance-attributes
    """
    Regressor wrapper of the StructureModel.
    Implements the sklearn .fit and .predict interface.
    Currently only supports linear NOTEARS fitting by the DAG.

    Example:
    ::
        >>> from causalnex.sklearn import DAGRegressor
        >>>
        >>> smr = DAGRegressor(threshold=0.1)
        >>> smr.fit(X_train, y_train)
        >>>
        >>> y_preds = smr.predict(X_test)
        >>> type(y_preds)
        np.ndarray
        >>>
        >>> type(smr.feature_importances_)
        np.ndarray
    ::

    Attributes:
        feature_importances_ (np.ndarray): An array of edge weights corresponding
                                         positionally to the feature X.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        fit_intercept: bool = True,
        hidden_layer_units: Iterable[int] = None,
        threshold: float = 0.0,
        tabu_edges: List = None,
        tabu_parent_nodes: List = None,
        tabu_child_nodes: List = None,
        dependent_target: bool = True,
        enforce_dag: bool = False,
        standardize: bool = False,
        **kwargs
    ):
        """
        Args:
            alpha: l1 loss weighting. When using nonlinear layers this is only applied
            to the first layer.

            beta: l2 loss weighting. Applied across all layers. Reccomended to use this
            when fitting nonlinearities.

            fit_intercept: Whether to fit an intercept in the structure model
            equation. Use this if variables are offset.

            hidden_layer_units: An iterable where its length determine the number of layers used,
            and the numbers determine the number of nodes used for the layer in order.

            threshold: The thresholding to apply to the DAG weights.
            If 0.0, does not apply any threshold.

            tabu_edges: Tabu edges passed directly to the NOTEARS algorithm.

            tabu_parent_nodes: Tabu nodes passed directly to the NOTEARS algorithm.

            tabu_child_nodes: Tabu nodes passed directly to the NOTEARS algorithm.

            dependent_target: If True, constrains NOTEARS so that y can only
            be dependent (i.e. cannot have children) and imputes from parent nodes.

            enforce_dag: If True, thresholds the graph until it is a DAG.
            NOTE a properly trained model should be a DAG, and failure
            indicates other issues. Use of this is only recommended if
            features have similar units, otherwise comparing edge weight
            magnitude has limited meaning.

            standardize: Whether to standardize the X and y variables before fitting.
            The L-BFGS algorithm used to fit the underlying NOTEARS works best on data
            all of the same scale so this parameter is reccomended.

            kwargs: Extra arguments passed to the NOTEARS from_pandas function.

        Raises:
            TypeError: if alpha is not numeric.
            TypeError: if beta is not numeric.
            TypeError: if fit_intercept is not a bool.
            TypeError: if threshold is not numeric.
        """

        # core causalnex parameters
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.hidden_layer_units = hidden_layer_units
        self.threshold = threshold
        self.tabu_edges = tabu_edges
        self.tabu_parent_nodes = tabu_parent_nodes
        self.tabu_child_nodes = tabu_child_nodes
        self.kwargs = kwargs

        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha should be numeric")
        if not isinstance(beta, (int, float)):
            raise TypeError("beta should be numeric")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept should be a bool")
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold should be numeric")

        # sklearn wrapper paramters
        self.dependent_target = dependent_target
        self.enforce_dag = enforce_dag
        self.standardize = standardize

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "DAGRegressor":
        """
        Fits the sm model using the concat of X and y.
        """

        # defensive X, y checks
        check_X_y(X, y, y_numeric=True)

        # force as DataFrame and Series (for later calculations)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        # force name so that name != None (causes errors in notears)
        y.name = y.name or "__target"

        if self.standardize:
            self.ss_X = StandardScaler()
            self.ss_y = StandardScaler()
            X = pd.DataFrame(self.ss_X.fit_transform(X), columns=X.columns)
            y = pd.Series(
                self.ss_y.fit_transform(y.values.reshape(-1, 1)).reshape(-1),
                name=y.name,
            )

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
        self.graph_ = notears.from_pandas(
            X,
            lasso_beta=self.alpha,
            ridge_beta=self.beta,
            hidden_layer_units=self.hidden_layer_units,
            w_threshold=self.threshold,
            tabu_edges=self.tabu_edges,
            tabu_parent_nodes=tabu_parent_nodes,
            tabu_child_nodes=self.tabu_child_nodes,
            use_bias=self.fit_intercept,
            **self.kwargs
        )

        # keep thresholding until the DAG constraint is enforced
        if self.enforce_dag:
            self.graph_.threshold_till_dag()

        return self

    def _predict_from_parents(self, X: Union[pd.DataFrame, np.ndarray]):

        # extract the base solver
        structure_learner = self.graph_.graph["structure_learner"]

        # convert the predict data to pytorch tensor
        X = torch.from_numpy(X).float().to(structure_learner.device)
        # need to concat y onto X so that the dimensions are the same
        y = torch.zeros(X.shape[0], 1).float().to(structure_learner.device)
        X = torch.cat([X, y], dim=1)

        # perform forward reconstruction
        X_hat = structure_learner(X)

        # FUTURE NOTE: with dtypes the projection from latent -> dtype goes here

        # extract the desired y column, return as array
        y_pred = X_hat[:, -1]
        return y_pred.cpu().detach().numpy()

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get the predictions of the structured model.
        This is done by multiplying the edge weights with the feature i.e. X @ W
        """
        # force convert to ndarray
        X = np.asarray(X)
        if self.standardize:
            X = self.ss_X.transform(X)

        # check that the model has been fit
        check_is_fitted(self, "graph_")

        y_pred = np.asarray(self._predict_from_parents(X))
        if self.standardize:
            y_pred = self.ss_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
        return y_pred

    def get_edges_to_node(self, name: str, data: str = "weight") -> pd.Series:
        """
        Get the edges to a specific node.
        Args:
            name: The name of the node which to get weights towards.

            data: The edge parameter to get. Default is "weight" to return
                  the adjacency matrix. Set to "mean_effect" to return the
                  signed average effect of features on the target node.

        Returns:
            The specified edge data.
        """
        check_is_fitted(self, "graph_")

        # build base data series
        edges = pd.Series(index=self._features)

        # iterate over all edges
        for (i, j, w) in self.graph_.edges(data=data):
            # for edges directed towards the "name" node
            if j == name:
                # insert the weight into the series
                edges[i] = w

        # fill edges not present in the iteration with zeros
        edges = edges.fillna(0)

        return edges

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Unsigned importances of the features wrt to the target.
        NOTE: these are used as the graph adjacency matrix.
        Returns:
            the L2 relationship between nodes.
        """
        return self.get_edges_to_node(self._target).values

    @property
    def coef_(self) -> np.ndarray:
        """
        Signed relationship between features and the target.
        For this linear case this equivalent to linear regression coefficients.
        Returns:
            the mean effect relationship between nodes.
        """
        return self.get_edges_to_node(self._target, data="mean_effect").values

    @property
    def intercept_(self) -> float:
        """ The bias term from the target node """
        bias = self.graph_.nodes[self._target]["bias"]
        return 0.0 if bias is None else float(bias)

    def plot_dag(self, enforce_dag: bool = False, filename: str = "./graph.png"):
        """ Util function used to plot the fitted graph """

        try:
            # pylint: disable=import-outside-toplevel
            from IPython.display import Image
        except ImportError as e:
            raise ImportError(
                "DAGRegressor.plot_dag method requires IPython installed."
            ) from e

        check_is_fitted(self, "graph_")

        graph = copy.deepcopy(self.graph_)
        if enforce_dag:
            graph.threshold_till_dag()

        # silence annoying plotting warning
        warnings.filterwarnings("ignore")

        viz = plot_structure(
            graph,
            graph_attributes={"scale": "0.5"},
            all_node_attributes=NODE_STYLE.WEAK,
            all_edge_attributes=EDGE_STYLE.WEAK,
        )
        viz.draw(filename)

        # reset warnings to always show
        warnings.simplefilter("always")
        return Image(filename)
