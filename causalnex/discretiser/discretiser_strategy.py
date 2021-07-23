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
"""Tools to help discretise data."""

import logging
from copy import deepcopy
from typing import Any, Dict, List

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from causalnex.discretiser.abstract_discretiser import (
    AbstractSupervisedDiscretiserMethod,
)
from causalnex.utils.decision_tree_tools import extract_thresholds_from_dtree

try:
    from mdlp.discretization import MDLP
except ImportError:
    MDLP = None
    logging.warning("MDLP was not imported successfully")


class DecisionTreeSupervisedDiscretiserMethod(AbstractSupervisedDiscretiserMethod):
    """Allows the discretisation of continuous features based on the split thresholds of either
    sklearn's DecisionTreeRegressor or DecisionTreeClassifier.
    DecisionTreeSupervisedDiscretiserMethod is inhereited from AbstractSupervisedDiscretiserMethod.
    When instantiated, we have an object with .fit method to learn discretisation thresholds from data
    and .transform method to process the input.


    Example:
    ::
        >>> import pandas as pd
        >>> import numpy as np
        >>> from causalnex.discretiser.discretiser_strategy import DecisionTreeSupervisedDiscretiserMethod
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X, y = iris["data"], iris["target"]
        >>> names = iris["feature_names"]
        >>> data = pd.DataFrame(X, columns=names)
        >>> data["target"] = y
        >>> dt_multi = DecisionTreeSupervisedDiscretiserMethod(
        >>>     mode="multi", tree_params={"max_depth": 3, "random_state": 2020}
        >>> )
        >>> tree_discretiser = dt_multi.fit(
        >>>     feat_names=[
        >>>         "sepal length (cm)",
        >>>         "sepal width (cm)",
        >>>         "petal length (cm)",
        >>>         "petal width (cm)",
        >>>     ],
        >>>     dataframe=data,
        >>>     target="target",
        >>>     target_continuous=False,
        >>> )
        >>> discretised_data = tree_discretiser.transform(data[["petal width (cm)"]])
        >>> discretised_data.values.ravel()
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
           2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    """

    def __init__(
        self,
        mode: str = "single",
        split_unselected_feat: bool = False,
        tree_params: Dict[str, Any] = None,
    ):
        """
        This Discretiser Method uses Decision Trees to predict the target.
        The cutting points on the the Decision Tree becomes the chosen discretisation thresholds

        If the target is a continuous variable, we fit a `DecisionTreeRegressor` to discretise the data.
        Otherwise, we fit a Classifier.

        Args:
            max_depth (int): maximum depth of the decision tree.
            mode (str): Either 'single' or 'multi'.
            - if single, Train a univariate decision tree for each continuous variable being discretised.
                The splitting points of the decision tree become discretiser fixed points
            - if multi, Train a decision tree over all the variables passed.
                The splitting points of each variable used in the Decision tree become the thresholds for discretisation
            split_unselected_feat (bool): only applicable if self.mode = 'multi'.
            - if True, features not selected by the decision tree will be discretised using 'single' mode
            with the same tree parameters
            - if False, features not selected by the decision tree will be left unchanged
            tree_params: keyword arguments, which are parameters
            used for `sklearn.tree.DecisionTreeClassifier`/`sklearn.tree.DecisionTreeRegressor`
        Raises:
            KeyError: if an incorrect argument is passed
        """

        super().__init__()
        tree_params = tree_params or {"max_depth": 2}
        self.tree_params = tree_params
        self.feat_names = None
        self.map_thresholds = {}
        if mode not in ["single", "multi"]:
            raise KeyError(
                f"mode, `{mode}` is not valid, please choose in ['single', 'multi']"
            )
        self.mode = mode
        self.split_unselected_feat = split_unselected_feat

    def fit(
        self,
        feat_names: List[str],
        target: str,
        dataframe: pd.DataFrame,
        target_continuous: bool,
    ) -> "DecisionTreeSupervisedDiscretiserMethod":
        """
        The fit method allows DecisionTrees to learn split thresholds from the input data

        Args:
            feat_names (List[str]): a list of feature to be discretised
            target (str): name of variable that is going to be used a target for the decision tree
            dataframe (pd.DataFrame): pandas dataframe of input data
            target_continuous (bool): a boolean that indicates if the target variable is continuous

        Returns:
            self: DecisionTreeSupervisedDiscretiserMethod object with learned split thresholds from the decision tree
        """
        dtree = (
            DecisionTreeRegressor(**self.tree_params)
            if target_continuous
            else DecisionTreeClassifier(**self.tree_params)
        )
        self.feat_names = feat_names
        self.map_thresholds = {}

        if self.mode == "single":
            for feat in feat_names:
                dtree = deepcopy(dtree)

                dtree.fit(dataframe[[feat]], dataframe[[target]])
                thresholds = extract_thresholds_from_dtree(dtree, 1)[0]
                self.map_thresholds[feat] = thresholds

        elif self.mode == "multi":
            dtree = deepcopy(dtree)
            dtree.fit(dataframe[feat_names], dataframe[[target]])
            threshold_list = extract_thresholds_from_dtree(dtree, len(feat_names))

            for feat, threshold in zip(feat_names, threshold_list):
                self.map_thresholds[feat] = threshold

            if self.split_unselected_feat:
                for feat, thres in self.map_thresholds.items():
                    if thres.size == 0:
                        dtree = deepcopy(dtree)
                        dtree.fit(dataframe[[feat]], dataframe[[target]])
                        thresholds = extract_thresholds_from_dtree(dtree, 1)[0]
                        self.map_thresholds[feat] = thresholds

            else:
                no_use = []
                for feat in list(self.map_thresholds.keys()):
                    if self.map_thresholds[feat].size == 0:
                        no_use.append(feat)
                        del self.map_thresholds[feat]
                if len(no_use) > 0:
                    logging.warning(
                        "%s not selected by the decision tree. No discretisation thresholds were learned. "
                        "Consider setting split_unselected_feat = True or discretise them using single mode",
                        no_use,
                    )

        return self


class MDLPSupervisedDiscretiserMethod(AbstractSupervisedDiscretiserMethod):
    """Allows discretisation of continuous features using mdlp algorithm

    Example:
    ::
        >>> import pandas as pd
        >>> import numpy as np
        >>> from causalnex.discretiser.discretiser_strategy import MDLPSupervisedDiscretiserMethod
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X, y = iris["data"], iris["target"]
        >>> names = iris["feature_names"]
        >>> data = pd.DataFrame(X, columns=names)
        >>> data["target"] = y
        >>> discretiser = MDLPSupervisedDiscretiserMethod(
        >>>     {"min_depth": 0, "random_state": 2020, "min_split": 1e-3, "dtype": int}
        >>> )
        >>> discretiser.fit(
        >>>     feat_names=["sepal length (cm)"],
        >>>     dataframe=data,
        >>>     target="target",
        >>>     target_continuous=False,
        >>> )
        >>> discretised_data = discretiser.transform(data[["sepal length (cm)"]])
        >>> discretised_data.values.ravel()

        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 2, 1, 2, 0, 2, 0, 0, 2, 2, 2, 1, 2,
               1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 2, 2, 2,
               1, 1, 1, 2, 1, 0, 1, 1, 1, 2, 0, 1, 2, 1, 2, 2, 2, 2, 0, 2, 2, 2,
               2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2])

    """

    def __init__(
        self,
        mdlp_args: Dict[str, Any] = None,
    ):
        """
        This method of discretisation applies MDLP to discretise the data

        Args:
            min_depth: The minimum depth of the interval splitting.
            min_split: The minmum size to split a bin
            dtype: The type of the array returned by the `transform()` method
            **dlp_args: keyword arguments, which are parameters used for `mdlp.discretization.MDLP`
        Raises:
            ImportError: if mdlp-discretization is not installed successfully
        """
        super().__init__()
        mdlp_args = mdlp_args or {"min_depth": 0, "min_split": 1e-3, "dtype": int}
        self.mdlp_args = mdlp_args
        self.feat_names = None
        self.map_feat_transformer = {}
        if MDLP is None:
            raise ImportError(
                "mdlp-discretisation was not installed and imported successfully"
            )
        self.mdlp = MDLP(**mdlp_args)

    def fit(
        self,
        feat_names: List[str],
        target: str,
        dataframe: pd.DataFrame,
        target_continuous: bool,
    ) -> "MDLPSupervisedDiscretiserMethod":
        """
        The fit method allows MDLP to learn split thresholds from the input data.
        The target feature cannot be continuous

        Args:
            feat_names (List[str]): a list of feature to be discretised
            target (str): name of the variable that is going to be used a target for MDLP
            dataframe (pd.DataFrame): pandas dataframe of input data
            target_continuous (bool): boolean that indicates if target variable is continuous.

        Returns:
            self: MDLPSupervisedDiscretiserMethod object with learned split thresholds from mdlp algorithm

        Raises:
            ValueError: if the target is continuous
        """
        self.feat_names = feat_names
        self.map_feat_transformer = {}
        if target_continuous:
            raise ValueError(
                "Target variable should not be continuous when using MDLP."
            )

        for feat in feat_names:
            mdlp = deepcopy(self.mdlp)

            mdlp.fit(dataframe[[feat]], dataframe[[target]])
            self.map_thresholds[feat] = mdlp.cut_points_[0]

        return self
