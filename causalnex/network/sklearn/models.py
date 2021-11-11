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
This module contains the implementation of ``BayesianNetworkClassifier``.

``BayesianNetworkClassifier`` is a class that supports learning CPDs from input data
and making predictions
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from causalnex.discretiser import Discretiser
from causalnex.discretiser.discretiser_strategy import (
    DecisionTreeSupervisedDiscretiserMethod,
    MDLPSupervisedDiscretiserMethod,
)
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel


class BayesianNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    A class that supports discretising features and probability fitting with scikit-learn syntax

    Example:
    ::
        # Dataset is from https://archive.ics.uci.edu/ml/datasets/student+performance
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.preprocessing import LabelEncoder
        >>> from causalnex.discretiser import Discretiser
        >>> from causalnex.network.sklearn import BayesianNetworkClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> data = pd.read_csv('student-por.csv', delimiter=';')
        >>> drop_col = ['school','sex','age','Mjob', 'Fjob','reason','guardian']
        >>> data = data.drop(columns=drop_col)
        >>> non_numeric_columns = list(data.select_dtypes(exclude=[np.number]).columns)
        >>> le = LabelEncoder()
        >>> for col in non_numeric_columns:
        >>>     data[col] = le.fit_transform(data[col])
        >>> data["G3"] = Discretiser(method="fixed",
                      numeric_split_points=[10]).transform(data["G3"].values)
        >>> label = data["G3"]
        >>> data.drop(['G3'], axis=1, inplace=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(
                        data, label, test_size=0.1, random_state=7)
        >>> edge_list = [('address', 'absences'),
                         ('Pstatus', 'famrel'),
                         ('Pstatus', 'absences'),
                         ('studytime', 'G1'),
                         ('G1', 'G2'),
                         ('failures', 'absences'),
                         ('failures', 'G1'),
                         ('schoolsup', 'G1'),
                         ('paid', 'absences'),
                         ('higher', 'famrel'),
                         ('higher', 'G1'),
                         ('internet', 'absences'),
                         ('G2', 'G3')]
        >>> discretiser_param = {
                'absences': {'method':"fixed",
                             'numeric_split_points':[1, 10]
                            },
                 'G1': {'method':"fixed",
                        'numeric_split_points':[10]
                       },
                 'G2': {'method':"fixed",
                        'numeric_split_points':[10]
                       }
                }
        >>> discretiser_alg = {'absences': 'unsupervised',
                              'G1': 'unsupervised',
                              'G2': 'unsupervised'
                             }
        >>> bayesian_param = {'method':"BayesianEstimator", 'bayes_prior':"K2"}
        >>> clf = BayesianNetworkClassifier(edge_list, discretiser_alg, discretiser_param, bayesian_param)
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)
        array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
               1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
               1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0])

    """

    def __init__(
        self,
        list_of_edges: List[Tuple[str]],
        discretiser_alg: Optional[Dict[str, str]] = None,
        discretiser_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        probability_kwargs: Dict[str, Dict[str, Any]] = None,
        return_prob: bool = False,
    ):
        """
        Args:
            list_of_edges (list): Edge list to construct graph
            - if True: return pandas dataframe with predicted probability for each state
            - if False: return a 1-D prediction array
            discretiser_alg (dict): Specify a supervised algorithm to discretise
            each feature in the data. Available options for the dictionary values
            are ['unsupervised', 'tree', 'mdlp']
            - if 'unsupervised': discretise the data using unsupervised method
            - if 'tree': discretise the data using decision tree method
            - if 'mdlp': discretise the data using MDLP method
            discretiser_kwargs (dict): Keyword arguments for discretisation methods.
            Only applicable if discretiser_alg is not None.
            probability_kwargs (dict): keyword arguments for the probability model
            return_prob (bool): choose to return predictions or probability

        Raises:
            KeyError: If an incorrect argument is passed
            ValueError: If the keys in discretiser_alg and discretiser_kwargs differ
        """

        probability_kwargs = probability_kwargs or {
            "method": "BayesianEstimator",
            "bayes_prior": "K2",
        }

        if discretiser_alg is None:
            logging.info(
                "No discretiser algorithm was given "
                "The training data will not be discretised"
            )
            discretiser_alg = {}

        discretiser_kwargs = discretiser_kwargs or {}

        self._validate_discretiser(discretiser_alg, discretiser_kwargs)

        self.list_of_edges = list_of_edges
        self.structure = StructureModel(self.list_of_edges)
        self.bn = BayesianNetwork(self.structure)
        self.return_prob = return_prob
        self.probability_kwargs = probability_kwargs
        self.discretiser_kwargs = discretiser_kwargs
        self.discretiser_alg = discretiser_alg
        self._target_name = None
        self._discretise_data = None

    @staticmethod
    def _validate_discretiser(discretiser_alg, discretiser_kwargs):
        unavailable_discretiser_algs = {
            k: v not in ["unsupervised", "tree", "mdlp"]
            for k, v in discretiser_alg.items()
        }

        if any(unavailable_discretiser_algs.values()):
            algs = {
                k: discretiser_alg[k]
                for k, v in unavailable_discretiser_algs.items()
                if v
            }
            raise KeyError(
                f"Some discretiser algorithms are not supported: `{algs}`. "
                "Please choose in ['unsupervised', 'tree', 'mdlp']"
            )

        if set(discretiser_kwargs) != set(discretiser_alg):
            raise ValueError(
                "discretiser_alg and discretiser_kwargs should have the same keys"
            )

    def _discretise_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to discretise input data using parameters in
        `discretiser_kwargs` and `discretiser_alg`.
        The splitting thresholds are extracted from the training data

        Args:
            X (pd.DataFrame): a dataframe to be discretised

        Returns:
            a discretised version of the input dataframe
        """

        X = X.copy()

        for col in self.discretiser_alg.keys():

            if self.discretiser_alg[col] == "unsupervised":

                if self.discretiser_kwargs[col]["method"] == "fixed":
                    X[col] = Discretiser(**self.discretiser_kwargs[col]).transform(
                        X[col].values
                    )
                else:
                    discretiser = Discretiser(**self.discretiser_kwargs[col]).fit(
                        self._discretise_data[col].values
                    )
                    X[col] = discretiser.transform(X[col].values)

            else:
                if self.discretiser_alg[col] == "tree":
                    discretiser = DecisionTreeSupervisedDiscretiserMethod(
                        mode="single", tree_params=self.discretiser_kwargs[col]
                    )

                elif self.discretiser_alg[col] == "mdlp":
                    discretiser = MDLPSupervisedDiscretiserMethod(
                        self.discretiser_kwargs[col]
                    )

                discretiser.fit(
                    dataframe=self._discretise_data,
                    feat_names=[col],
                    target=self._target_name,
                    target_continuous=False,
                )

                X[col] = discretiser.transform(X[[col]])

        return X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BayesianNetworkClassifier":
        """
        Build a Bayesian Network classifier from a set of training data.
        The method first discretises the feature using parameters in `discretiser_kwargs`
        and `discretiser_alg`. Next, it learns all the possible nodes that each feature
        can have. Finally, it learns the CPDs of the Bayesian Network.

        Args:
            X (pd.DataFrame): input training data
            y (pd.Series): categorical label for each row of X

        Returns:
            self
        """
        self._discretise_data = X.copy()
        self._discretise_data[y.name] = y
        self._target_name = y.name
        X = self._discretise_features(X)

        X[y.name] = y
        self.bn = self.bn.fit_node_states(X)
        self.bn = self.bn.fit_cpds(X, **self.probability_kwargs)

        return self

    def predict(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """
        Return predictions for the input data

        Args:
            X (pd.DataFrame): A dataframe of shape (num_row, num_features) for model to predict

        Returns:
            Model's prediction: A numpy array of shape (num_row,)

        Raises:
            ValueError: if CPDs are empty

        """
        if self.bn.cpds == {}:
            raise ValueError("No CPDs found. The model has not been fitted")

        X = self._discretise_features(X)

        if self.return_prob:
            pred = self.bn.predict_probability(X, self._target_name)
        else:
            pred = self.bn.predict(X, self._target_name).to_numpy().reshape(-1)

        return pred
