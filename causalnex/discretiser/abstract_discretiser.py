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
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class AbstractSupervisedDiscretiserMethod(BaseEstimator, ABC):
    """
    Base class for advanced discretisation methods

    """

    def __init__(self):
        self.map_thresholds = {}
        self.feat_names = None

    @abstractmethod
    def fit(
        self,
        feat_names: List[str],
        target: str,
        dataframe: pd.DataFrame,
        target_continuous: bool,
    ):
        """
        Discretise the features in `feat_names` in such a way that maximises the prediction of `target`.

        Args:
            feat_names (List[str]): List of feature names to be discretised.
            target (str): Name of the target variable - the node that adjusts how `feat_names` will be discretised
            dataframe: The full dataset prior to discretisation.
            target_continuous (bool): Boolean indicates if target variable is continuous
        Raises:
            NotImplementedError: AbstractSupervisedDiscretiserMethod should not be called directly

        """
        raise NotImplementedError("The method is not implemented")

    def _transform_one_column(self, dataframe_one_column: pd.DataFrame) -> np.array:
        """
        Given one "original" feature (continuous), discretise it.

        Args:
            dataframe_one_column: dataframe with a single continuous feature, to be transformed into discrete
        Returns:
            Discrete feature, as an np.array of shape (len(df),)
        """
        cols = list(dataframe_one_column.columns)
        if cols[0] in self.map_thresholds:
            split_points = self.map_thresholds[cols[0]]
            return np.digitize(dataframe_one_column.values.reshape(-1), split_points)

        if cols[0] not in self.feat_names:
            logging.warning(
                "%s is not in feat_names. The column is left unchanged", cols[0]
            )
        return dataframe_one_column.values.reshape(-1)

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Given one "original" dataframe, discretise it.

        Args:
            data: dataframe with continuous features, to be transformed into discrete
        Returns:
            discretised version of the input data
        """
        outputs = {}
        for col in data.columns:
            outputs[col] = self._transform_one_column(data[[col]])

        transformed_df = pd.DataFrame.from_dict(outputs)

        return transformed_df.set_index(data.index)

    def fit_transform(self, *args, **kwargs):
        """
        Raises:
            NotImplementedError: fit_transform is not implemented
        """
        raise NotImplementedError(
            "fit_transform is not implemented. Please use .fit() and .transform() separately"
        )
