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
Collection of sklearn style transformers designed to assist with causal structure learning.
"""

from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class DynamicDataTransformer(BaseEstimator, TransformerMixin):
    """
    Format a time series dataframe or list of dataframes into the a format that matches the structure learned by
        `from_pandas_dynamic`. This is done to allow for bayesian network probability fitting.

        Example of utilisation:
        >>> ddt = DynamicDataTransformer(p=p).fit(time_series, return_df=False)
        >>> X, Xlags = ddt.transform(time_series)

        >>> ddt = DynamicDataTransformer(p=p).fit(time_series, return_df=True)
        >>> df = ddt.transform(time_series)
    """

    def __init__(self, p: int):
        """
        Initialise Transformer
        Args:
            p: Number of past interactions we allow the model to create. The state of a variable at time `t` is
                affected by the variables at the time stamp + the variables at `t-1`, `t-2`,... `t-p`.
        """
        self.p = p
        self.columns = None
        self.return_df = None

    def fit(
        self,
        time_series: Union[pd.DataFrame, List[pd.DataFrame]],
        return_df: bool = True,
    ) -> "DynamicDataTransformer":
        """
        Fits the time series. This consists memorizing:
            - Column names and column positions
            - whether a dataframe or a tuple of arrays should be returned by `transform` (details below)
        Args:
            time_series: pd.DataFrame or List of pd.DataFrame instances.
                If a list is provided each element of the list being an realisation of a time series
                (i.e. time series governed by the same processes)
                The columns of the data frame represent the variables in the model, and the *index represents
                the time index*.
                Successive events, therefore, must be indexed with one integer of difference between them too.

            return_df: Whether the `transform` method should return a pandas.DataFrame or a tuple with (X,Xlags)
                (Details on the documentation of the `transform` method)

        Returns:
            self

        """
        time_series = time_series if isinstance(time_series, list) else [time_series]
        self._check_input_from_pandas(time_series)
        self.columns = list(time_series[0].columns)
        self.return_df = return_df
        return self

    def transform(
        self, time_series: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """
        Applies transformation to format the dataframe properly
        Args:
            time_series: time_series: pd.DataFrame or List of pd.DataFrame instances. Details on `fit` documentation

        Returns:
            - If `self.return_df=True`, returns a pandas.DataFrame on the following format:

                A_lag0 B_lag0 C_lag0 ... A_lag1 B_lag1 C_lag1 ... A_lag`p` B_lag`p` C_lag`p`
                    X     X      X          X     X      X          X        X        X
                    X     X      X          X     X      X          X        X        X
                    X     X      X          X     X      X          X        X        X
            `lag0` denotes the current variable state and lag`k` denotes the states `k` time stamps in the past.

            - If `self.return_df=False`, returns a tuple of two numpy.ndarrayy: X and Xlags
                    X (np.ndarray): 2d input data, axis=1 is data columns, axis=0 is data rows.
                        Each column represents one variable,
                        and each row represents x(m,t) i.e. the mth time series at time t.
                    Xlags (np.ndarray):
                        Shifted data of X with lag orders stacking horizontally. Xlags=[shift(X,1)|...|shift(X,p)]
        Raises:
            NotFittedError: if `transform` called before `fit`
        """
        if self.columns is None:
            raise NotFittedError(
                "This DynamicDataTransformer is not fitted yet. "
                "Call `fit` before using this method"
            )

        time_series = time_series if isinstance(time_series, list) else [time_series]

        self._check_input_from_pandas(time_series)

        time_series = [t[self.columns] for t in time_series]
        ts_realisations = self._cut_dataframes_on_discontinuity_points(time_series)
        X, Xlags = self._convert_realisations_into_dynotears_format(
            ts_realisations, self.p
        )

        if self.return_df:
            res = self._concat_lags(X, Xlags)
            return res
        return X, Xlags

    def _concat_lags(self, X: np.ndarray, Xlags: np.ndarray) -> pd.DataFrame:
        df_x = pd.DataFrame(X, columns=[f"{col}_lag0" for col in self.columns])
        df_xlags = pd.DataFrame(
            Xlags,
            columns=[
                f"{col}_lag{l_}" for l_ in range(1, self.p + 1) for col in self.columns
            ],
        )
        return pd.concat([df_x, df_xlags], axis=1)

    def _check_input_from_pandas(self, time_series: List[pd.DataFrame]):
        """
        Check if the input of function `from_pandas_dynamic` is valid
        Args:
            time_series: List of pd.DataFrame instances.
                each element of the list being an realisation of a same time series

        Raises:
            ValueError: if empty list of time_series is provided
            ValueError: if dataframes contain non numeric data
            TypeError: if elements provided are not pandas dataframes
            ValueError: if dataframes contain different columns
            ValueError: if dataframes index is not in increasing order
            TypeError: if dataframes index are not index
        """
        if not time_series:
            raise ValueError(
                "Provided empty list of time_series. At least one DataFrame must be provided"
            )

        df = deepcopy(time_series[0])

        for t in time_series:
            if not isinstance(t, pd.DataFrame):
                raise TypeError(
                    "Time series entries must be instances of `pd.DataFrame`"
                )

            non_numeric_cols = t.select_dtypes(exclude="number").columns

            if not non_numeric_cols.empty:
                raise ValueError(
                    "All columns must have numeric data. Consider mapping the "
                    f"following columns to int: {list(non_numeric_cols)}"
                )

            if (not np.all(df.columns == t.columns)) or (
                not np.all(df.dtypes == t.dtypes)
            ):
                raise ValueError("All inputs must have the same columns and same types")

            if not np.all(t.index == t.index.sort_values()):
                raise ValueError(
                    "Index for dataframe must be provided in increasing order"
                )

            if not t.index.is_integer():
                raise TypeError("Index must be integers")

            if self.columns is not None:
                missing_cols = [c for c in self.columns if c not in t.columns]
                if missing_cols:
                    raise ValueError(
                        "We should provide all necessary columns in the time series. "
                        f"Columns not provided: {missing_cols}"
                    )

    @staticmethod
    def _cut_dataframes_on_discontinuity_points(
        time_series: List[pd.DataFrame],
    ) -> List[np.ndarray]:
        """
        Helper function for `from_pandas_dynamic`
        Receive a list of dataframes. For each dataframe, cut the points of discontinuity as two different dataframes.
        Discontinuities are determined by the indexes.

        For Example:
        If the following is a dataframe:
            index   variable_1  variable_2
            1       X           X
            2       X           X
            3       X           X
            4       X           X
            8       X           X               <- discontinuity point
            9       X           X
            10      X           X

        We cut this dataset in two:

            index   variable_1  variable_2
            1       X           X
            2       X           X
            3       X           X
            4       X           X

            and:
            index   variable_1  variable_2
            8       X           X
            9       X           X
            10      X           X


        Args:
            time_series: list of dataframes representing various realisations of a same time series

        Returns:
            List of np.ndarrays representing the pieces of the input datasets with no discontinuity

        """
        time_series_realisations = []
        for t in time_series:
            cutting_points = np.where(np.diff(t.index) > 1)[0]
            cutting_points = [0] + list(cutting_points + 1) + [len(t)]
            for start, end in zip(cutting_points[:-1], cutting_points[1:]):
                time_series_realisations.append(t.iloc[start:end, :].values)
        return time_series_realisations

    @staticmethod
    def _convert_realisations_into_dynotears_format(
        realisations: List[np.ndarray], p: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a list of realisations of a time series, convert it to the format received by the dynotears algorithm.
        Each realisation on `realisations` is a realisation of the time series,
        where the time dimension is represented by the rows.
            - The higher the row, the higher the time index
            - The data is complete, meaning that the difference between two time stamps is equal one
        Args:
            realisations: a list of realisations of a time series
            p: the number of lagged columns to create

        Returns:
            X and Y as in the SVAR model and DYNOTEARS paper. I.e. X being representing X(m,t) and Y the concatenated
            differences [X(m,t-1) | X(m,t-2) | ... | X(m,t-p)]
        """
        X = np.concatenate([realisation[p:] for realisation in realisations], axis=0)
        y_lag_list = [
            np.concatenate([realisation[p - i - 1 : -i - 1] for i in range(p)], axis=1)
            for realisation in realisations
        ]
        y_lag = np.concatenate(y_lag_list, axis=0)

        return X, y_lag
