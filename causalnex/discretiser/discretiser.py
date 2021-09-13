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

from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Discretiser(BaseEstimator, TransformerMixin):
    """Allows the discretisation of numeric data.

    Example:
    ::
        >>> import causalnex
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({'Age': [12, 13, 18, 19, 22, 60]})
        >>>
        >>> from causalnex.discretiser import Discretiser
        >>> df["Transformed_Age_1"] = Discretiser(method="fixed",
        >>> numeric_split_points=[11,18,50]).transform(df["Age"])
        >>> df.to_dict()
        {'Age': {0: 7, 1: 12, 2: 13, 3: 18, 4: 19, 5: 22, 6: 60},
        'Transformed_Age': {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3}}
    """

    def __init__(
        self,
        method: str = "uniform",
        num_buckets: int = None,
        outlier_percentile: float = None,
        numeric_split_points: List[float] = None,
        percentile_split_points: List[float] = None,
    ):
        """
        Creates a new Discretiser, that provides fit, fit_transform, and transform function to discretise data.

        Args:
            method (str): can be one of:
             - uniform: discretise data into uniformly spaced buckets. Note, complete uniformity
             cannot be guaranteed under all circumstances, for example, if 5 data points are to split
             into 2 buckets, then one will contain 2 points, and the other will contain 3.
             Provide num_buckets.
             - quantile: discretise data according to the distribution of values. For example, providing
             num_buckets=4 will discretise data into 4 buckets, [0-25th, 25th-50th, 50th-75th, 75th-100th]
             percentiles. Provide num_buckets.
             - outlier: discretise data into 3 buckets - [low_outliers, normal, high_outliers] based on
             outliers being below outlier_percentile, or above 1-outlier_percentile. Provide outlier_percentile.
             - fixed: discretise according to pre-defined split points. Provide numeric_split_points
             - percentiles: discretise data according to the distribution of percentiles values.
             Provide percentile_split_points.
            num_buckets: (int): used by method=uniform and method=quantile.
            outlier_percentile: used by method=outlier.
            numeric_split_points: used by method=fixed. to split such that values below 10 go into bucket 0,
            10 to 20 go into bucket 1, and above 20 go into bucket 2, provide [10, 21]. Note that split_point
            values are non-inclusive.
            percentile_split_points: used by method=percentiles. to split such that values below 10th percentiles
            go into bucket 0, 10th to below 75th percentiles go into bucket 1, and 75th percentiles and above go into
            bucket 2, provide [0.1, 0.75].

        Raises:
            ValueError: If an incorrect argument is passed.
        """

        self.numeric_split_points = []

        self.method = method
        self.num_buckets = num_buckets
        self.outlier_percentile = outlier_percentile
        self.numeric_split_points = numeric_split_points
        self.percentile_split_points = percentile_split_points

        allowed_methods = ["uniform", "quantile", "outlier", "fixed", "percentiles"]

        if self.method not in allowed_methods:
            raise ValueError(
                f"{self.method} is not a recognised method. "
                f"Use one of: {' '.join(allowed_methods)}"
            )
        if self.method in {"uniform", "quantile"} and num_buckets is None:
            raise ValueError(f"{self.method} method expects num_buckets")

        if self.method == "outlier" and outlier_percentile is None:
            raise ValueError(f"{self.method} method expects outlier_percentile")

        if outlier_percentile is not None and not 0 <= outlier_percentile < 0.5:
            raise ValueError("outlier_percentile must be between 0 and 0.5")

        if self.method == "fixed" and numeric_split_points is None:
            raise ValueError(f"{self.method} method expects numeric_split_points")

        if (
            numeric_split_points is not None
            and sorted(numeric_split_points) != numeric_split_points
        ):
            raise ValueError("numeric_split_points must be monotonically increasing")

        if self.method == "percentiles" and percentile_split_points is None:
            raise ValueError(f"{self.method} method expects percentile_split_points")

        if percentile_split_points is not None and not all(
            0 <= p <= 1 for p in percentile_split_points
        ):
            raise ValueError("percentile_split_points must be between 0 and 1")

        if (
            percentile_split_points is not None
            and sorted(percentile_split_points) != percentile_split_points
        ):
            raise ValueError("percentile_split_points must be monotonically increasing")

        if self.method == "fixed":
            self.numeric_split_points = numeric_split_points

    def fit(self, data: np.ndarray) -> "Discretiser":
        """
        Fit where split points are based on the input data.

        Args:
            data (np.ndarray): values used to learn where split points exist.

        Returns:
            self

        Raises:
            RuntimeError: If an attempt to fit fixed numeric_split_points is made.
        """

        x = data.flatten()
        x.sort()

        if self.method == "uniform":
            bucket_width = (np.max(x) - np.min(x)) / self.num_buckets
            self.numeric_split_points = [
                np.min(x) + bucket_width * (n + 1) for n in range(self.num_buckets - 1)
            ]

        elif self.method == "quantile":
            bucket_width = 1.0 / self.num_buckets
            quantiles = [bucket_width * (n + 1) for n in range(self.num_buckets - 1)]
            self.numeric_split_points = np.quantile(x, quantiles)

        elif self.method == "outlier":
            self.numeric_split_points = np.quantile(
                x, [self.outlier_percentile, 1 - self.outlier_percentile]
            )

        elif self.method == "percentiles":
            percentiles = [p * 100 for p in self.percentile_split_points]
            self.numeric_split_points = np.percentile(x, percentiles)

        else:
            raise RuntimeError("cannot call fit using method=fixed")

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the input data into discretised digits, based on the numeric_split_points that were either
        learned through using fit(), or from initialisation if method="fixed".

        Args:
            data (np.ndarray): values that will be transformed into discretised digits.

        Returns:
            input data transformed into discretised digits.
        """

        return np.digitize(data, self.numeric_split_points, right=False)
