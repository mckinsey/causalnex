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

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from causalnex.structure.transformers import DynamicDataTransformer


class TestDynamicDataTransformer:
    def test_naming_nodes(self, data_dynotears_p3):
        """
        Nodes should have the format {var}_lag{l}
        """
        df = pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"])
        df_dyno = DynamicDataTransformer(p=3).fit_transform(df)

        pattern = re.compile(r"[abcde]_lag[0-3]")
        for node in df_dyno.columns:
            match = pattern.match(node)
            assert match
            assert match.group() == node

    def test_all_nodes_in_df(self, data_dynotears_p3):
        """
        Nodes should have the format {var}_lag{l}
        """
        df = pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"])
        df_dyno = DynamicDataTransformer(p=3).fit_transform(df)

        assert list(df_dyno.columns) == [
            el + "_lag" + str(i) for i in range(4) for el in ["a", "b", "c", "d", "e"]
        ]

    def test_incorrect_input_format(self):
        with pytest.raises(
            ValueError,
            match="Provided empty list of time_series."
            " At least one DataFrame must be provided",
        ):
            DynamicDataTransformer(p=3).fit_transform([])

        with pytest.raises(
            ValueError,
            match=r"All columns must have numeric data\. "
            r"Consider mapping the following columns to int: \['a'\]",
        ):
            DynamicDataTransformer(p=1).fit_transform(
                pd.DataFrame([["1"]], columns=["a"])
            )

        with pytest.raises(
            TypeError,
            match="Time series entries must be instances of `pd.DataFrame`",
        ):
            DynamicDataTransformer(p=1).fit_transform([np.array([1, 2])])

        with pytest.raises(
            ValueError,
            match="Index for dataframe must be provided in increasing order",
        ):
            df = pd.DataFrame(np.random.random([5, 5]), index=[3, 1, 2, 5, 0])
            DynamicDataTransformer(p=1).fit_transform(df)

        with pytest.raises(
            ValueError,
            match="All inputs must have the same columns and same types",
        ):
            df = pd.DataFrame(
                np.random.random([5, 5]),
                columns=["a", "b", "c", "d", "e"],
            )
            df_2 = pd.DataFrame(
                np.random.random([5, 5]),
                columns=["a", "b", "c", "d", "f"],
            )
            DynamicDataTransformer(p=1).fit_transform([df, df_2])

        with pytest.raises(
            ValueError,
            match="All inputs must have the same columns and same types",
        ):
            cols = ["a", "b", "c", "d", "e"]
            df = pd.DataFrame(np.random.random([5, 5]), columns=cols)
            df_2 = pd.DataFrame(np.random.random([5, 5]), columns=cols)
            df_2["a"] = df_2["a"].astype(int)
            DynamicDataTransformer(p=1).fit_transform([df, df_2])

        with pytest.raises(
            TypeError,
            match="Index must be integers",
        ):
            df = pd.DataFrame(np.random.random([5, 5]), index=[0, 1, 2, 3.0, 4])
            DynamicDataTransformer(p=1).fit_transform(df)

    def test_not_fitted_transform(self):
        """if transform called before fit: raise error"""
        with pytest.raises(
            NotFittedError,
            match=r"This DynamicDataTransformer is not fitted yet\."
            " Call `fit` before using this method",
        ):
            df = pd.DataFrame(np.random.random([5, 5]))
            DynamicDataTransformer(p=1).transform(df)

    def test_transform_wrong_input(self):
        """If transform df does not have all necessaty columns, raise error"""
        with pytest.raises(
            ValueError,
            match="We should provide all necessary columns in "
            r"the time series\. Columns not provided: \[2, 3\]",
        ):
            df = pd.DataFrame(np.random.random([5, 5]))
            ddt = DynamicDataTransformer(p=1).fit(df)
            ddt.transform(df.drop([2, 3], axis=1))

    def test_return_df_true_equivalent_to_false(self):
        """Check that the df from `return_df=true` is
        equivalent the result if `return_df=false`"""
        df = pd.DataFrame(np.random.random([50, 10]))
        df_dyno = DynamicDataTransformer(p=3).fit_transform(df, return_df=True)
        X, Xlags = DynamicDataTransformer(p=3).fit_transform(df, return_df=False)
        assert np.all(df_dyno.values[:, :10] == X)
        assert np.all(df_dyno.values[:, 10:] == Xlags)
