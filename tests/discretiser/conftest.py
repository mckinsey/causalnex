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

from typing import List

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from causalnex.discretiser.abstract_discretiser import (
    AbstractSupervisedDiscretiserMethod,
)


class Dummy(AbstractSupervisedDiscretiserMethod):
    def fit(
        self,
        feat_names: List[str],
        target: str,
        dataframe: pd.DataFrame,
        target_continuous: bool,
    ):
        raise NotImplementedError("This is not implemented")

    def learn(self, get_iris_data):
        super().fit(
            feat_names=["petal width (cm)"],
            dataframe=get_iris_data,
            target_continuous=False,
            target="target",
        )

    def learn_transform(self, get_iris_data):
        super().fit_transform(
            feat_names=["petal width (cm)"],
            dataframe=get_iris_data,
            target_continuous=False,
            target="target",
        )


@pytest.fixture
def get_dummy_class():
    return Dummy()


@pytest.fixture
def get_iris_data():
    iris = load_iris()
    X, y = iris["data"], iris["target"]
    names = iris["feature_names"]
    df = pd.DataFrame(X, columns=names)
    df["target"] = y
    return df


@pytest.fixture
def get_diabete_data():
    diabetes = load_diabetes()
    X, y = diabetes["data"], diabetes["target"]
    names = diabetes["feature_names"]
    df = pd.DataFrame(X, columns=names)
    df["target"] = y
    return df


@pytest.fixture
def categorical_data(get_iris_data):
    return get_iris_data[["petal width (cm)", "target"]]


@pytest.fixture
def continuous_data(get_diabete_data):
    return get_diabete_data[["s6", "target"]]
