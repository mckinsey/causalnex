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

import numpy as np
import pandas as pd
import pytest
import torch

from causalnex.structure.pytorch.dist_type import DistTypeBinary, DistTypeContinuous
from causalnex.structure.pytorch.notears import from_numpy, from_pandas


class TestDistTypeClasses:
    @pytest.mark.parametrize("dist_type", [DistTypeBinary, DistTypeContinuous])
    def test_default_init(self, dist_type):
        idx = 1
        dt = dist_type(idx=idx)

        assert dt.idx == idx

    @pytest.mark.parametrize(
        "dist_type, X, X_hat",
        [
            (
                DistTypeContinuous,
                torch.from_numpy(np.random.normal(size=(5, 2))),
                torch.from_numpy(np.random.normal(size=(5, 2))),
            ),
            (
                DistTypeBinary,
                torch.from_numpy(np.random.randint(2, size=(5, 2))).float(),
                torch.from_numpy(np.random.randint(2, size=(5, 2))).float(),
            ),
        ],
    )
    def test_loss(self, dist_type, X, X_hat):
        dist_types = [dist_type(idx=idx) for idx in np.arange(X.shape[1])]
        loss = 0.0
        with torch.no_grad():
            for dt in dist_types:
                loss = loss + dt.loss(X, X_hat)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])


class TestDistTypeNotears:
    def test_schema_mismatch_error(self):
        X = np.ones(shape=(10, 2))
        schema = {0: "cont", 1: "cont", 2: "cont"}
        with pytest.raises(ValueError):
            from_numpy(X, schema)

    @pytest.mark.parametrize(
        "X, schema",
        [
            (np.random.normal(size=(10, 3)), {0: "cont", 1: "cont", 2: "cont"}),
            (np.random.randint(2, size=(10, 3)), {0: "bin", 1: "bin", 2: "bin"}),
            (
                np.hstack(
                    [np.random.normal(size=(5, 2)), np.random.randint(2, size=(5, 2))]
                ),
                {0: "cont", 1: "cont", 2: "bin", 3: "bin"},
            ),
        ],
    )
    def test_numpy_notears_with_schema(self, X, schema):
        from_numpy(X, schema)

    @pytest.mark.parametrize(
        "X, schema",
        [
            (np.random.normal(size=(10, 3)), {0: "cont", 1: "cont", 2: "cont"}),
            (np.random.randint(2, size=(10, 3)), {0: "bin", 1: "bin", 2: "bin"}),
            (
                np.hstack(
                    [np.random.normal(size=(5, 2)), np.random.randint(2, size=(5, 2))]
                ),
                {0: "cont", 1: "cont", 2: "bin", 3: "bin"},
            ),
        ],
    )
    def test_pandas_notears_with_schema(self, X, schema):
        X = pd.DataFrame(X)
        from_pandas(X, schema)
