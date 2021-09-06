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

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from causalnex.structure.pytorch.dist_type import (
    DistTypeBinary,
    DistTypeCategorical,
    DistTypeContinuous,
    DistTypeOrdinal,
    DistTypePoisson,
)
from causalnex.structure.pytorch.notears import from_numpy, from_pandas


class TestDistTypeClasses:
    @pytest.mark.parametrize(
        "dist_type", [DistTypeBinary, DistTypeContinuous, DistTypeCategorical]
    )
    def test_default_init(self, dist_type):
        idx = 1
        dt = dist_type(idx=idx)

        assert dt.idx == idx

    @pytest.mark.parametrize(
        "dist_type, X, X_hat",
        [
            (
                DistTypeContinuous,
                np.random.normal(size=(5, 2)),
                np.random.normal(size=(5, 2)),
            ),
            (
                DistTypeBinary,
                np.random.randint(2, size=(50, 2)),
                np.random.normal(size=(50, 2)),
            ),
            (
                DistTypeCategorical,
                np.random.randint(3, size=(50, 1)),
                np.random.normal(size=(50, 3)),
            ),
            (
                DistTypePoisson,
                np.random.randint(2, size=(5, 2)),
                np.random.normal(size=(5, 2)),
            ),
            (
                DistTypeOrdinal,
                np.random.randint(3, size=(50, 2)),
                np.random.normal(size=(50, 2)),
            ),
        ],
    )
    def test_loss(self, dist_type, X, X_hat):
        dist_types = [dist_type(idx=idx) for idx in np.arange(X.shape[1])]
        for dt in dist_types:
            X = dt.preprocess_X(X)
        X = torch.from_numpy(X).float()
        X_hat = torch.from_numpy(X_hat)

        loss = 0.0
        with torch.no_grad():
            for dt in dist_types:
                loss = loss + dt.loss(X, X_hat)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])

    @pytest.mark.parametrize(
        "dist_type, X",
        [
            (
                DistTypePoisson,
                np.random.normal(size=(100, 2)),
            ),
        ],
    )
    def test_preprocess_type_checks(self, dist_type, X):
        dist_types = [dist_type(idx=idx) for idx in np.arange(X.shape[1])]
        for dt in dist_types:
            with pytest.raises(
                ValueError,
                match=r"All data must be positive for Poisson\.",
            ):
                X = dt.preprocess_X(X)

    @pytest.mark.parametrize(
        "dist_type, X, X_hat",
        [
            (
                DistTypeContinuous,
                np.random.normal(size=(5, 2)),
                np.random.normal(size=(5, 2)),
            ),
            (
                DistTypeBinary,
                np.random.randint(2, size=(50, 2)),
                np.random.normal(size=(50, 2)),
            ),
            (
                DistTypeCategorical,
                np.random.randint(3, size=(50, 1)),
                np.random.normal(size=(50, 3)),
            ),
            (
                DistTypePoisson,
                np.random.randint(3, size=(5, 1)),
                np.random.normal(size=(5, 3)),
            ),
            (
                DistTypeOrdinal,
                np.random.randint(3, size=(50, 2)),
                np.random.normal(size=(50, 2)),
            ),
        ],
    )
    def test_inverse_link_functions(self, dist_type, X, X_hat):
        dist_types = [dist_type(idx=idx) for idx in np.arange(X.shape[1])]
        for dt in dist_types:
            X = dt.preprocess_X(X)
        X = torch.from_numpy(X).float()
        X_hat = torch.from_numpy(X_hat)

        with torch.no_grad():
            for dt in dist_types:
                pred = dt.inverse_link_function(X_hat)
                assert isinstance(pred, torch.Tensor)

    @pytest.mark.parametrize(
        "dist_type, X",
        [
            (
                DistTypeCategorical,
                np.random.randint(3, size=(50, 2)),
            ),
        ],
    )
    def test_preprocess_X_expanded_cols(self, dist_type, X):
        np.random.seed(42)
        dist_types = [dist_type(idx=idx) for idx in np.arange(X.shape[1])]
        for dt in dist_types:
            X = dt.preprocess_X(X)

        # check size of column expansion
        assert X.shape[1] == 6

        # check that the correct indecies are pulled out
        assert dist_types[0].idx_group == [0, 2, 3]
        assert dist_types[1].idx_group == [1, 4, 5]
        # test that the expanded get_columns works
        assert np.array_equal(
            dist_types[0].get_columns(X), X[:, dist_types[0].idx_group]
        )
        assert np.array_equal(
            dist_types[1].get_columns(X), X[:, dist_types[1].idx_group]
        )

    @pytest.mark.parametrize(
        "dist_type, X",
        [
            (
                DistTypeCategorical,
                np.random.randint(3, size=(50, 2)),
            ),
        ],
    )
    def test_expanded_cols_recovery(self, dist_type, X):
        np.random.seed(42)
        X_orig = X
        dist_types = [dist_type(idx=idx) for idx in np.arange(X.shape[1])]
        for dt in dist_types:
            X = dt.preprocess_X(X)

        # check that original X can be recovered from expanded cols
        for dt in dist_types:
            X_recovered = np.argmax(X[:, dt.idx_group], axis=1)
            assert (X_recovered == X_orig[:, dt.idx]).sum() == X.shape[0]

    @pytest.mark.parametrize(
        "dist_type, X, tabu_nodes, tabu_nodes_updated",
        [
            (
                DistTypeContinuous,
                np.random.normal(size=(50, 2)),
                [0],
                [0],
            ),
            (
                DistTypeBinary,
                np.random.randint(2, size=(50, 2)),
                [0],
                [0],
            ),
            (
                DistTypeCategorical,
                np.random.randint(3, size=(50, 1)),
                [0],
                [0, 1, 2],
            ),
            (
                DistTypePoisson,
                np.random.randint(3, size=(5, 1)),
                [0],
                [0],
            ),
            (
                DistTypeOrdinal,
                np.random.randint(3, size=(5, 1)),
                [0],
                [0],
            ),
        ],
    )
    def test_preprocess_tabu_nodes(self, dist_type, X, tabu_nodes, tabu_nodes_updated):
        dist_types = [dist_type(idx=idx) for idx in np.arange(X.shape[1])]
        for dt in dist_types:
            X = dt.preprocess_X(X)
            tabu_nodes = dt.preprocess_tabu_nodes(tabu_nodes)

        assert tabu_nodes == tabu_nodes_updated

    @pytest.mark.parametrize(
        "dist_type, X, tabu_edges, tabu_edges_updated",
        [
            (
                DistTypeContinuous,
                np.random.normal(size=(50, 2)),
                [(0, 1)],
                [(0, 1)],
            ),
            (
                DistTypeBinary,
                np.random.randint(2, size=(50, 2)),
                [(0, 1)],
                [(0, 1)],
            ),
            (
                DistTypeCategorical,
                np.random.randint(2, size=(50, 2)),
                [],
                [(0, 2), (2, 0), (1, 3), (3, 1)],
            ),
            (
                DistTypeCategorical,
                np.random.randint(2, size=(50, 2)),
                [(0, 1)],
                [(0, 2), (2, 0), (1, 3), (3, 1), (0, 1), (2, 1), (0, 3), (2, 3)],
            ),
            (
                DistTypePoisson,
                np.random.randint(3, size=(50, 2)),
                [(0, 1)],
                [(0, 1)],
            ),
            (
                DistTypeOrdinal,
                np.random.randint(3, size=(50, 2)),
                [(0, 1)],
                [(0, 1)],
            ),
        ],
    )
    def test_preprocess_tabu_edges(self, dist_type, X, tabu_edges, tabu_edges_updated):
        np.random.seed(42)
        dist_types = [dist_type(idx=idx) for idx in np.arange(X.shape[1])]
        for dt in dist_types:
            X = dt.preprocess_X(X)
            tabu_edges = dt.preprocess_tabu_edges(tabu_edges)

        # check that all expected edges are generated
        for tabu_edge in tabu_edges:
            assert tabu_edge in tabu_edges_updated
        # assert lengths are the same
        assert len(tabu_edges_updated) == len(tabu_edges)
        # assert uniqueness
        assert len(set(tabu_edges)) == len(tabu_edges)

    @pytest.mark.parametrize(
        "dist_type, X",
        [
            (
                DistTypeContinuous,
                np.random.normal(size=(50, 2)),
            ),
            (
                DistTypeBinary,
                np.random.randint(2, size=(50, 2)),
            ),
            (
                DistTypeCategorical,
                np.random.randint(3, size=(50, 1)),
            ),
            (
                DistTypePoisson,
                np.random.randint(3, size=(50, 2)),
            ),
            (
                DistTypeOrdinal,
                np.random.randint(3, size=(50, 2)),
            ),
        ],
    )
    def test_update_idx_col(self, dist_type, X):
        """ Test to ensure that first column is always preserved """
        idx_col_original = {i: f"{i}" for i in range(X.shape[1])}
        dist_types = [dist_type(idx=idx) for idx in np.arange(X.shape[1])]
        for dt in dist_types:
            X = dt.preprocess_X(X)
        idx_col_new = copy.deepcopy(idx_col_original)
        for dt in dist_types:
            idx_col_new = dt.update_idx_col(idx_col_new)

        for idx in idx_col_original.keys():
            # ensure that all original columns exist
            assert idx_col_original[idx] == idx_col_new[idx]


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
            (np.random.randint(3, size=(50, 3)), {0: "cat", 1: "cat", 2: "cat"}),
            (
                np.hstack(
                    [
                        np.random.normal(size=(50, 2)),
                        np.random.randint(2, size=(50, 2)),
                        np.random.randint(3, size=(50, 2)),
                    ]
                ),
                {0: "cont", 1: "cont", 2: "bin", 3: "bin", 4: "cat", 5: "cat"},
            ),
            (np.random.randint(3, size=(50, 3)), {0: "poiss", 1: "poiss", 2: "poiss"}),
            (np.random.randint(3, size=(50, 3)), {0: "ord", 1: "ord", 2: "ord"}),
        ],
    )
    def test_numpy_notears_with_schema(self, X, schema):
        np.random.seed(42)
        from_numpy(X, schema)

    @pytest.mark.parametrize(
        "X, schema",
        [
            (np.random.normal(size=(10, 3)), {0: "cont", 1: "cont", 2: "cont"}),
            (np.random.randint(2, size=(10, 3)), {0: "bin", 1: "bin", 2: "bin"}),
            (np.random.randint(3, size=(50, 3)), {0: "cat", 1: "cat", 2: "cat"}),
            (
                np.hstack(
                    [
                        np.random.normal(size=(50, 2)),
                        np.random.randint(2, size=(50, 2)),
                        np.random.randint(3, size=(50, 2)),
                    ]
                ),
                {0: "cont", 1: "cont", 2: "bin", 3: "bin", 4: "cat", 5: "cat"},
            ),
            (np.random.randint(3, size=(50, 3)), {0: "poiss", 1: "poiss", 2: "poiss"}),
            (np.random.randint(3, size=(50, 3)), {0: "ord", 1: "ord", 2: "ord"}),
        ],
    )
    def test_pandas_notears_with_schema(self, X, schema):
        X = pd.DataFrame(X)
        from_pandas(X, schema)
