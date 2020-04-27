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
import pytest

from causalnex.structure.dynotears import from_numpy_dynamic


class TestFromNumpyDynotears:
    """Test behaviour of the learn_dynamic_structure of dynotear"""

    def test_empty_data_raises_error(self):
        """
        Providing an empty data set should result in a Value Error explaining that data must not be empty.
        This error is useful to catch and handle gracefully, because otherwise the user would experience
        misleading division by zero, or unpacking errors.
        """

        with pytest.raises(
            ValueError, match="Input data X is empty, cannot learn any structure"
        ):
            from_numpy_dynamic(np.empty([0, 5]), np.zeros([5, 5]))
        with pytest.raises(
            ValueError, match="Input data Xlags is empty, cannot learn any structure"
        ):
            from_numpy_dynamic(np.zeros([5, 5]), np.empty([0, 5]))

    def test_nrows_data_mismatch_raises_error(self):
        """
        Providing input data and lagged data with different number of rows should result in a Value Error.
        """

        with pytest.raises(
            ValueError, match="Input data X and Xlags must have the same number of rows"
        ):
            from_numpy_dynamic(np.zeros([5, 5]), np.zeros([6, 5]))

    def test_ncols_lagged_data_not_multiple_raises_error(self):
        """
        Number of columns of lagged data is not a multiple of those of input data should result in a Value Error.
        """

        with pytest.raises(
            ValueError,
            match="Number of columns of Xlags must be a multiple of number of columns of X",
        ):
            from_numpy_dynamic(np.zeros([5, 5]), np.zeros([5, 6]))

    def test_single_iter_gets_converged_fail_warnings(self, train_data_num_temporal):
        """
        With a single iteration on this dataset, learn_structure fails to converge and should give warnings.
        """

        with pytest.warns(
            UserWarning, match="Failed to converge. Consider increasing max_iter."
        ):
            from_numpy_dynamic(
                train_data_num_temporal[1:], train_data_num_temporal[:-1], max_iter=1
            )

    def test_expected_structure_learned(self, train_data_num_temporal):
        """
        Given a small data set, the learned weights should be deterministic and within the expected range
        """

        sm = from_numpy_dynamic(
            train_data_num_temporal[1:], train_data_num_temporal[:-1], w_threshold=0.3
        )
        assert list(sm.edges) == [
            ("0_lag0", "3_lag0"),
            ("1_lag0", "4_lag0"),
            ("3_lag0", "1_lag0"),
            ("1_lag1", "4_lag0"),
        ]

    def test_tabu_parents(self, train_data_num_temporal):
        """
        If tabu relationships are set, the corresponding edges must not exist
        """

        sm = from_numpy_dynamic(
            train_data_num_temporal[1:],
            train_data_num_temporal[:-1],
            tabu_parent_nodes=[1],
        )
        assert not ([el for el in sm.edges if el[0] == "1_lag0"])
        assert not (
            [el for el in sm.edges if isinstance(el[0], str) and ("1_lag" in el[0])]
        )

    def test_tabu_children(self, train_data_num_temporal):
        """
        If tabu relationships are set, the corresponding edges must not exist
        """
        sm = from_numpy_dynamic(
            train_data_num_temporal[1:],
            train_data_num_temporal[:-1],
            tabu_child_nodes=[4],
        )
        assert not ([el for el in sm.edges if el[1] == 4])
        assert not (
            [el for el in sm.edges if isinstance(el[1], str) and ("_lag4" in el[1])]
        )

        sm = from_numpy_dynamic(
            train_data_num_temporal[1:],
            train_data_num_temporal[:-1],
            tabu_child_nodes=[1],
        )
        assert not ([el for el in sm.edges if el[1] == 1])
        assert not (
            [el for el in sm.edges if isinstance(el[1], str) and ("_lag1" in el[1])]
        )

    def test_tabu_edges(self, train_data_num_temporal):
        sm = from_numpy_dynamic(
            train_data_num_temporal[1:],
            train_data_num_temporal[:-1],
            tabu_edges=[(0, 2, 4), (0, 0, 3), (1, 1, 4), (1, 3, 4)],
        )
        assert ("2_lag0", "4_lag0") not in sm.edges
        assert ("0_lag0", "3_lag0") not in sm.edges
        assert ("1_lag1", "4_lag0") not in sm.edges
        assert ("3_lag1", "4_lag0") not in sm.edges
