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

import networkx as nx
import numpy as np
import pandas as pd
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

    def test_single_iter_gets_converged_fail_warnings(self, data_dynotears_p1):
        """
        With a single iteration on this dataset, learn_structure fails to converge and should give warnings.
        """

        with pytest.warns(
            UserWarning, match="Failed to converge. Consider increasing max_iter."
        ):
            from_numpy_dynamic(
                data_dynotears_p1["X"], data_dynotears_p1["Y"], max_iter=1
            )

    def test_naming_nodes(self, data_dynotears_p3):
        """
        Nodes should have the format {var}_lag{l}
        """
        sm = from_numpy_dynamic(data_dynotears_p3["X"], data_dynotears_p3["Y"])
        pattern = re.compile(r"[0-9]_lag[0-3]")

        for node in sm.nodes:
            match = pattern.match(node)
            assert match
            assert match.group() == node

    def test_inter_edges(self, data_dynotears_p3):
        """
        inter-slice edges must be {var}_lag{l} -> {var'}_lag0 , l > 0
        """

        sm = from_numpy_dynamic(data_dynotears_p3["X"], data_dynotears_p3["Y"])

        for start, end in sm.edges:
            if int(start[-1]) > 0:
                assert int(end[-1]) == 0

    def test_expected_structure_learned_p1(self, data_dynotears_p1):
        """
        Given a small data set with p=1, find all the intra-slice edges and the majority of the inter-slice ones
        """

        sm = from_numpy_dynamic(
            data_dynotears_p1["X"], data_dynotears_p1["Y"], w_threshold=0.2
        )
        w_edges = [
            ("{i}_lag0".format(i=i), "{j}_lag0".format(j=j))
            for i in range(5)
            for j in range(5)
            if data_dynotears_p1["W"][i, j] != 0
        ]
        a_edges = [
            ("{i_1}_lag{i_2}".format(i_1=i % 5, i_2=1 + i // 5), "{j}_lag0".format(j=j))
            for i in range(5)
            for j in range(5)
            if data_dynotears_p1["A"][i, j] != 0
        ]

        edges_in_sm_and_a = [el for el in sm.edges if el in a_edges]
        sm_inter_edges = [el for el in sm.edges if "lag0" not in el[0]]

        assert sorted([el for el in sm.edges if "lag0" in el[0]]) == sorted(w_edges)
        assert len(edges_in_sm_and_a) / len(a_edges) > 0.6
        assert len(edges_in_sm_and_a) / len(sm_inter_edges) > 0.9

    def test_expected_structure_learned_p2(self, data_dynotears_p2):
        """
        Given a small data set with p=2, all the intra-slice must be correct, and 90%+ found.
        the majority of the inter edges must be found too
        """

        sm = from_numpy_dynamic(
            data_dynotears_p2["X"], data_dynotears_p2["Y"], w_threshold=0.25
        )
        w_edges = [
            ("{i}_lag0".format(i=i), "{j}_lag0".format(j=j))
            for i in range(5)
            for j in range(5)
            if data_dynotears_p2["W"][i, j] != 0
        ]
        a_edges = [
            ("{i_1}_lag{i_2}".format(i_1=i % 5, i_2=1 + i // 5), "{j}_lag0".format(j=j))
            for i in range(5)
            for j in range(5)
            if data_dynotears_p2["A"][i, j] != 0
        ]

        edges_in_sm_and_a = [el for el in sm.edges if el in a_edges]
        sm_inter_edges = [el for el in sm.edges if "lag0" not in el[0]]
        sm_intra_edges = [el for el in sm.edges if "lag0" in el[0]]

        assert len([el for el in sm_intra_edges if el not in w_edges]) == 0
        assert (
            len([el for el in w_edges if el not in sm_intra_edges]) / len(w_edges)
            <= 1.0
        )
        assert len(edges_in_sm_and_a) / len(a_edges) > 0.5
        assert len(edges_in_sm_and_a) / len(sm_inter_edges) > 0.5

    def test_tabu_parents(self, data_dynotears_p2):
        """
        If tabu relationships are set, the corresponding edges must not exist
        """

        sm = from_numpy_dynamic(
            data_dynotears_p2["X"], data_dynotears_p2["Y"], tabu_parent_nodes=[1],
        )
        assert not [el for el in sm.edges if "1_lag" in el[0]]

    def test_tabu_children(self, data_dynotears_p2):
        """
        If tabu relationships are set, the corresponding edges must not exist
        """
        sm = from_numpy_dynamic(
            data_dynotears_p2["X"], data_dynotears_p2["Y"], tabu_child_nodes=[4],
        )
        assert not ([el for el in sm.edges if "4_lag" in el[1]])

        sm = from_numpy_dynamic(
            data_dynotears_p2["X"], data_dynotears_p2["Y"], tabu_child_nodes=[1],
        )
        assert not ([el for el in sm.edges if "1_lag" in el[1]])

    def test_tabu_edges(self, data_dynotears_p2):
        """
        Tabu edges must not be in the edges learnt
        """
        sm = from_numpy_dynamic(
            data_dynotears_p2["X"],
            data_dynotears_p2["Y"],
            tabu_edges=[(0, 2, 4), (0, 0, 3), (1, 1, 4), (1, 3, 4)],
        )

        assert ("2_lag0", "4_lag0") not in sm.edges
        assert ("0_lag0", "3_lag0") not in sm.edges
        assert ("1_lag1", "4_lag0") not in sm.edges
        assert ("3_lag1", "4_lag0") not in sm.edges

    def test_multiple_tabu(self, data_dynotears_p2):
        """
        If tabu relationships are set, the corresponding edges must not exist
        """
        sm = from_numpy_dynamic(
            data_dynotears_p2["X"],
            data_dynotears_p2["Y"],
            tabu_edges=[(0, 1, 4), (0, 0, 3), (1, 1, 4), (1, 3, 4)],
            tabu_child_nodes=[0, 1],
            tabu_parent_nodes=[3],
        )

        assert ("1_lag0", "4_lag0") not in sm.edges
        assert ("0_lag0", "3_lag0") not in sm.edges
        assert ("1_lag1", "4_lag0") not in sm.edges
        assert ("3_lag1", "4_lag0") not in sm.edges
        assert not ([el for el in sm.edges if "0_lag" in el[1]])
        assert not ([el for el in sm.edges if "1_lag" in el[1]])
        assert not ([el for el in sm.edges if "3_lag" in el[0]])

    def test_all_columns_in_structure(self, data_dynotears_p2):
        """Every columns that is in the data should become a node in the learned structure"""
        sm = from_numpy_dynamic(data_dynotears_p2["X"], data_dynotears_p2["Y"],)
        assert sorted(sm.nodes) == [
            "{var}_lag{l_val}".format(var=var, l_val=l_val)
            for var in range(5)
            for l_val in range(3)
        ]

    def test_isolated_nodes_exist(self, data_dynotears_p2):
        """Isolated nodes should still be in the learned structure"""
        sm = from_numpy_dynamic(
            data_dynotears_p2["X"], data_dynotears_p2["Y"], w_threshold=1
        )
        assert len(sm.edges) == 2
        assert len(sm.nodes) == 15

    def test_edges_contain_weight(self, data_dynotears_p2):
        """Edges must contain the 'weight' from the adjacent table """
        sm = from_numpy_dynamic(data_dynotears_p2["X"], data_dynotears_p2["Y"])
        assert np.all([w is not None for u, v, w in sm.edges(data="weight")])

    def test_certain_relationships_get_near_certain_weight(self):
        """If a == b always, ther should be an edge a->b or b->a with coefficient close to one """

        np.random.seed(17)
        data = pd.DataFrame(
            [[np.sqrt(el), np.sqrt(el)] for el in np.random.choice(100, size=500)],
            columns=["a", "b"],
        )
        sm = from_numpy_dynamic(data.values[1:], data.values[:-1], w_threshold=0.1)
        edge = (
            sm.get_edge_data("1_lag0", "0_lag0") or sm.get_edge_data("0_lag0", "1_lag0")
        )["weight"]

        assert 0.99 < edge <= 1.01

    def test_inverse_relationships_get_negative_weight(self):
        """If a == -b always, ther should be an edge a->b or b->a with coefficient close to minus one """

        np.random.seed(17)
        data = pd.DataFrame(
            [[el, -el] for el in np.random.choice(100, size=500)], columns=["a", "b"]
        )
        sm = from_numpy_dynamic(data.values[1:], data.values[:-1], w_threshold=0.1)
        edge = (
            sm.get_edge_data("1_lag0", "0_lag0") or sm.get_edge_data("0_lag0", "1_lag0")
        )["weight"]
        assert -1.01 < edge <= -0.99

    def test_no_cycles(self, data_dynotears_p2):
        """
        The learned structure should be acyclic
        """

        sm = from_numpy_dynamic(
            data_dynotears_p2["X"], data_dynotears_p2["Y"], w_threshold=0.05
        )
        assert nx.algorithms.is_directed_acyclic_graph(sm)

    def test_tabu_edges_on_non_existing_edges_do_nothing(self, data_dynotears_p2):
        """If tabu edges do not exist in the original unconstrained network then nothing changes"""
        sm = from_numpy_dynamic(
            data_dynotears_p2["X"], data_dynotears_p2["Y"], w_threshold=0.2
        )

        sm_2 = from_numpy_dynamic(
            data_dynotears_p2["X"],
            data_dynotears_p2["Y"],
            w_threshold=0.2,
            tabu_edges=[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],
        )
        assert set(sm_2.edges) == set(sm.edges)
