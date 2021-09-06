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

from causalnex.structure.dynotears import from_numpy_dynamic, from_pandas_dynamic


class TestFromNumpyDynotears:
    """Test behaviour of the learn_dynamic_structure of `from_numpy_dynamic`"""

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
            UserWarning, match=r"Failed to converge\. Consider increasing max_iter."
        ):
            from_numpy_dynamic(
                data_dynotears_p1["X"], data_dynotears_p1["Y"], max_iter=1
            )

    def test_naming_nodes(self, data_dynotears_p3):
        """
        Nodes should have the format {var}_lag{l}
        """
        sm = from_numpy_dynamic(data_dynotears_p3["X"], data_dynotears_p3["Y"])
        pattern = re.compile(r"[0-5]_lag[0-3]")

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
            (f"{i}_lag0", f"{j}_lag0")
            for i in range(5)
            for j in range(5)
            if data_dynotears_p1["W"][i, j] != 0
        ]
        a_edges = [
            (f"{i % 5}_lag{1 + i // 5}", f"{j}_lag0")
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
            (f"{i}_lag0", f"{j}_lag0")
            for i in range(5)
            for j in range(5)
            if data_dynotears_p2["W"][i, j] != 0
        ]
        a_edges = [
            (f"{i % 5}_lag{1 + i // 5}", f"{j}_lag0")
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
            data_dynotears_p2["X"],
            data_dynotears_p2["Y"],
            tabu_parent_nodes=[1],
        )
        assert not [el for el in sm.edges if "1_lag" in el[0]]

    def test_tabu_children(self, data_dynotears_p2):
        """
        If tabu relationships are set, the corresponding edges must not exist
        """
        sm = from_numpy_dynamic(
            data_dynotears_p2["X"],
            data_dynotears_p2["Y"],
            tabu_child_nodes=[4],
        )
        assert not ([el for el in sm.edges if "4_lag" in el[1]])

        sm = from_numpy_dynamic(
            data_dynotears_p2["X"],
            data_dynotears_p2["Y"],
            tabu_child_nodes=[1],
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
        sm = from_numpy_dynamic(
            data_dynotears_p2["X"],
            data_dynotears_p2["Y"],
        )
        assert sorted(sm.nodes) == [
            f"{var}_lag{l_val}" for var in range(5) for l_val in range(3)
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
        assert np.all(
            [
                isinstance(w, (float, int, np.number))
                for u, v, w in sm.edges(data="weight")
            ]
        )

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


class TestFromPandasDynotears:
    """Test behaviour of the learn_dynamic_structure of `from_pandas_dynamic`"""

    def test_empty_data_raises_error(self):
        """
        Providing an empty data set should result in a Value Error explaining that data must not be empty.
        This error is useful to catch and handle gracefully, because otherwise the user would experience
        misleading division by zero, or unpacking errors.
        """
        with pytest.raises(
            ValueError, match="Input data X is empty, cannot learn any structure"
        ):
            from_pandas_dynamic(pd.DataFrame(np.empty([2, 5])), p=2)

        with pytest.raises(
            ValueError, match="Input data X is empty, cannot learn any structure"
        ):
            from_pandas_dynamic(pd.DataFrame(np.empty([1, 5])), p=1)

        with pytest.raises(
            ValueError, match="Input data X is empty, cannot learn any structure"
        ):
            from_pandas_dynamic(pd.DataFrame(np.empty([0, 5])), p=1)

    def test_single_iter_gets_converged_fail_warnings(self, data_dynotears_p1):
        """
        With a single iteration on this dataset, learn_structure fails to converge and should give warnings.
        """

        with pytest.warns(
            UserWarning, match=r"Failed to converge\. Consider increasing max_iter."
        ):
            from_pandas_dynamic(pd.DataFrame(data_dynotears_p1["X"]), p=1, max_iter=1)

    def test_naming_nodes(self, data_dynotears_p3):
        """
        Nodes should have the format {var}_lag{l}
        """
        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"]),
            p=3,
        )
        pattern = re.compile(r"[abcde]_lag[0-3]")

        for node in sm.nodes:
            match = pattern.match(node)
            assert match
            assert match.group() == node

    def test_inter_edges(self, data_dynotears_p3):
        """
        inter-slice edges must be {var}_lag{l} -> {var'}_lag0 , l > 0
        """

        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"]),
            p=3,
        )
        for start, end in sm.edges:
            if int(start[-1]) > 0:
                assert int(end[-1]) == 0

    def test_expected_structure_learned_p1(self, data_dynotears_p1):
        """
        Given a small data set with p=1, find all the intra-slice edges and the majority of the inter-slice ones
        """
        df = pd.DataFrame(data_dynotears_p1["X"], columns=["a", "b", "c", "d", "e"])
        df.loc[-1, :] = data_dynotears_p1["Y"][0, :]
        df = df.sort_index()

        sm = from_pandas_dynamic(
            df,
            p=1,
            w_threshold=0.2,
        )
        map_ = dict(zip(range(5), ["a", "b", "c", "d", "e"]))
        w_edges = [
            (f"{map_[i]}_lag0", f"{map_[j]}_lag0")
            for i in range(5)
            for j in range(5)
            if data_dynotears_p1["W"][i, j] != 0
        ]
        a_edges = [
            (
                f"{map_[i % 5]}_lag{1 + i // 5}",
                f"{map_[j]}_lag0",
            )
            for i in range(5)
            for j in range(5)
            if data_dynotears_p1["A"][i, j] != 0
        ]

        edges_in_sm_and_a = [el for el in sm.edges if el in a_edges]
        sm_inter_edges = [el for el in sm.edges if "lag0" not in el[0]]
        assert sorted(el for el in sm.edges if "lag0" in el[0]) == sorted(w_edges)
        assert len(edges_in_sm_and_a) / len(a_edges) > 0.6
        assert len(edges_in_sm_and_a) / len(sm_inter_edges) > 0.9

    def test_expected_structure_learned_p2(self, data_dynotears_p2):
        """
        Given a small data set with p=2, all the intra-slice must be correct, and 90%+ found.
        the majority of the inter edges must be found too
        """
        df = pd.DataFrame(data_dynotears_p2["X"], columns=["a", "b", "c", "d", "e"])
        df.loc[-1, :] = data_dynotears_p2["Y"][0, :5]
        df.loc[-2, :] = data_dynotears_p2["Y"][0, 5:10]

        df = df.sort_index()

        sm = from_pandas_dynamic(
            df,
            p=2,
            w_threshold=0.25,
        )
        map_ = dict(zip(range(5), ["a", "b", "c", "d", "e"]))
        w_edges = [
            (f"{map_[i]}_lag0", f"{map_[j]}_lag0")
            for i in range(5)
            for j in range(5)
            if data_dynotears_p2["W"][i, j] != 0
        ]
        a_edges = [
            (
                f"{map_[i % 5]}_lag{1 + i // 5}",
                f"{map_[j]}_lag0",
            )
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

    def test_tabu_parents(self, data_dynotears_p3):
        """
        If tabu relationships are set, the corresponding edges must not exist
        """

        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"]),
            p=3,
            tabu_parent_nodes=["a"],
        )
        assert not [el for el in sm.edges if "a_lag" in el[0]]

    def test_tabu_children(self, data_dynotears_p3):
        """
        If tabu relationships are set, the corresponding edges must not exist
        """
        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"]),
            p=3,
            tabu_child_nodes=["c", "d"],
        )
        assert not ([el for el in sm.edges if "c_lag" in el[1]])
        assert not ([el for el in sm.edges if "d_lag" in el[1]])
        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"]),
            p=3,
            tabu_child_nodes=["a", "b"],
        )
        assert not ([el for el in sm.edges if "a_lag" in el[1]])
        assert not ([el for el in sm.edges if "b_lag" in el[1]])

    def test_tabu_edges(self, data_dynotears_p3):
        """
        Tabu edges must not be in the edges learnt
        """
        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"]),
            p=3,
            tabu_edges=[(0, "c", "e"), (0, "a", "d"), (1, "b", "e"), (1, "d", "e")],
        )

        assert ("c_lag0", "e_lag0") not in sm.edges
        assert ("a_lag0", "d_lag0") not in sm.edges
        assert ("b_lag1", "e_lag0") not in sm.edges
        assert ("d_lag1", "e_lag0") not in sm.edges

    def test_multiple_tabu(self, data_dynotears_p3):
        """
        If tabu relationships are set, the corresponding edges must not exist
        """
        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"]),
            p=3,
            tabu_edges=[(0, "a", "e"), (0, "a", "d"), (1, "b", "e"), (1, "d", "e")],
            tabu_child_nodes=["a", "b"],
            tabu_parent_nodes=["d"],
        )

        assert ("a_lag0", "e_lag0") not in sm.edges
        assert ("a_lag0", "d_lag0") not in sm.edges
        assert ("b_lag1", "e_lag0") not in sm.edges
        assert ("d_lag1", "e_lag0") not in sm.edges
        assert not ([el for el in sm.edges if "a_lag" in el[1]])
        assert not ([el for el in sm.edges if "b_lag" in el[1]])
        assert not ([el for el in sm.edges if "d_lag" in el[0]])

    def test_all_columns_in_structure(self, data_dynotears_p2):
        """Every columns that is in the data should become a node in the learned structure"""
        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p2["X"], columns=["a", "b", "c", "d", "e"]),
            p=2,
            w_threshold=0.4,
        )
        assert sorted(sm.nodes) == [
            f"{var}_lag{l_val}"
            for var in ["a", "b", "c", "d", "e"]
            for l_val in range(3)
        ]

    def test_isolated_nodes_exist(self, data_dynotears_p2):
        """Isolated nodes should still be in the learned structure"""
        df = pd.DataFrame(data_dynotears_p2["X"], columns=["a", "b", "c", "d", "e"])
        df.loc[-1, :] = data_dynotears_p2["Y"][0, :5]
        df.loc[-2, :] = data_dynotears_p2["Y"][0, 5:10]
        df = df.sort_index()

        sm = from_pandas_dynamic(df, p=2, w_threshold=1)
        assert len(sm.edges) == 2
        assert len(sm.nodes) == 15

    def test_edges_contain_weight(self, data_dynotears_p3):
        """Edges must contain the 'weight' from the adjacent table """
        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p3["X"], columns=["a", "b", "c", "d", "e"]),
            p=3,
        )
        assert np.all(
            [
                isinstance(w, (float, int, np.number))
                for u, v, w in sm.edges(data="weight")
            ]
        )

    def test_certain_relationships_get_near_certain_weight(self):
        """If a == b always, ther should be an edge a->b or b->a with coefficient close to one """

        np.random.seed(17)
        data = pd.DataFrame(
            [[np.sqrt(el), np.sqrt(el)] for el in np.random.choice(100, size=500)],
            columns=["a", "b"],
        )
        sm = from_pandas_dynamic(data, p=1, w_threshold=0.1)
        edge = (
            sm.get_edge_data("b_lag0", "a_lag0") or sm.get_edge_data("a_lag0", "b_lag0")
        )["weight"]

        assert 0.99 < edge <= 1.01

    def test_inverse_relationships_get_negative_weight(self):
        """If a == -b always, there should be an edge a->b or b->a with coefficient close to minus one """

        np.random.seed(17)
        data = pd.DataFrame(
            [[el, -el] for el in np.random.choice(100, size=500)], columns=["a", "b"]
        )
        sm = from_pandas_dynamic(data, p=1, w_threshold=0.1)
        edge = (
            sm.get_edge_data("b_lag0", "a_lag0") or sm.get_edge_data("a_lag0", "b_lag0")
        )["weight"]
        assert -1.01 < edge <= -0.99

    def test_no_cycles(self, data_dynotears_p2):
        """
        The learned structure should be acyclic
        """
        sm = from_pandas_dynamic(
            pd.DataFrame(data_dynotears_p2["X"], columns=["a", "b", "c", "d", "e"]),
            p=2,
            w_threshold=0.05,
        )
        assert nx.algorithms.is_directed_acyclic_graph(sm)

    def test_tabu_edges_on_non_existing_edges_do_nothing(self, data_dynotears_p2):
        """If tabu edges do not exist in the original unconstrained network then nothing changes"""
        df = pd.DataFrame(data_dynotears_p2["X"], columns=["a", "b", "c", "d", "e"])
        df.loc[-1, :] = data_dynotears_p2["Y"][0, :5]
        df.loc[-2, :] = data_dynotears_p2["Y"][0, 5:10]
        df = df.sort_index()

        sm = from_pandas_dynamic(
            df,
            p=2,
            w_threshold=0.2,
        )
        sm_2 = from_pandas_dynamic(
            df,
            p=2,
            w_threshold=0.2,
            tabu_edges=[(0, "a", "a"), (0, "a", "b"), (0, "a", "c"), (0, "a", "d")],
        )
        assert set(sm_2.edges) == set(sm.edges)

    def test_list_of_dfs_as_input(self, data_dynotears_p2):
        """
        the result when given a list of dataframes should be the same as a single dataframe.
        Also, stacking two dataframes should give the same result as well
        """
        df = pd.DataFrame(data_dynotears_p2["X"], columns=["a", "b", "c", "d", "e"])
        df.loc[-1, :] = data_dynotears_p2["Y"][0, :5]
        df.loc[-2, :] = data_dynotears_p2["Y"][0, 5:10]

        df = df.sort_index()
        df_ = df.copy()
        df_.index = range(100, 152)
        df = pd.concat([df, df_])
        sm = from_pandas_dynamic(df, p=2, w_threshold=0.05)
        sm_1 = from_pandas_dynamic([df], p=2, w_threshold=0.05)
        sm_2 = from_pandas_dynamic([df, df], p=2, w_threshold=0.05)

        assert list(sm_2.edges) == list(sm_1.edges)
        assert list(sm.edges) == list(sm_1.edges)

        weights = np.array([w for _, _, w in sm.edges(data="weight")])
        weights_1 = np.array([w for _, _, w in sm_1.edges(data="weight")])
        weights_2 = np.array([w for _, _, w in sm_2.edges(data="weight")])
        assert np.max(np.abs(weights - weights_1)) < 0.001
        assert np.max(np.abs(weights - weights_2)) < 0.001

    def test_discondinuity(self):
        """
        The result when having a point of discontinuity must be the same as if we cut the df in two (on the discont.
        point) and provide the two datasets as input.

        This is because, inside, the algorithm cuts the dfs into continuous datasets
        """
        np.random.seed(12)
        df = pd.DataFrame(np.random.random([100, 5]), columns=["a", "b", "c", "d", "e"])
        df_2 = pd.DataFrame(
            np.random.random([100, 5]),
            columns=["a", "b", "c", "d", "e"],
            index=np.arange(200, 300),
        )

        sm = from_pandas_dynamic(pd.concat([df, df_2], axis=0), p=2, w_threshold=0.05)
        sm_1 = from_pandas_dynamic([df, df_2], p=2, w_threshold=0.05)

        assert [(u, v, round(w, 3)) for u, v, w in sm_1.edges(data="weight")] == [
            (u, v, round(w, 3)) for u, v, w in sm.edges(data="weight")
        ]

    def test_incorrect_input_format(self):
        with pytest.raises(
            ValueError,
            match="Provided empty list of time_series."
            " At least one DataFrame must be provided",
        ):
            from_pandas_dynamic([], 1)

        with pytest.raises(
            ValueError,
            match=r"All columns must have numeric data\. "
            r"Consider mapping the following columns to int: \['a'\]",
        ):
            from_pandas_dynamic(pd.DataFrame([["1"]], columns=["a"]), 1)

        with pytest.raises(
            TypeError,
            match="Time series entries must be instances of `pd.DataFrame`",
        ):
            from_pandas_dynamic([np.array([1, 2])], 1)

        with pytest.raises(
            ValueError,
            match="Index for dataframe must be provided in increasing order",
        ):
            df = pd.DataFrame(np.random.random([5, 5]), index=[3, 1, 2, 5, 0])
            from_pandas_dynamic(df, 1)

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
            from_pandas_dynamic([df, df_2], 1)

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
                columns=["a", "b", "c", "d", "e"],
            )
            df_2["a"] = df_2["a"].astype(int)
            from_pandas_dynamic([df, df_2], 1)

        with pytest.raises(
            TypeError,
            match="Index must be integers",
        ):
            df = pd.DataFrame(np.random.random([5, 5]), index=[0, 1, 2, 3.0, 4])
            from_pandas_dynamic(df, 1)
