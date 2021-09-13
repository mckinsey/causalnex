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

import pytest
from networkx.exception import NodeNotFound

from causalnex.structure import StructureModel


class TestStructureModel:
    def test_init_has_origin(self):
        """Creating a StructureModel using constructor should give all edges unknown origin"""

        sm = StructureModel([(1, 2)])
        assert (1, 2) in sm.edges
        assert (1, 2, "unknown") in sm.edges.data("origin")

    def test_init_with_origin(self):
        """should be possible to specify origin during init"""

        sm = StructureModel([(1, 2)], origin="learned")
        assert (1, 2, "learned") in sm.edges.data("origin")

    def test_edge_unknown_property(self):
        """should return only edges whose origin is unknown"""

        sm = StructureModel()
        sm.add_edge(1, 2, origin="unknown")
        sm.add_edge(1, 3, origin="learned")
        sm.add_edge(1, 4, origin="expert")

        assert sm.edges_with_origin("unknown") == [(1, 2)]

    def test_edge_learned_property(self):
        """should return only edges whose origin is unknown"""

        sm = StructureModel()
        sm.add_edge(1, 2, origin="unknown")
        sm.add_edge(1, 3, origin="learned")
        sm.add_edge(1, 4, origin="expert")

        assert sm.edges_with_origin("learned") == [(1, 3)]

    def test_edge_expert_property(self):
        """should return only edges whose origin is unknown"""

        sm = StructureModel()
        sm.add_edge(1, 2, origin="unknown")
        sm.add_edge(1, 3, origin="learned")
        sm.add_edge(1, 4, origin="expert")

        assert sm.edges_with_origin("expert") == [(1, 4)]

    def test_to_directed(self):
        """should create a structure model"""

        sm = StructureModel()
        edges = [(1, 2), (2, 1), (2, 3), (3, 4)]
        sm.add_edges_from(edges)

        dag = sm.to_directed()
        assert isinstance(dag, StructureModel)
        assert all(edge in dag.edges for edge in edges)

    def test_to_undirected(self):
        """should create an undirected Graph"""

        sm = StructureModel()
        sm.add_edges_from([(1, 2), (2, 1), (2, 3), (3, 4)])

        udg = sm.to_undirected()
        assert all(edge in udg.edges for edge in [(2, 3), (3, 4)])
        assert (1, 2) in udg.edges or (2, 1) in udg.edges
        assert len(udg.edges) == 3


class TestStructureModelAddEdge:
    def test_add_edge_default(self):
        """edges added with default origin should be identified as unknown origin"""

        sm = StructureModel()
        sm.add_edge(1, 2)

        assert (1, 2) in sm.edges
        assert (1, 2, "unknown") in sm.edges.data("origin")

    def test_add_edge_unknown(self):
        """edges added with unknown origin should be labelled as unknown origin"""

        sm = StructureModel()
        sm.add_edge(1, 2, "unknown")

        assert (1, 2) in sm.edges
        assert (1, 2, "unknown") in sm.edges.data("origin")

    def test_add_edge_learned(self):
        """edges added with learned origin should be labelled as learned origin"""

        sm = StructureModel()
        sm.add_edge(1, 2, "learned")

        assert (1, 2) in sm.edges
        assert (1, 2, "learned") in sm.edges.data("origin")

    def test_add_edge_expert(self):
        """edges added with expert origin should be labelled as expert origin"""

        sm = StructureModel()
        sm.add_edge(1, 2, "expert")

        assert (1, 2) in sm.edges
        assert (1, 2, "expert") in sm.edges.data("origin")

    def test_add_edge_other(self):
        """edges added with other origin should throw an error"""

        sm = StructureModel()

        with pytest.raises(ValueError, match="^Unknown origin: must be one of.*$"):
            sm.add_edge(1, 2, "other")

    def test_add_edge_custom_attr(self):
        """it should be possible to add an edge with custom attributes"""

        sm = StructureModel()
        sm.add_edge(1, 2, x="Y")

        assert (1, 2) in sm.edges
        assert (1, 2, "Y") in sm.edges.data("x")

    def test_add_edge_multiple_times(self):
        """adding an edge again should update the edges origin attr"""

        sm = StructureModel()
        sm.add_edge(1, 2, origin="unknown")
        assert (1, 2, "unknown") in sm.edges.data("origin")
        sm.add_edge(1, 2, origin="learned")
        assert (1, 2, "learned") in sm.edges.data("origin")

    def test_add_multiple_edges(self):
        """it should be possible to add multiple edges with different origins"""

        sm = StructureModel()
        sm.add_edge(1, 2, origin="unknown")
        sm.add_edge(1, 3, origin="learned")
        sm.add_edge(1, 4, origin="expert")

        assert (1, 2, "unknown") in sm.edges.data("origin")
        assert (1, 3, "learned") in sm.edges.data("origin")
        assert (1, 4, "expert") in sm.edges.data("origin")


class TestStructureModelAddEdgesFrom:
    def test_add_edges_from_default(self):
        """edges added with default origin should be identified as unknown origin"""

        sm = StructureModel()
        edges = [(1, 2), (2, 3)]
        sm.add_edges_from(edges)

        assert all(edge in sm.edges for edge in edges)
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v in edges)

    def test_add_edges_from_unknown(self):
        """edges added with unknown origin should be labelled as unknown origin"""

        sm = StructureModel()
        edges = [(1, 2), (2, 3)]
        sm.add_edges_from(edges, "unknown")

        assert all(edge in sm.edges for edge in edges)
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v in edges)

    def test_add_edges_from_learned(self):
        """edges added with learned origin should be labelled as learned origin"""

        sm = StructureModel()
        edges = [(1, 2), (2, 3)]
        sm.add_edges_from(edges, "learned")

        assert all(edge in sm.edges for edge in edges)
        assert all((u, v, "learned") in sm.edges.data("origin") for u, v in edges)

    def test_add_edges_from_expert(self):
        """edges added with expert origin should be labelled as expert origin"""

        sm = StructureModel()
        edges = [(1, 2), (2, 3)]
        sm.add_edges_from(edges, "expert")

        assert all(edge in sm.edges for edge in edges)
        assert all((u, v, "expert") in sm.edges.data("origin") for u, v in edges)

    def test_add_edges_from_other(self):
        """edges added with other origin should throw an error"""

        sm = StructureModel()

        with pytest.raises(ValueError, match="^Unknown origin: must be one of.*$"):
            sm.add_edges_from([(1, 2)], "other")

    def test_add_edges_from_custom_attr(self):
        """it should be possible to add edges with custom attributes"""

        sm = StructureModel()
        edges = [(1, 2), (2, 3)]
        sm.add_edges_from(edges, x="Y")

        assert all(edge in sm.edges for edge in edges)
        assert all((u, v, "Y") in sm.edges.data("x") for u, v in edges)

    def test_add_edges_from_multiple_times(self):
        """adding edges again should update the edges origin attr"""

        sm = StructureModel()
        edges = [(1, 2), (2, 3)]
        sm.add_edges_from(edges, "unknown")
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v in edges)
        sm.add_edges_from(edges, "learned")
        assert all((u, v, "learned") in sm.edges.data("origin") for u, v in edges)

    def test_add_multiple_edges(self):
        """it should be possible to add multiple edges with different origins"""

        sm = StructureModel()
        sm.add_edges_from([(1, 2)], origin="unknown")
        sm.add_edges_from([(1, 3)], origin="learned")
        sm.add_edges_from([(1, 4)], origin="expert")

        assert (1, 2, "unknown") in sm.edges.data("origin")
        assert (1, 3, "learned") in sm.edges.data("origin")
        assert (1, 4, "expert") in sm.edges.data("origin")


class TestStructureModelAddWeightedEdgesFrom:
    def test_add_weighted_edges_from_default(self):
        """edges added with default origin should be identified as unknown origin"""

        sm = StructureModel()
        edges = [(1, 2, 0.5), (2, 3, 0.5)]
        sm.add_weighted_edges_from(edges)

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v, w in edges)

    def test_add_weighted_edges_from_unknown(self):
        """edges added with unknown origin should be labelled as unknown origin"""

        sm = StructureModel()
        edges = [(1, 2, 0.5), (2, 3, 0.5)]
        sm.add_weighted_edges_from(edges, origin="unknown")

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v, w in edges)

    def test_add_weighted_edges_from_learned(self):
        """edges added with learned origin should be labelled as learned origin"""

        sm = StructureModel()
        edges = [(1, 2, 0.5), (2, 3, 0.5)]
        sm.add_weighted_edges_from(edges, origin="learned")

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "learned") in sm.edges.data("origin") for u, v, w in edges)

    def test_add_weighted_edges_from_expert(self):
        """edges added with expert origin should be labelled as expert origin"""

        sm = StructureModel()
        edges = [(1, 2, 0.5), (2, 3, 0.5)]
        sm.add_weighted_edges_from(edges, origin="expert")

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "expert") in sm.edges.data("origin") for u, v, w in edges)

    def test_add_weighted_edges_from_other(self):
        """edges added with other origin should throw an error"""

        sm = StructureModel()

        with pytest.raises(ValueError, match="^Unknown origin: must be one of.*$"):
            sm.add_weighted_edges_from([(1, 2, 0.5)], origin="other")

    def test_add_weighted_edges_from_custom_attr(self):
        """it should be possible to add edges with custom attributes"""

        sm = StructureModel()
        edges = [(1, 2, 0.5), (2, 3, 0.5)]
        sm.add_weighted_edges_from(edges, x="Y")

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "Y") in sm.edges.data("x") for u, v, _ in edges)

    def test_add_weighted_edges_from_multiple_times(self):
        """adding edges again should update the edges origin attr"""

        sm = StructureModel()
        edges = [(1, 2, 0.5), (2, 3, 0.5)]
        sm.add_weighted_edges_from(edges, origin="unknown")
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v, _ in edges)
        sm.add_weighted_edges_from(edges, origin="learned")
        assert all((u, v, "learned") in sm.edges.data("origin") for u, v, _ in edges)

    def test_add_multiple_weighted_edges(self):
        """it should be possible to add multiple edges with different origins"""

        sm = StructureModel()
        sm.add_weighted_edges_from([(1, 2, 0.5)], origin="unknown")
        sm.add_weighted_edges_from([(1, 3, 0.5)], origin="learned")
        sm.add_weighted_edges_from([(1, 4, 0.5)], origin="expert")

        assert (1, 2, "unknown") in sm.edges.data("origin")
        assert (1, 3, "learned") in sm.edges.data("origin")
        assert (1, 4, "expert") in sm.edges.data("origin")


class TestStructureModelRemoveEdgesBelowThreshold:
    def test_remove_edges_below_threshold(self):
        """Edges whose weight is less than a defined threshold should be removed"""

        sm = StructureModel()
        strong_edges = [(1, 2, 1.0), (1, 3, 0.8), (1, 5, 2.0)]
        weak_edges = [(1, 4, 0.4), (2, 3, 0.6), (3, 5, 0.5)]
        sm.add_weighted_edges_from(strong_edges)
        sm.add_weighted_edges_from(weak_edges)

        sm.remove_edges_below_threshold(0.7)

        assert set(sm.edges(data="weight")) == set(strong_edges)

    def test_negative_weights(self):
        """Negative edges whose absolute value is greater than the defined threshold should not be removed"""

        sm = StructureModel()
        strong_edges = [(1, 2, -3.0), (3, 1, 0.7), (1, 5, -2.0)]
        weak_edges = [(1, 4, 0.4), (2, 3, -0.6), (3, 5, -0.5)]
        sm.add_weighted_edges_from(strong_edges)
        sm.add_weighted_edges_from(weak_edges)

        sm.remove_edges_below_threshold(0.7)

        assert set(sm.edges(data="weight")) == set(strong_edges)

    def test_equal_weights(self):
        """Edges whose absolute value is equal to the defined threshold should not be removed"""

        sm = StructureModel()
        strong_edges = [(1, 2, 1.0), (1, 5, 2.0)]
        equal_edges = [(1, 3, 0.6), (2, 3, 0.6)]
        weak_edges = [(1, 4, 0.4), (3, 5, 0.5)]
        sm.add_weighted_edges_from(strong_edges)
        sm.add_weighted_edges_from(equal_edges)
        sm.add_weighted_edges_from(weak_edges)

        sm.remove_edges_below_threshold(0.6)

        assert set(sm.edges(data="weight")) == set.union(
            set(strong_edges), set(equal_edges)
        )

    def test_graph_with_no_edges(self):
        """Can still run even if the graph is without edges"""

        sm = StructureModel()
        nodes = [1, 2, 3]
        sm.add_nodes_from(nodes)
        sm.remove_edges_below_threshold(0.6)

        assert set(sm.nodes) == set(nodes)
        assert set(sm.edges) == set()


class TestStructureModelGetLargestSubgraph:
    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([(0, 1), (1, 2), (1, 3), (4, 6)], [(0, 1), (1, 2), (1, 3)]),
            ([(3, 4), (3, 5), (7, 6)], [(3, 4), (3, 5)]),
        ],
    )
    def test_get_largest_subgraph(self, test_input, expected):
        """Should be able to return the largest subgraph"""
        sm = StructureModel()
        sm.add_edges_from(test_input)
        largest_subgraph = sm.get_largest_subgraph()

        expected_graph = StructureModel()
        expected_graph.add_edges_from(expected)

        assert set(largest_subgraph.nodes) == set(expected_graph.nodes)
        assert set(largest_subgraph.edges) == set(expected_graph.edges)

    def test_more_than_one_largest(self):
        """Return the first largest when there are more than one largest subgraph"""

        edges = [(0, 1), (1, 2), (3, 4), (3, 5)]
        sm = StructureModel()
        sm.add_edges_from(edges)
        largest_subgraph = sm.get_largest_subgraph()

        expected_edges = [(0, 1), (1, 2)]
        expected_graph = StructureModel()
        expected_graph.add_edges_from(expected_edges)

        assert set(largest_subgraph.nodes) == set(expected_graph.nodes)
        assert set(largest_subgraph.edges) == set(expected_graph.edges)

    def test_empty(self):
        """Should return None if the structure model is empty"""

        sm = StructureModel()
        assert sm.get_largest_subgraph() is None

    def test_isolates(self):
        """Should return None if the structure model only contains isolates"""
        nodes = [1, 3, 5, 2, 7]

        sm = StructureModel()
        sm.add_nodes_from(nodes)

        assert sm.get_largest_subgraph() is None

    def test_isolates_nodes_and_edges(self):
        """Should be able to return the largest subgraph"""

        edges = [(0, 1), (1, 2), (1, 3), (5, 6)]
        isolated_nodes = [7, 8, 9]
        sm = StructureModel()
        sm.add_edges_from(edges)
        sm.add_nodes_from(isolated_nodes)
        largest_subgraph = sm.get_largest_subgraph()

        expected_edges = [(0, 1), (1, 2), (1, 3)]
        expected_graph = StructureModel()
        expected_graph.add_edges_from(expected_edges)

        assert set(largest_subgraph.nodes) == set(expected_graph.nodes)
        assert set(largest_subgraph.edges) == set(expected_graph.edges)

    def test_different_origins_and_weights(self):
        """The largest subgraph returned should still have the edge data preserved from the original graph"""

        sm = StructureModel()
        sm.add_weighted_edges_from([(1, 2, 2.0)], origin="unknown")
        sm.add_weighted_edges_from([(1, 3, 1.0)], origin="learned")
        sm.add_weighted_edges_from([(5, 6, 0.7)], origin="expert")

        largest_subgraph = sm.get_largest_subgraph()

        assert set(largest_subgraph.edges.data("origin")) == {
            (1, 2, "unknown"),
            (1, 3, "learned"),
        }
        assert set(largest_subgraph.edges.data("weight")) == {(1, 2, 2.0), (1, 3, 1.0)}


class TestStructureModelGetTargetSubgraph:
    @pytest.mark.parametrize(
        "target_node, test_input, expected",
        [
            (1, [(0, 1), (1, 2), (1, 3), (4, 6)], [(0, 1), (1, 2), (1, 3)]),
            (3, [(3, 4), (3, 5), (7, 6)], [(3, 4), (3, 5)]),
            (7, [(7, 8), (1, 2), (7, 6), (2, 3), (5, 1)], [(7, 8), (7, 6)]),
        ],
    )
    def test_get_target_subgraph(self, target_node, test_input, expected):
        """Should be able to return the subgraph with the specified node"""
        sm = StructureModel()
        sm.add_edges_from(test_input)
        subgraph = sm.get_target_subgraph(target_node)
        expected_graph = StructureModel()
        expected_graph.add_edges_from(expected)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)

    @pytest.mark.parametrize(
        "target_node, test_input, expected",
        [
            (
                "a",
                [("a", "b"), ("a", "c"), ("c", "d"), ("e", "f")],
                [("a", "b"), ("a", "c"), ("c", "d")],
            ),
            (
                "g",
                [("g", "h"), ("g", "z"), ("a", "b"), ("a", "c"), ("c", "d")],
                [("g", "h"), ("g", "z")],
            ),
        ],
    )
    def test_get_subgraph_string(self, target_node, test_input, expected):
        """Should be able to return the subgraph with the specified node"""
        sm = StructureModel()
        sm.add_edges_from(test_input)
        subgraph = sm.get_target_subgraph(target_node)
        expected_graph = StructureModel()
        expected_graph.add_edges_from(expected)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)

    @pytest.mark.parametrize(
        "target_node, test_input",
        [(7, [(0, 1), (1, 2), (1, 3), (4, 6)]), (1, [(3, 4), (3, 5), (7, 6)])],
    )
    def test_node_not_in_graph(self, target_node, test_input):
        """Should raise an error if the target_node is not found in the graph"""

        sm = StructureModel()
        sm.add_edges_from(test_input)

        with pytest.raises(
            NodeNotFound,
            match=f"Node {target_node} not found in the graph",
        ):
            sm.get_target_subgraph(target_node)

    def test_isolates(self):
        """Should return an isolated node"""
        nodes = [1, 3, 5, 2, 7]

        sm = StructureModel()
        sm.add_nodes_from(nodes)
        subgraph = sm.get_target_subgraph(1)

        expected_graph = StructureModel()
        expected_graph.add_node(1)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)

    def test_isolates_nodes_and_edges(self):
        """Should be able to return the subgraph with the specified node"""

        edges = [(0, 1), (1, 2), (1, 3), (5, 6), (4, 5)]
        isolated_nodes = [7, 8, 9]
        sm = StructureModel()
        sm.add_edges_from(edges)
        sm.add_nodes_from(isolated_nodes)
        subgraph = sm.get_target_subgraph(5)
        expected_edges = [(5, 6), (4, 5)]
        expected_graph = StructureModel()
        expected_graph.add_edges_from(expected_edges)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)

    def test_different_origins_and_weights(self):
        """The subgraph returned should still have the edge data preserved from the original graph"""

        sm = StructureModel()
        sm.add_weighted_edges_from([(1, 2, 2.0)], origin="unknown")
        sm.add_weighted_edges_from([(1, 3, 1.0)], origin="learned")
        sm.add_weighted_edges_from([(5, 6, 0.7)], origin="expert")

        subgraph = sm.get_target_subgraph(2)

        assert set(subgraph.edges.data("origin")) == {
            (1, 2, "unknown"),
            (1, 3, "learned"),
        }
        assert set(subgraph.edges.data("weight")) == {(1, 2, 2.0), (1, 3, 1.0)}

    def test_instance(self):
        """The subgraph returned should still be a StructureModel instance"""
        sm = StructureModel()
        sm.add_edges_from([(0, 1), (1, 2), (1, 3), (4, 6)])

        subgraph = sm.get_target_subgraph(2)

        assert isinstance(subgraph, StructureModel)

    def test_get_target_subgraph_twice(self):
        """get_target_subgraph should be able to run more than once"""
        sm = StructureModel()
        sm.add_edges_from([(0, 1), (1, 2), (1, 3), (4, 6)])

        subgraph = sm.get_target_subgraph(0)
        subgraph.remove_edge(0, 1)
        subgraph = subgraph.get_target_subgraph(1)

        expected_graph = StructureModel()
        expected_edges = [(1, 2), (1, 3)]
        expected_graph.add_edges_from(expected_edges)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)
