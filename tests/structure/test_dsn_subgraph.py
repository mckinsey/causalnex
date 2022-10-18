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

import pytest
from networkx.exception import NodeNotFound

from causalnex.structure import DynamicStructureModel, DynamicStructureNode

class TestDynamicStructureModelGetLargestSubgraph:
    @pytest.mark.parametrize(
        "test_input, expected",
        [
            (
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(6, 0)),
                ],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                ],
            ),
            (
                [
                    (DynamicStructureNode(3, 0), DynamicStructureNode(4, 0)),
                    (DynamicStructureNode(3, 0), DynamicStructureNode(5, 0)),
                    (DynamicStructureNode(7, 0), DynamicStructureNode(6, 0)),
                ],
                [
                    (DynamicStructureNode(3, 0), DynamicStructureNode(4, 0)),
                    (DynamicStructureNode(3, 0), DynamicStructureNode(5, 0)),
                ],
            ),
            # ([(0, 1), (1, 2), (1, 3), (4, 6)], [(0, 1), (1, 2), (1, 3)]),
            # ([(3, 4), (3, 5), (7, 6)], [(3, 4), (3, 5)]),
        ],
    )
    def test_get_largest_subgraph(self, test_input, expected):
        """Should be able to return the largest subgraph"""

        sm = DynamicStructureModel()
        sm.add_edges_from(test_input)
        largest_subgraph = sm.get_largest_subgraph()

        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected)

        assert set(largest_subgraph.nodes) == set(expected_graph.nodes)
        assert set(largest_subgraph.edges) == set(expected_graph.edges)

    def test_more_than_one_largest(self):
        """Return the first largest when there are more than one largest subgraph"""

        nodes = [
            DynamicStructureNode(0, 0),
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
            DynamicStructureNode(5, 0),
        ]
        edges = [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
            (nodes[3], nodes[4]),
            (nodes[3], nodes[5]),
        ]
        sm = DynamicStructureModel()
        sm.add_edges_from(edges)
        largest_subgraph = sm.get_largest_subgraph()

        expected_edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected_edges)

        assert set(largest_subgraph.nodes) == set(expected_graph.nodes)
        assert set(largest_subgraph.edges) == set(expected_graph.edges)

    def test_empty(self):
        """Should return None if the structure model is empty"""

        sm = DynamicStructureModel()
        assert sm.get_largest_subgraph() is None

    def test_isolates(self):
        """Should return None if the structure model only contains isolates"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(5, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(7, 0),
        ]
        sm.add_nodes(nodes)
        assert sm.get_largest_subgraph() is None

    def test_isolates_nodes_and_edges(self):
        """Should be able to return the largest subgraph"""

        nodes = [
            DynamicStructureNode(0, 0),
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
            DynamicStructureNode(5, 0),
            DynamicStructureNode(6, 0),
            DynamicStructureNode(7, 0),
            DynamicStructureNode(8, 0),
            DynamicStructureNode(9, 0),
        ]
        edges = [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[5], nodes[6]),
        ]
        isolated_nodes = [nodes[7], nodes[8], nodes[9]]
        sm = DynamicStructureModel()
        sm.add_edges_from(edges)
        sm.add_nodes(isolated_nodes)
        largest_subgraph = sm.get_largest_subgraph()

        expected_edges = [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
            (nodes[1], nodes[3]),
        ]
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected_edges)

        assert set(largest_subgraph.nodes) == set(expected_graph.nodes)
        assert set(largest_subgraph.edges) == set(expected_graph.edges)

    def test_different_origins_and_weights(self):
        """The largest subgraph returned should still have the edge data preserved from the original graph"""
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(5, 0),
            DynamicStructureNode(6, 0),
        ]
        sm = DynamicStructureModel()
        sm.add_weighted_edges_from([(nodes[0], nodes[1], 2.0)], origin="unknown")
        sm.add_weighted_edges_from([(nodes[0], nodes[2], 1.0)], origin="learned")
        sm.add_weighted_edges_from([(nodes[3], nodes[4], 0.7)], origin="expert")

        largest_subgraph = sm.get_largest_subgraph()

        assert set(largest_subgraph.edges.data("origin")) == {
            (nodes[0], nodes[1], "unknown"),
            (nodes[0], nodes[2], "learned"),
        }
        assert set(largest_subgraph.edges.data("weight")) == {
            (nodes[0], nodes[1], 2.0),
            (nodes[0], nodes[2], 1.0),
        }


class TestDynamicStructureModelGetTargetSubgraph:
    @pytest.mark.parametrize(
        "target_node, test_input, expected",
        [
            (
                DynamicStructureNode(1, 0),
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(6, 0)),
                ],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                ],
            ),
            (
                DynamicStructureNode(3, 0),
                [
                    (DynamicStructureNode(3, 0), DynamicStructureNode(4, 0)),
                    (DynamicStructureNode(3, 0), DynamicStructureNode(5, 0)),
                    (DynamicStructureNode(7, 0), DynamicStructureNode(6, 0)),
                ],
                [
                    (DynamicStructureNode(3, 0), DynamicStructureNode(4, 0)),
                    (DynamicStructureNode(3, 0), DynamicStructureNode(5, 0)),
                ],
            ),
            (
                DynamicStructureNode(7, 0),
                [
                    (DynamicStructureNode(7, 0), DynamicStructureNode(8, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(7, 0), DynamicStructureNode(6, 0)),
                    (DynamicStructureNode(2, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(5, 0), DynamicStructureNode(1, 0)),
                ],
                [
                    (DynamicStructureNode(7, 0), DynamicStructureNode(8, 0)),
                    (DynamicStructureNode(7, 0), DynamicStructureNode(6, 0)),
                ],
            ),
        ],
    )
    def test_get_target_subgraph(self, target_node, test_input, expected):
        """Should be able to return the subgraph with the specified node"""

        sm = DynamicStructureModel()
        sm.add_edges_from(test_input)
        subgraph = sm.get_target_subgraph(target_node)
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)

    @pytest.mark.parametrize(
        "target_node, test_input, expected",
        [
            (
                DynamicStructureNode("a", 0),
                [
                    (DynamicStructureNode("a", 0), DynamicStructureNode("b", 0)),
                    (DynamicStructureNode("a", 0), DynamicStructureNode("c", 0)),
                    (DynamicStructureNode("c", 0), DynamicStructureNode("d", 0)),
                    (DynamicStructureNode("e", 0), DynamicStructureNode("f", 0)),
                ],
                [
                    (DynamicStructureNode("a", 0), DynamicStructureNode("b", 0)),
                    (DynamicStructureNode("a", 0), DynamicStructureNode("c", 0)),
                    (DynamicStructureNode("c", 0), DynamicStructureNode("d", 0)),
                ],
            ),
            (
                DynamicStructureNode("g", 0),
                [
                    (DynamicStructureNode("g", 0), DynamicStructureNode("h", 0)),
                    (DynamicStructureNode("g", 0), DynamicStructureNode("z", 0)),
                    (DynamicStructureNode("a", 0), DynamicStructureNode("b", 0)),
                    (DynamicStructureNode("a", 0), DynamicStructureNode("c", 0)),
                    (DynamicStructureNode("c", 0), DynamicStructureNode("d", 0)),
                ],
                [
                    (DynamicStructureNode("g", 0), DynamicStructureNode("h", 0)),
                    (DynamicStructureNode("g", 0), DynamicStructureNode("z", 0)),
                ],
            ),
        ],
    )
    def test_get_subgraph_string(self, target_node, test_input, expected):
        """Should be able to return the subgraph with the specified node"""

        sm = DynamicStructureModel()
        sm.add_edges_from(test_input)
        subgraph = sm.get_target_subgraph(target_node)
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)

    @pytest.mark.parametrize(
        "target_node, test_input",
        [
            (
                DynamicStructureNode(7, 0),
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(6, 0)),
                ],
            ),
            (
                DynamicStructureNode(1, 0),
                [
                    (DynamicStructureNode(3, 0), DynamicStructureNode(4, 0)),
                    (DynamicStructureNode(3, 0), DynamicStructureNode(5, 0)),
                    (DynamicStructureNode(7, 0), DynamicStructureNode(6, 0)),
                ],
            ),
        ],
    )
    def test_node_not_in_graph(self, target_node, test_input):
        """Should raise an error if the target_node is not found in the graph"""

        sm = DynamicStructureModel()
        sm.add_edges_from(test_input)

        with pytest.raises(
            NodeNotFound,
            match=re.escape(f"Node {target_node} not found in the graph"),
        ):
            sm.get_target_subgraph(target_node)

    def test_isolates(self):
        """Should return an isolated node"""

        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(5, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(7, 0),
        ]
        sm = DynamicStructureModel()
        sm.add_nodes(nodes)
        subgraph = sm.get_target_subgraph(DynamicStructureNode(1, 0))
        expected_graph = DynamicStructureModel()
        expected_graph.add_node(DynamicStructureNode(1, 0))
        print(f"subgraph nodes {subgraph.nodes}\n")
        print(f"expected nodes {expected_graph.nodes}")
        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)

    def test_isolates_nodes_and_edges(self):
        """Should be able to return the subgraph with the specified node"""

        nodes = [
            DynamicStructureNode(0, 0),
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
            DynamicStructureNode(5, 0),
            DynamicStructureNode(6, 0),
            DynamicStructureNode(7, 0),
            DynamicStructureNode(8, 0),
            DynamicStructureNode(9, 0),
        ]
        edges = [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[5], nodes[6]),
            (nodes[4], nodes[5]),
        ]
        isolated_nodes = [nodes[7], nodes[8], nodes[9]]
        sm = DynamicStructureModel()
        sm.add_edges_from(edges)
        sm.add_nodes(isolated_nodes)
        subgraph = sm.get_target_subgraph(nodes[5])
        expected_edges = [(nodes[5], nodes[6]), (nodes[4], nodes[5])]
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected_edges)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)

    def test_different_origins_and_weights(self):
        """The subgraph returned should still have the edge data preserved from the original graph"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(5, 0),
            DynamicStructureNode(6, 0),
        ]

        sm.add_weighted_edges_from([(nodes[0], nodes[1], 2.0)], origin="unknown")
        sm.add_weighted_edges_from([(nodes[0], nodes[2], 1.0)], origin="learned")
        sm.add_weighted_edges_from([(nodes[3], nodes[4], 0.7)], origin="expert")

        subgraph = sm.get_target_subgraph(nodes[1])

        assert set(subgraph.edges.data("origin")) == {
            (nodes[0], nodes[1], "unknown"),
            (nodes[0], nodes[2], "learned"),
        }
        assert set(subgraph.edges.data("weight")) == {
            (nodes[0], nodes[1], 2.0),
            (nodes[0], nodes[2], 1.0),
        }

    def test_instance_type(self):
        """The subgraph returned should still be a DynamicStructureModel instance"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(0, 0),
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
            DynamicStructureNode(6, 0),
        ]
        sm.add_edges_from(
            [
                (nodes[0], nodes[1]),
                (nodes[1], nodes[2]),
                (nodes[1], nodes[3]),
                (nodes[4], nodes[5]),
            ]
        )
        subgraph = sm.get_target_subgraph(nodes[2])

        assert isinstance(subgraph, DynamicStructureModel)

    def test_get_target_subgraph_twice(self):
        """get_target_subgraph should be able to run more than once"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(0, 0),
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
            DynamicStructureNode(6, 0),
        ]
        sm.add_edges_from(
            [
                (nodes[0], nodes[1]),
                (nodes[1], nodes[2]),
                (nodes[1], nodes[3]),
                (nodes[4], nodes[5]),
            ]
        )

        subgraph = sm.get_target_subgraph(nodes[0])
        subgraph.remove_edge(nodes[0], nodes[1])
        subgraph = subgraph.get_target_subgraph(nodes[1])

        expected_graph = DynamicStructureModel()
        expected_edges = [(nodes[1], nodes[2]), (nodes[1], nodes[3])]
        expected_graph.add_edges_from(expected_edges)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)
