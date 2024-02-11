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


class TestDynamicStructureModel:
    def test_init_has_origin(self):
        """Creating a DynamicStructureModel using constructor should give all edges unknown origin"""
        nodes = [DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)]
        sm = DynamicStructureModel([(nodes[0], nodes[1])])
        assert (nodes[0], nodes[1]) in sm.edges
        assert (nodes[0], nodes[1], "unknown") in sm.edges.data("origin")

    def test_init_with_origin(self):
        """should be possible to specify origin during init"""

        nodes = [DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)]
        sm = DynamicStructureModel([(nodes[0], nodes[1])], origin="learned")
        assert (nodes[0], nodes[1], "learned") in sm.edges.data("origin")

    def test_edge_unknown_property(self):
        """should return only edges whose origin is unknown"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0), origin="unknown")
        sm.add_edge((1, 0), (3, 0), origin="learned")
        sm.add_edge((1, 0), (4, 0), origin="expert")

        assert sm.edges_with_origin("unknown") == [
            (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0))
        ]

    def test_edge_learned_property(self):
        """should return only edges whose origin is unknown"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0), origin="unknown")
        sm.add_edge((1, 0), (3, 0), origin="learned")
        sm.add_edge((1, 0), (4, 0), origin="expert")

        assert sm.edges_with_origin("learned") == [
            (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0))
        ]

    def test_edge_expert_property(self):
        """should return only edges whose origin is unknown"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0), origin="unknown")
        sm.add_edge((1, 0), (3, 0), origin="learned")
        sm.add_edge((1, 0), (4, 0), origin="expert")

        assert sm.edges_with_origin("expert") == [
            (DynamicStructureNode(1, 0), DynamicStructureNode(4, 0))
        ]

    def test_to_directed(self):
        """should create a structure model"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
        ]

        edges = [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[0]),
            (nodes[1], nodes[2]),
            (nodes[2], nodes[3]),
        ]
        sm.add_edges_from(edges)

        dag = sm.to_directed()

        assert isinstance(dag, DynamicStructureModel)
        assert all((edge[0], edge[1]) in dag.edges for edge in edges)

    def test_to_undirected(self):
        """should create an undirected Graph"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
        ]

        edges = [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[0]),
            (nodes[1], nodes[2]),
            (nodes[2], nodes[3]),
        ]
        sm.add_edges_from(edges)

        udg = sm.to_undirected()

        assert all(
            (edge[0], edge[1]) in udg.edges
            for edge in [(nodes[1], nodes[2]), (nodes[2], nodes[3])]
        )
        assert (nodes[0], nodes[1]) in udg.edges or (nodes[1], nodes[0]) in udg.edges
        assert len(udg.edges) == 3


class TestDynamicStructureModelAddEdge:
    def test_add_edge_default(self):
        """edges added with default origin should be identified as unknown origin"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0))

        assert (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)) in sm.edges
        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            "unknown",
        ) in sm.edges.data("origin")

    def test_add_edge_unknown(self):
        """edges added with unknown origin should be labelled as unknown origin"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0), "unknown")

        assert (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)) in sm.edges
        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            "unknown",
        ) in sm.edges.data("origin")

    def test_add_edge_learned(self):
        """edges added with learned origin should be labelled as learned origin"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0), "learned")

        assert (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)) in sm.edges
        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            "learned",
        ) in sm.edges.data("origin")

    def test_add_edge_expert(self):
        """edges added with expert origin should be labelled as expert origin"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0), "expert")

        assert (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)) in sm.edges
        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            "expert",
        ) in sm.edges.data("origin")

    def test_add_edge_other(self):
        """edges added with other origin should throw an error"""

        sm = DynamicStructureModel()

        with pytest.raises(ValueError, match="^Unknown origin: must be one of.*$"):
            sm.add_edge((1, 0), (2, 0), "other")

    def test_add_edge_custom_attr(self):
        """it should be possible to add an edge with custom attributes"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0), x="Y")

        assert (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)) in sm.edges
        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            "Y",
        ) in sm.edges.data("x")

    def test_add_edge_multiple_times(self):
        """adding an edge again should update the edges origin attr"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0), origin="unknown")
        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            "unknown",
        ) in sm.edges.data("origin")
        sm.add_edge((1, 0), (2, 0), origin="learned")
        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            "learned",
        ) in sm.edges.data("origin")

    def test_add_multiple_edges(self):
        """it should be possible to add multiple edges with different origins"""

        sm = DynamicStructureModel()
        sm.add_edge((1, 0), (2, 0), origin="unknown")
        sm.add_edge((1, 0), (3, 0), origin="learned")
        sm.add_edge((1, 0), (4, 0), origin="expert")

        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            "unknown",
        ) in sm.edges.data("origin")
        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(3, 0),
            "learned",
        ) in sm.edges.data("origin")
        assert (
            DynamicStructureNode(1, 0),
            DynamicStructureNode(4, 0),
            "expert",
        ) in sm.edges.data("origin")


class TestDynamicStructureModelAddEdgesFrom:
    def test_add_edges_from_default(self):
        """edges added with default origin should be identified as unknown origin"""
        print("******************* hello **************************")
        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
        sm.add_edges_from(edges)
        assert all((edge[0], edge[1]) in sm.edges for edge in edges)
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v in edges)

    def test_add_edges_from_unknown(self):
        """edges added with unknown origin should be labelled as unknown origin"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
        sm.add_edges_from(edges, "unknown")

        assert all((u, v) in sm.edges for u, v in edges)
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v in edges)

    def test_add_edges_from_learned(self):
        """edges added with learned origin should be labelled as learned origin"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
        sm.add_edges_from(edges, "learned")

        assert all((u, v) in sm.edges for u, v in edges)
        assert all((u, v, "learned") in sm.edges.data("origin") for u, v in edges)

    def test_add_edges_from_expert(self):
        """edges added with expert origin should be labelled as expert origin"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
        sm.add_edges_from(edges, "expert")

        assert all((u, v) in sm.edges for u, v in edges)
        assert all((u, v, "expert") in sm.edges.data("origin") for u, v in edges)

    def test_add_edges_from_other(self):
        """edges added with other origin should throw an error"""

        sm = DynamicStructureModel()

        with pytest.raises(ValueError, match="^Unknown origin: must be one of.*$"):
            nodes = [DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)]
            sm.add_edges_from([(nodes[0], nodes[1])], "other")

    def test_add_edges_from_custom_attr(self):
        """it should be possible to add edges with custom attributes"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
        sm.add_edges_from(edges, x="Y")

        assert all((u, v) in sm.edges for u, v in edges)
        assert all((u, v, "Y") in sm.edges.data("x") for u, v in edges)

    def test_add_edges_from_multiple_times(self):
        """adding edges again should update the edges origin attr"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
        sm.add_edges_from(edges, "unknown")
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v in edges)
        sm.add_edges_from(edges, "learned")
        assert all((u, v, "learned") in sm.edges.data("origin") for u, v in edges)

    def test_add_multiple_edges(self):
        """it should be possible to add multiple edges with different origins"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
        ]
        sm.add_edges_from([(nodes[0], nodes[1])], origin="unknown")
        sm.add_edges_from([(nodes[0], nodes[2])], origin="learned")
        sm.add_edges_from([(nodes[0], nodes[3])], origin="expert")

        assert (nodes[0], nodes[1], "unknown") in sm.edges.data("origin")
        assert (nodes[0], nodes[2], "learned") in sm.edges.data("origin")
        assert (nodes[0], nodes[3], "expert") in sm.edges.data("origin")


class TestDynamicStructureModelAddWeightedEdgesFrom:
    def test_add_weighted_edges_from_default(self):
        """edges added with default origin should be identified as unknown origin"""
        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        edges = [(nodes[0], nodes[1], 0.5), (nodes[1], nodes[2], 0.5)]
        sm.add_weighted_edges_from(edges)

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v, w in edges)

    def test_add_weighted_edges_from_unknown(self):
        """edges added with unknown origin should be labelled as unknown origin"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        edges = [(nodes[0], nodes[1], 0.5), (nodes[1], nodes[2], 0.5)]
        sm.add_weighted_edges_from(edges, origin="unknown")

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v, w in edges)

    def test_add_weighted_edges_from_learned(self):
        """edges added with learned origin should be labelled as learned origin"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        edges = [(nodes[0], nodes[1], 0.5), (nodes[1], nodes[2], 0.5)]
        sm.add_weighted_edges_from(edges, origin="learned")

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "learned") in sm.edges.data("origin") for u, v, w in edges)

    def test_add_weighted_edges_from_expert(self):
        """edges added with expert origin should be labelled as expert origin"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        # edges = [(1, 2, 0.5), (2, 3, 0.5)]
        edges = [(nodes[0], nodes[1], 0.5), (nodes[1], nodes[2], 0.5)]
        sm.add_weighted_edges_from(edges, origin="expert")

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "expert") in sm.edges.data("origin") for u, v, w in edges)

    def test_add_weighted_edges_from_other(self):
        """edges added with other origin should throw an error"""

        sm = DynamicStructureModel()
        nodes = [DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)]

        with pytest.raises(ValueError, match="^Unknown origin: must be one of.*$"):
            sm.add_weighted_edges_from([(nodes[0], nodes[1], 0.5)], origin="other")

    def test_add_weighted_edges_from_custom_attr(self):
        """it should be possible to add edges with custom attributes"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        # edges = [(1, 2, 0.5), (2, 3, 0.5)]
        edges = [(nodes[0], nodes[1], 0.5), (nodes[1], nodes[2], 0.5)]
        sm.add_weighted_edges_from(edges, x="Y")

        assert all((u, v, w) in sm.edges.data("weight") for u, v, w in edges)
        assert all((u, v, "Y") in sm.edges.data("x") for u, v, _ in edges)

    def test_add_weighted_edges_from_multiple_times(self):
        """adding edges again should update the edges origin attr"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
        ]

        # edges = [(1, 2, 0.5), (2, 3, 0.5)]
        edges = [(nodes[0], nodes[1], 0.5), (nodes[1], nodes[2], 0.5)]

        sm.add_weighted_edges_from(edges, origin="unknown")
        assert all((u, v, "unknown") in sm.edges.data("origin") for u, v, _ in edges)

        sm.add_weighted_edges_from(edges, origin="learned")
        assert all((u, v, "learned") in sm.edges.data("origin") for u, v, _ in edges)

    def test_add_multiple_weighted_edges(self):
        """it should be possible to add multiple edges with different origins"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
        ]
        sm.add_weighted_edges_from([(nodes[0], nodes[1], 0.5)], origin="unknown")
        sm.add_weighted_edges_from([(nodes[0], nodes[2], 0.5)], origin="learned")
        sm.add_weighted_edges_from([(nodes[0], nodes[3], 0.5)], origin="expert")

        assert (nodes[0], nodes[1], "unknown") in sm.edges.data("origin")
        assert (nodes[0], nodes[2], "learned") in sm.edges.data("origin")
        assert (nodes[0], nodes[3], "expert") in sm.edges.data("origin")


class TestDynamicStructureModelRemoveEdgesBelowThreshold:
    def test_remove_edges_below_threshold(self):
        """Edges whose weight is less than a defined threshold should be removed"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
            DynamicStructureNode(5, 0),
        ]

        strong_edges = [
            (nodes[0], nodes[1], 1.0),
            (nodes[0], nodes[2], 0.8),
            (nodes[0], nodes[4], 2.0),
        ]
        weak_edges = [
            (nodes[0], nodes[3], 0.4),
            (nodes[1], nodes[2], 0.6),
            (nodes[2], nodes[4], 0.5),
        ]
        sm.add_weighted_edges_from(strong_edges)
        sm.add_weighted_edges_from(weak_edges)

        sm.remove_edges_below_threshold(0.7)
        assert set(sm.edges(data="weight")) == set(
            (u, v, w) for u, v, w in strong_edges
        )

    def test_negative_weights(self):
        """Negative edges whose absolute value is greater than the defined threshold should not be removed"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
            DynamicStructureNode(5, 0),
        ]

        strong_edges = [
            (nodes[0], nodes[1], -3.0),
            (nodes[2], nodes[0], 0.7),
            (nodes[0], nodes[4], -2.0),
        ]
        weak_edges = [
            (nodes[0], nodes[3], 0.4),
            (nodes[1], nodes[2], -0.6),
            (nodes[2], nodes[4], -0.5),
        ]

        sm.add_weighted_edges_from(strong_edges)
        sm.add_weighted_edges_from(weak_edges)

        sm.remove_edges_below_threshold(0.7)

        assert set(sm.edges(data="weight")) == set(
            (u, v, w) for u, v, w in strong_edges
        )

    def test_equal_weights(self):
        """Edges whose absolute value is equal to the defined threshold should not be removed"""

        sm = DynamicStructureModel()
        nodes = [
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
            DynamicStructureNode(5, 0),
        ]

        strong_edges = [(nodes[0], nodes[1], 1.0), (nodes[0], nodes[4], 2.0)]
        equal_edges = [(nodes[0], nodes[2], 0.6), (nodes[1], nodes[2], 0.6)]
        weak_edges = [(nodes[0], nodes[3], 0.4), (nodes[2], nodes[4], 0.5)]
        sm.add_weighted_edges_from(strong_edges)
        sm.add_weighted_edges_from(equal_edges)
        sm.add_weighted_edges_from(weak_edges)

        sm.remove_edges_below_threshold(0.6)

        assert set(sm.edges(data="weight")) == set.union(
            set((u, v, w) for u, v, w in strong_edges),
            set((u, v, w) for u, v, w in equal_edges),
        )

    def test_graph_with_no_edges(self):
        """Can still run even if the graph is without edges"""
        sm = DynamicStructureModel()
        # (var, lag) - all nodes here are in current timestep
        nodes = [
            DynamicStructureNode(0, 0),
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
        ]

        sm.add_nodes(nodes)
        sm.remove_edges_below_threshold(0.6)

        assert set(sm.nodes) == set(nodes)
        assert set(sm.edges) == set()


class TestDynamicStructureModelGetMarkovBlanket:
    @pytest.mark.parametrize(
        "target_node, test_input, expected",
        [
            (
                DynamicStructureNode(1, 0),
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(5, 0)),
                ],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                ],
            ),
            (
                DynamicStructureNode(1, 0),
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(3, 0)),
                ],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(3, 0)),
                ],
            ),
            (
                DynamicStructureNode(3, 0),
                [
                    (DynamicStructureNode(3, 0), DynamicStructureNode(4, 0)),
                    (DynamicStructureNode(3, 0), DynamicStructureNode(5, 0)),
                    (DynamicStructureNode(6, 0), DynamicStructureNode(7, 0)),
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
                    (DynamicStructureNode(6, 0), DynamicStructureNode(7, 0)),
                    (DynamicStructureNode(2, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(5, 0), DynamicStructureNode(8, 0)),
                ],
                [
                    (DynamicStructureNode(7, 0), DynamicStructureNode(8, 0)),
                    (DynamicStructureNode(6, 0), DynamicStructureNode(7, 0)),
                    (DynamicStructureNode(5, 0), DynamicStructureNode(8, 0)),
                ],
            ),
        ],
    )
    def test_get_markov_blanket_single(self, target_node, test_input, expected):
        """Should be able to return Markov blanket with the specified single node"""

        sm = DynamicStructureModel()
        sm.add_edges_from(test_input)
        blanket = sm.get_markov_blanket(target_node)
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected)

        assert set(blanket.nodes) == set(expected_graph.nodes)
        assert set(blanket.edges) == set(expected_graph.edges)

    @pytest.mark.parametrize(
        "target_nodes, test_input, expected",
        [
            (
                [DynamicStructureNode(1, 0), DynamicStructureNode(4, 0)],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(5, 0)),
                ],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(5, 0)),
                ],
            ),
            (
                [DynamicStructureNode(2, 0), DynamicStructureNode(4, 0)],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(3, 0)),
                ],
                [
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(4, 0), DynamicStructureNode(3, 0)),
                ],
            ),
            (
                [DynamicStructureNode(3, 0), DynamicStructureNode(6, 0)],
                [
                    (DynamicStructureNode(3, 0), DynamicStructureNode(4, 0)),
                    (DynamicStructureNode(3, 0), DynamicStructureNode(5, 0)),
                    (DynamicStructureNode(6, 0), DynamicStructureNode(7, 0)),
                ],
                [
                    (DynamicStructureNode(3, 0), DynamicStructureNode(4, 0)),
                    (DynamicStructureNode(3, 0), DynamicStructureNode(5, 0)),
                    (DynamicStructureNode(6, 0), DynamicStructureNode(7, 0)),
                ],
            ),
            (
                [DynamicStructureNode(2, 0), DynamicStructureNode(5, 0)],
                [
                    (DynamicStructureNode(7, 0), DynamicStructureNode(8, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(6, 0), DynamicStructureNode(7, 0)),
                    (DynamicStructureNode(2, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(5, 0), DynamicStructureNode(8, 0)),
                ],
                [
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                    (DynamicStructureNode(2, 0), DynamicStructureNode(3, 0)),
                    (DynamicStructureNode(7, 0), DynamicStructureNode(8, 0)),
                    (DynamicStructureNode(5, 0), DynamicStructureNode(8, 0)),
                ],
            ),
        ],
    )
    def test_get_markov_blanket_multiple(self, target_nodes, test_input, expected):
        """Should be able to return Markov blanket with the specified list of nodes"""

        sm = DynamicStructureModel()
        sm.add_edges_from(test_input)
        blanket = sm.get_markov_blanket(target_nodes)
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected)

        assert set(blanket.nodes) == set(expected_graph.nodes)
        assert set(blanket.edges) == set(expected_graph.edges)

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
    def test_get_markov_blanket_string(self, target_node, test_input, expected):
        """Should be able to return the subgraph with the specified node"""

        sm = DynamicStructureModel()
        sm.add_edges_from(test_input)
        blanket = sm.get_markov_blanket(target_node)
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected)

        assert set(blanket.nodes) == set(expected_graph.nodes)
        assert set(blanket.edges) == set(expected_graph.edges)

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
            sm.get_markov_blanket(target_node)

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
        blanket = sm.get_markov_blanket(nodes[0])

        expected_graph = DynamicStructureModel()
        expected_graph.add_node(nodes[0])

        assert set(blanket.nodes) == set(expected_graph.nodes)
        assert set(blanket.edges) == set(expected_graph.edges)

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
        subgraph = sm.get_markov_blanket(nodes[5])
        expected_edges = [(nodes[5], nodes[6]), (nodes[4], nodes[5])]
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected_edges)

        assert set(subgraph.nodes) == set(expected_graph.nodes)
        assert set(subgraph.edges) == set(expected_graph.edges)

    def test_instance_type(self):
        """The subgraph returned should still be a DynamicStructureModel instance"""
        nodes = [
            DynamicStructureNode(0, 0),
            DynamicStructureNode(1, 0),
            DynamicStructureNode(2, 0),
            DynamicStructureNode(3, 0),
            DynamicStructureNode(4, 0),
            DynamicStructureNode(6, 0),
        ]
        sm = DynamicStructureModel()
        sm.add_edges_from(
            [
                (nodes[0], nodes[1]),
                (nodes[1], nodes[2]),
                (nodes[1], nodes[3]),
                (nodes[4], nodes[5]),
            ]
        )
        subgraph = sm.get_markov_blanket(nodes[2])

        assert isinstance(subgraph, DynamicStructureModel)
