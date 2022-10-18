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

from causalnex.structure import DynamicStructureModel, DynamicStructureNode


class TestDynamicStructureModelEdgeCoercion:
    def test_edge_not_tuple(self):
        edges = [((1, 0), (3, 0), 0.5), 6]
        sm = DynamicStructureModel()

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Edges must be tuples containing 2 or 3 elements, received {edges}"
            ),
        ):
            sm.add_edges_from(edges)

    def test_multi_edge_not_dsn(self):
        edges = [((0, 0), (1, 0)), ((1, 0), (2, 0))]
        sm = DynamicStructureModel()
        sm.add_edges_from(edges)

        expected_edges = [
            (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
            (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
        ]
        expected_graph = DynamicStructureModel()
        expected_graph.add_edges_from(expected_edges)

        assert set(sm.nodes) == set(expected_graph.nodes)
        assert set(sm.edges) == set(expected_graph.edges)

    def test_weighted_multi_edge_not_dsn(self):
        edges = [((0, 0), (1, 0), 0.5), ((1, 0), (2, 0), 0.7)]
        sm = DynamicStructureModel()
        sm.add_weighted_edges_from(edges)

        expected_edges = [
            (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0), 0.5),
            (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0), 0.7),
        ]
        expected_graph = DynamicStructureModel()
        expected_graph.add_weighted_edges_from(expected_edges)

        assert set(sm.nodes) == set(expected_graph.nodes)
        assert set(sm.edges) == set(expected_graph.edges)

    @pytest.mark.parametrize(
        "input_edges, expected_edges",
        [
            (
                [((0, 0), (1, 0), 0.5), ((1, 0), (2, 0), 0.7)],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0), 0.5),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0), 0.7),
                ],
            ),
            (
                [((0, 0), (1, 0)), ((1, 0), (2, 0))],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                ],
            ),
        ],
    )
    def test_multi_edge_dsn(self, input_edges, expected_edges):
        sm = DynamicStructureModel()
        weighted = len(input_edges[0]) == 3
        if not weighted:
            sm.add_edges_from(input_edges)
        else:
            sm.add_weighted_edges_from(input_edges)

        expected_graph = DynamicStructureModel()
        if not weighted:
            expected_graph.add_edges_from(expected_edges)
        else:
            expected_graph.add_weighted_edges_from(expected_edges)

        assert set(sm.nodes) == set(expected_graph.nodes)
        assert set(sm.edges) == set(expected_graph.edges)

    def test_node_not_tuple(self):
        edges = [((1, 0), (3, 0), 0.5), ((1, 0), 3, 0.7)]
        sm = DynamicStructureModel()

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Nodes in {edges[1]} must be tuples with node name and time step"
            ),
        ):
            sm.add_edges_from(edges)

    @pytest.mark.parametrize(
        "input_edges, expected_edges",
        [
            (
                [
                    (DynamicStructureNode(0, 0), (1, 0), 0.5),
                    ((1, 0), DynamicStructureNode(2, 0), 0.7),
                ],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0), 0.5),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0), 0.7),
                ],
            ),
            (
                [
                    (DynamicStructureNode(0, 0), (1, 0)),
                    ((1, 0), DynamicStructureNode(2, 0)),
                ],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                ],
            ),
            (
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    ((1, 0), DynamicStructureNode(2, 0)),
                ],
                [
                    (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                    (DynamicStructureNode(1, 0), DynamicStructureNode(2, 0)),
                ],
            ),
        ],
    )
    def test_multi_edge_one_dsn(self, input_edges, expected_edges):
        sm = DynamicStructureModel()
        weighted = len(input_edges[0]) == 3
        if not weighted:
            sm.add_edges_from(input_edges)
        else:
            sm.add_weighted_edges_from(input_edges)

        expected_graph = DynamicStructureModel()
        if not weighted:
            expected_graph.add_edges_from(expected_edges)
        else:
            expected_graph.add_weighted_edges_from(expected_edges)

        assert set(sm.nodes) == set(expected_graph.nodes)
        assert set(sm.edges) == set(expected_graph.edges)

    def test_multi_edge_bad_tuple(self):
        edges = [((0, 0), (1, 0), 0.5), ((1, 0), (2, 0), 0.7, 0.8)]
        sm = DynamicStructureModel()
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Argument {edges[1]} must be a tuple containing 2 or 3 elements"
            ),
        ):
            sm.add_weighted_edges_from(edges)

    def test_single_edge_node_not_tuple(self):
        u = (1, 0)
        v = 3
        sm = DynamicStructureModel()

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Nodes in {(u, v)} must be tuples with node name and time step"
            ),
        ):
            sm.add_edge(u, v)

    @pytest.mark.parametrize(
        "input_edge, expected_edge",
        [
            (
                ((0, 0), (1, 0), 0.5),
                (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0), 0.5),
            ),
            (
                ((0, 0), (1, 0)),
                (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
            ),
            (
                (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
                (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
            ),
            (
                (DynamicStructureNode(0, 0), (1, 0)),
                (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
            ),
            (
                ((0, 0), DynamicStructureNode(1, 0)),
                (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)),
            ),
            (
                (DynamicStructureNode(0, 0), (1, 0), 0.5),
                (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0), 0.5),
            ),
            (
                ((0, 0), DynamicStructureNode(1, 0), 0.5),
                (DynamicStructureNode(0, 0), DynamicStructureNode(1, 0), 0.5),
            ),
        ],
    )
    def test_single_edge_dsn(self, input_edge, expected_edge):
        sm = DynamicStructureModel()
        weighted = len(input_edge) == 3
        if not weighted:
            sm.add_edge(input_edge[0], input_edge[1])
        else:
            sm.add_weighted_edges_from(input_edge)

        expected_graph = DynamicStructureModel()
        if not weighted:
            expected_graph.add_edge(expected_edge[0], expected_edge[1])
        else:
            expected_graph.add_weighted_edges_from(expected_edge)

        assert set(sm.nodes) == set(expected_graph.nodes)
        assert set(sm.edges) == set(expected_graph.edges)

    def test_single_edge_bad_tuple(self):
        edge = ((1, 0), (2, 0), 0.7, 0.8)
        sm = DynamicStructureModel()
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Argument {edge} must be either a DynamicStructureNode or tuple containing 2 or 3 elements"
            ),
        ):
            sm.add_weighted_edges_from(edge)


class TestDynamicStructureModelNodeCoercion:
    @pytest.mark.parametrize(
        "input_nodes, expected_nodes",
        [
            (
                [(0, 0), (1, 0)],
                [DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)],
            ),
            (
                [DynamicStructureNode(0, 0), (1, 0)],
                [DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)],
            ),
            (
                (DynamicStructureNode(n, 0) for n in range(2)),
                [DynamicStructureNode(0, 0), DynamicStructureNode(1, 0)],
            ),
        ],
    )
    def test_multi_node(self, input_nodes, expected_nodes):
        sm = DynamicStructureModel()
        sm.add_nodes(input_nodes)

        expected_graph = DynamicStructureModel()
        expected_graph.add_nodes(expected_nodes)

        assert set(sm.nodes) == set(expected_graph.nodes)

    @pytest.mark.parametrize(
        "input_node, expected_node",
        [
            ((0, 0), DynamicStructureNode(0, 0)),
            (DynamicStructureNode(0, 0), DynamicStructureNode(0, 0)),
        ],
    )
    def test_single_node(self, input_node, expected_node):
        sm = DynamicStructureModel()
        sm.add_nodes(input_node)

        expected_graph = DynamicStructureModel()
        expected_graph.add_nodes(expected_node)

        assert set(sm.nodes) == set(expected_graph.nodes)

    def test_multi_node_bad_tuple(self):
        nodes = [(0, 0), (1, 0, 1)]
        sm = DynamicStructureModel()
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Argument {nodes[1]} must be either a DynamicStructureNode or tuple containing 2 elements"
            ),
        ):
            sm.add_nodes(nodes)

    def test_single_node_bad_tuple(self):
        node = (1, 0, 1)
        sm = DynamicStructureModel()
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"Argument {node} must be either a DynamicStructureNode or tuple containing 2 elements"
            ),
        ):
            sm.add_node(node)
