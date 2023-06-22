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
from collections import namedtuple

from causalnex.plots import EDGE_STYLE, GRAPH_STYLE, plot_structure
from causalnex.structure import StructureModel

_style = namedtuple("Style", ["WEAK", "NORMAL", "STRONG"])

default_edge_color = EDGE_STYLE.NORMAL["color"]
bgcolor = GRAPH_STYLE["bgcolor"]
cdn_resources = GRAPH_STYLE["cdn_resources"]


class TestToPyvis:
    # pylint: disable=no-member

    def test_all_nodes_exist(self):
        """Both connected and unconnected nodes should exist"""
        sm = StructureModel([("a", "b")])
        sm.add_node("c")
        a_graph = plot_structure(sm)

        assert all(node in a_graph.get_nodes() for node in ["a", "b", "c"])

    def test_all_edges_exist(self):
        """All edges in original graph should exist in graph"""
        edges = [(str(a), str(a + b + 1)) for a in range(2) for b in range(3)]
        sm = StructureModel(edges)
        a_graph = plot_structure(sm)

        a_graph_edges_list = [(ed["from"], ed["to"]) for ed in a_graph.edges]

        assert all(edge in a_graph_edges_list for edge in edges)

    def test_all_node_attributes(self):
        """all node attributes should be set correctly"""
        sm = StructureModel([("a", "b")])
        a_graph = plot_structure(sm)

        default_color = a_graph.get_node("a")["color"]["background"]
        test_color = "black"

        assert default_color != test_color
        assert all(
            a_graph.get_node(node)["color"] != test_color
            for node in a_graph.get_nodes()
        )

        a_graph = plot_structure(sm, all_node_attributes={"color": test_color})
        assert all(
            a_graph.get_node(node)["color"] == test_color
            for node in a_graph.get_nodes()
        )

    def test_all_edge_attributes(self):
        """all edge attributes should be set correctly"""
        sm = StructureModel([("a", "b"), ("b", "c")])
        a_graph = plot_structure(sm)

        test_color = "white"

        assert all(obj["color"] == default_edge_color for obj in a_graph.edges)

        a_graph = plot_structure(sm, all_edge_attributes={"color": test_color})
        assert all(obj["color"] == test_color for obj in a_graph.edges)

    def test_node_attributes(self):
        """specific node attributes should be set correctly"""

        sm = StructureModel([("a", "b"), ("b", "c")])
        a_graph = plot_structure(sm)

        default_color = a_graph.get_node("a")["color"]
        test_color = "white"

        assert default_color != test_color
        assert all(
            a_graph.get_node(node)["color"] == default_color
            for node in a_graph.get_nodes()
        )

        a_graph = plot_structure(sm, node_attributes={"a": {"color": test_color}})
        assert all(
            a_graph.get_node(node)["color"] == default_color
            for node in a_graph.get_nodes()
            if node != "a"
        )
        assert a_graph.get_node("a")["color"] == test_color

    def test_edge_attributes(self):
        """specific edge attributes should be set correctly"""

        sm = StructureModel([("a", "b"), ("b", "c")])
        a_graph = plot_structure(sm)

        default_color = a_graph.get_node("a")["color"]
        test_color = "white"

        assert default_color != test_color
        assert all("color" in obj for obj in a_graph.edges)

        a_graph = plot_structure(
            sm, edge_attributes={("a", "b"): {"color": test_color}}
        )
        assert all(
            edge["color"] == default_edge_color
            for edge in a_graph.edges
            if edge["from"] != "a" and edge["to"] != "b"
        )

        assert all(
            edge["color"] == test_color
            for edge in a_graph.edges
            if edge["from"] == "a" and edge["to"] == "b"
        )

    def test_default_graph_attributes(self):
        """default graph attributes should be set correctly"""

        sm = StructureModel([("a", "b")])

        a_graph = plot_structure(sm)
        assert not a_graph.widget
        assert a_graph.bgcolor == bgcolor
        assert a_graph.cdn_resources == cdn_resources
