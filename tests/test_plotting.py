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
from importlib import reload

import matplotlib.pyplot as plt
import pytest
from IPython.display import Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mock import patch

from causalnex.plots import color_gradient_string, display, plot_structure
from causalnex.structure import StructureModel


class TestToPygraphviz:
    # pylint: disable=no-member

    def test_all_nodes_exist(self):
        """Both connected and unconnected nodes should exist"""
        sm = StructureModel([("a", "b")])
        sm.add_node("c")
        a_graph = plot_structure(sm)

        assert all(node in a_graph.nodes() for node in ["a", "b", "c"])

    def test_all_edges_exist(self):
        """All edges in original graph should exist in pygraphviz graph"""
        edges = [(str(a), str(a + b + 1)) for a in range(2) for b in range(3)]
        sm = StructureModel(edges)
        a_graph = plot_structure(sm)

        assert all(edge in a_graph.edges() for edge in edges)

    def test_has_layout(self):
        """Returned AGraph should have an existing layout"""
        sm = StructureModel([("a", "b")])
        a_graph = plot_structure(sm)
        assert a_graph.has_layout

    def test_all_node_attributes(self):
        """all node attributes should be set correctly"""
        sm = StructureModel([("a", "b")])
        a_graph = plot_structure(sm)

        default_color = a_graph.get_node("a").attr["color"]
        test_color = "black"

        assert default_color != test_color
        assert all(
            a_graph.get_node(node).attr["color"] != test_color
            for node in a_graph.nodes()
        )

        a_graph = plot_structure(sm, all_node_attributes={"color": test_color})
        assert all(
            a_graph.get_node(node).attr["color"] == test_color
            for node in a_graph.nodes()
        )

    def test_all_edge_attributes(self):
        """all edge attributes should be set correctly"""
        sm = StructureModel([("a", "b"), ("b", "c")])
        a_graph = plot_structure(sm)

        default_color = a_graph.get_edge("a", "b").attr["color"]
        test_color = "black"

        assert default_color != test_color
        assert all(
            a_graph.get_edge(u, v).attr["color"] != test_color
            for u, v in a_graph.edges()
        )

        a_graph = plot_structure(sm, all_edge_attributes={"color": test_color})
        assert all(
            a_graph.get_edge(u, v).attr["color"] == test_color
            for u, v in a_graph.edges()
        )

    def test_node_attriibutes(self):
        """specific node attributes should be set correctly"""

        sm = StructureModel([("a", "b"), ("b", "c")])
        a_graph = plot_structure(sm)

        default_color = a_graph.get_node("a").attr["color"]
        test_color = "black"

        assert default_color != test_color
        assert all(
            a_graph.get_node(node).attr["color"] == default_color
            for node in a_graph.nodes()
        )

        a_graph = plot_structure(sm, node_attributes={"a": {"color": test_color}})
        assert all(
            a_graph.get_node(node).attr["color"] == default_color
            for node in a_graph.nodes()
            if node != "a"
        )
        assert a_graph.get_node("a").attr["color"] == test_color

    def test_edge_attriibutes(self):
        """specific edge attributes should be set correctly"""

        sm = StructureModel([("a", "b"), ("b", "c")])
        a_graph = plot_structure(sm)

        default_color = a_graph.get_edge("a", "b").attr["color"]
        test_color = "black"

        assert default_color != test_color
        assert all(
            a_graph.get_edge(u, v).attr["color"] == default_color
            for u, v in a_graph.edges()
        )

        a_graph = plot_structure(
            sm, edge_attributes={("a", "b"): {"color": test_color}}
        )
        assert all(
            a_graph.get_edge(u, v).attr["color"] == default_color
            for u, v in a_graph.edges()
            if (u, v) != ("a", "b")
        )
        assert a_graph.get_edge("a", "b").attr["color"] == test_color

    def test_graph_attributes(self):
        """graph attributes should be set correctly"""

        sm = StructureModel([("a", "b")])

        a_graph = plot_structure(sm)
        assert "label" not in a_graph.graph_attr.keys()

        a_graph = plot_structure(sm, graph_attributes={"label": "test"})
        assert a_graph.graph_attr["label"] == "test"

    def test_prog(self):
        """Layout should be based on given prog"""
        sm = StructureModel([("a", "b")])
        a = plot_structure(sm, prog="neato")
        b = plot_structure(sm, prog="neato")
        c = plot_structure(sm, prog="dot")

        assert str(a) == str(b)
        assert str(a) != str(c)

    @patch("networkx.nx_agraph.to_agraph", side_effect=ImportError())
    def test_install_warning(self, mocked_to_agraph):
        sm = StructureModel()
        with pytest.raises(Warning, match="Pygraphviz not installed"):
            _ = plot_structure(sm)
        mocked_to_agraph.assert_called_once()


class TestColorGradientString:
    def test_starts_with_color(self):
        """string should start with provided colour"""
        s = color_gradient_string("#ffffff33", "#ffffffaa", 30)
        assert s.startswith("#ffffff33")

    def test_ends_with_color(self):
        """string should end with provided colour"""
        s = color_gradient_string("#ffffff33", "#ffffffaa", 1)
        assert s.endswith("#ffffffaa;0.50")

    def test_correct_num_steps(self):
        """string should have the correct number of steps"""
        for steps in range(1, 10):
            s = color_gradient_string("#ffffff33", "#ffffffaa", steps)
            assert s.count(":") == steps

    def test_expected_string(self):
        """should produce the expected reference example"""
        s = color_gradient_string("#00000000", "#99999999", 9)
        expected = ":".join(["#" + str(i) * 8 + ";0.10" for i in range(10)])
        assert s == expected


class TestDisplay:
    def test_display_importerror_ipython(self):
        sm = StructureModel([("a", "b")])
        viz = plot_structure(sm, prog="neato")
        with patch.dict("sys.modules", {"IPython.display": None}):
            reload(display)
            with pytest.raises(
                ImportError,
                match=r"display_plot_ipython method requires IPython installed.",
            ):
                display.display_plot_ipython(viz)
        # NOTE: must reload display again after patch exit
        reload(display)

    def test_display_importerror_mpl(self):
        sm = StructureModel([("a", "b")])
        viz = plot_structure(sm, prog="neato")
        with patch.dict("sys.modules", {"matplotlib": None}):
            reload(display)
            with pytest.raises(
                ImportError,
                match=r"display_plot_mpl method requires matplotlib installed.",
            ):
                display.display_plot_mpl(viz)
        # NOTE: must reload display again after patch exit
        reload(display)

    def test_agraph_import(self):
        with patch.dict("sys.modules", {"pygraphviz.agraph": None}):
            reload(display)
        reload(display)

    def test_return_types_ipython(self):
        sm = StructureModel([("a", "b")])
        viz = plot_structure(sm, prog="neato")
        d = display.display_plot_ipython(viz)
        assert isinstance(d, Image)

    def test_return_types_mpl(self):
        sm = StructureModel([("a", "b")])
        viz = plot_structure(sm, prog="neato")
        d = display.display_plot_mpl(viz)
        assert isinstance(d, tuple)
        assert isinstance(d[0], Figure)
        assert isinstance(d[1], Axes)

        _, ax = plt.subplots()
        d = display.display_plot_mpl(viz, ax=ax)
        assert isinstance(d, tuple)
        assert d[0] is None
        assert isinstance(d[1], Axes)
