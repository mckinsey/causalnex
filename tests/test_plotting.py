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

from string import ascii_lowercase

import matplotlib as plt
import pytest
from matplotlib.colors import to_rgba

from causalnex.plots import plot_structure
from causalnex.structure import StructureModel


class TestPlotStructure:
    """Test behaviour of plot structure method"""

    @pytest.mark.parametrize(
        "test_input,expected", [(None, ""), ("", ""), ("TEST", "TEST")]
    )
    def test_title(self, test_input, expected):
        """Title should be set correctly"""
        sm = StructureModel([("a", "b")])
        _, ax, _ = plot_structure(sm, title=test_input)
        assert ax.get_title() == expected

    def test_edges_exist(self):
        """All edges should exist"""

        for num_nodes in range(2, 10):
            nodes = [c for i, c in enumerate(ascii_lowercase) if i < num_nodes]
            sm = StructureModel(list(zip(nodes[:-1], nodes[1:])))
            _, ax, _ = plot_structure(sm)
            ax_edges = [
                patch
                for patch in ax.patches
                if isinstance(patch, plt.patches.FancyArrowPatch)
            ]
            assert len(ax_edges) == num_nodes - 1

    @pytest.mark.parametrize(
        "test_input,expected",
        [("#123456", to_rgba("#123456")), ("blue", to_rgba("blue"))],
    )
    def test_edge_color(self, test_input, expected):
        """Edge color should be set if given"""
        sm = StructureModel([("a", "b")])
        _, ax, _ = plot_structure(sm, edge_color=test_input)
        ax_edges = [
            patch
            for patch in ax.patches
            if isinstance(patch, plt.patches.FancyArrowPatch)
        ]
        assert ax_edges[0].get_edgecolor() == expected

    def test_nodes_exist(self):
        """All nodes should exist"""

        for num_nodes in range(2, 10):
            nodes = [c for i, c in enumerate(ascii_lowercase) if i < num_nodes]
            sm = StructureModel(list(zip(nodes[:-1], nodes[1:])))
            _, ax, _ = plot_structure(sm)
            ax_nodes = ax.collections[0].get_offsets()
            assert len(ax_nodes) == num_nodes

    @pytest.mark.parametrize(
        "input_positions,expected_positions",
        [({"a": [1, 1], "b": [2, 2]}, [[1.0, 1.0], [2.0, 2.0]])],
    )
    def test_node_positions_respected(self, input_positions, expected_positions):
        """Nodes should be at the positions provided"""
        sm = StructureModel([("a", "b")])
        _, ax, _ = plot_structure(sm, node_positions=input_positions)
        node_coords = [list(coord) for coord in ax.collections[0].get_offsets()]
        assert all(
            [
                node_x == exp_x and node_y == exp_y
                for ((exp_x, exp_y), (node_x, node_y)) in zip(
                    expected_positions, sorted(node_coords)
                )
            ]
        )

    @pytest.mark.parametrize(
        "test_input,expected",
        [("#123456", to_rgba("#123456")), ("blue", to_rgba("blue"))],
    )
    def test_node_color(self, test_input, expected):
        """Node color should be set if given"""
        sm = StructureModel([("a", "b")])
        _, ax, _ = plot_structure(sm, node_color=test_input)
        assert all(
            all(face_color == expected)
            for face_color in ax.collections[0].get_facecolors()
        )

    @pytest.mark.parametrize("test_input,expected", [(False, False), (True, True)])
    def test_show_labels(self, test_input, expected):
        """Labels should be hidden when show_labels set to False"""
        sm = StructureModel([("a", "b")])
        _, ax, _ = plot_structure(sm, show_labels=test_input)

        assert bool(ax.texts) == expected

    @pytest.mark.parametrize(
        "test_input,expected", [("r", "r"), ("#123456", "#123456")]
    )
    def test_label_colors(self, test_input, expected):
        """Labels should have color provided to them"""
        sm = StructureModel([("a", "b")])
        _, ax, _ = plot_structure(sm, show_labels=True, label_color=test_input)
        assert all(text.get_color() == expected for text in ax.texts)
