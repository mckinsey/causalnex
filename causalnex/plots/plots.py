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
"""Plot Methods."""
from collections import namedtuple
from typing import Dict, Tuple

import networkx as nx
from pyvis.network import Network


def plot_structure(
    sm: nx.DiGraph,
    all_node_attributes: Dict[str, str] = None,
    all_edge_attributes: Dict[str, str] = None,
    node_attributes: Dict[str, Dict[str, str]] = None,
    edge_attributes: Dict[Tuple[str, str], Dict[str, str]] = None,
    plot_options: Dict[str, str] = None,
) -> Network:
    """
    Plot a `StructureModel` using pyvis.

    Return a pyvis graph object - which can be visualized using the pyvis method
    `.show(name)`

    Default node, edge, and graph attributes are provided to style and layout
    the plot. These defaults can be overridden for all nodes and edges through
    `all_node_attributes` and `all_edge_attributes` respectively. Graph
    attributes can be overridden through `graph_attributes`.

    Styling and layout attributes can be set for individual nodes and edges
    through `node_attributes` and `edge_attributes` respectively.

    Attributes are set in the following order, overriding any previously set attributes
    1. default attributes
    2. all_node_attributes and all_edge_attributes
    3. node_attributes and edge_attributes

    Detailed documentation on available attributes and how they behave is available at:
    https://visjs.github.io/vis-network/docs/network/.

    Default style attributes provided in CausalNex are:

    - causalnex.plots.GRAPH_STYLE - defaults graph styling

    - causalnex.plots.NODE_STYLE.NORMAL - default node styling
    - causalnex.plots.NODE_STYLE.WEAK - intended for less important nodes in structure
    - causalnex.plots.NODE_STYLE.STRONG - intended for more important nodes in structure

    - causalnex.plots.EDGE_STYLE.NORMAL - default edge styling
    - causalnex.plots.EDGE_STYLE.WEAK - intended for less important edges in structure
    - causalnex.plots.EDGE_STYLE.STRONG - intended for more important edges in structure
    Example:
    ::
        >>> from causalnex.plots import plot_structure
        >>> plot = plot_structure(structure_model)
    Args:
        sm: structure to plot
        all_node_attributes: attributes to apply to all nodes
        all_edge_attributes: attrinbutes to apply to all edges
        node_attributes: attributes to apply to specific nodes
        edge_attributes: attributes to apply to specific edges
        plot_options: attributes to apply to pyvis plotting function

    Returns:
        a styled pyvis graph that can be rendered externally as an HTML object

    """
    node_attributes = node_attributes or {}
    edge_attributes = edge_attributes or {}
    all_edge_attributes = all_edge_attributes or {}
    all_node_attributes = all_node_attributes or {}
    plot_options = plot_options or {}

    pyvis_graph = Network(**{**GRAPH_STYLE, **plot_options})

    all_node_attributes = {**NODE_STYLE.NORMAL, **all_node_attributes}
    all_edge_attributes = {**EDGE_STYLE.NORMAL, **all_edge_attributes}

    for node, _ in sm.nodes(data=True):
        pyvis_graph.add_node(
            node,
            **{
                **all_node_attributes,
                "label": f"{node}",
                **node_attributes.get(node, {}),
            },
        )

    # for each edge and its attributes in the networkx graph
    for source, target in sm.edges:
        pyvis_graph.add_edge(
            source,
            target,
            **{**all_edge_attributes, **edge_attributes.get((source, target), {})},
        )

    return pyvis_graph


GRAPH_STYLE = {
    "height": "600px",
    "width": "100%",
    "heading": "",
    "notebook": True,
    "bgcolor": "#001521",
    "cdn_resources": "in_line",
    "directed": True,
}


_style = namedtuple("Style", ["WEAK", "NORMAL", "STRONG"])

NODE_STYLE = _style(
    {
        "font": {"color": "#FFFFFF8c", "face": "Helvetica", "size": 25},
        "shape": "dot",
        "size": 15,
        "borderWidth": 5,
        "color": {"border": "#FFFFFFD9", "background": "#4a90e2d9"},
        "mass": 1,
    },
    {
        "font": {"color": "#FFFFFFD9", "face": "Helvetica", "size": 40},
        "shape": "dot",
        "size": 25,
        "borderWidth": 5,
        "color": {"border": "#4a90e220", "background": "#4a90e2d9"},
        "mass": 1.3,
    },
    {
        "font": {"color": "#4a90e2", "face": "Helvetica", "size": 60},
        "shape": "dot",
        "size": 35,
        "borderWidth": 5,
        "color": {"border": "#4a90e2", "background": "#4a90e2d9"},
        "mass": 2,
    },
)

EDGE_STYLE = _style(
    {
        "color": "#ffffffaa",
        "arrows": {
            "to": {
                "enabled": True,
                "scaleFactor": 0.5,
            },
        },
        "endPointOffset": {"from": 1, "to": -1},
        "width": 2,
        "length": 15,
    },
    {
        "color": "#ffffffaa",
        "arrows": {
            "to": {
                "enabled": True,
                "scaleFactor": 1.0,
            },
        },
        "endPointOffset": {"from": 1, "to": -1},
        "width": 5,
        "length": 300,
    },
    {
        "color": "#1F78B4aa",
        "arrows": {
            "to": {
                "enabled": True,
                "scaleFactor": 1.5,
            },
        },
        "endPointOffset": {"from": 1, "to": -1},
        "width": 30,
        "length": 500,
    },
)
