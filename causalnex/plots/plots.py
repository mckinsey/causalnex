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
import re
from collections import namedtuple
from copy import deepcopy
from typing import Dict, Tuple

import networkx as nx


def plot_structure(
    sm: nx.DiGraph,
    prog: str = "neato",
    all_node_attributes: Dict[str, str] = None,
    all_edge_attributes: Dict[str, str] = None,
    node_attributes: Dict[str, Dict[str, str]] = None,
    edge_attributes: Dict[Tuple[str, str], Dict[str, str]] = None,
    graph_attributes: Dict[str, str] = None,
):  # pylint: disable=missing-return-type-doc
    """
    Plot a `StructureModel` using pygraphviz.

    Return a pygraphviz graph from a StructureModel. The pygraphgiz graph
    is decorated and laid out so that it can be plotted easily.

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
    4. graph_attributes

    Detailed documentation on available attributes and how they behave is available at:
    https://www.graphviz.org/doc/info/attrs.html

    Default style attributes provided in CausalNex are:

    - causalnex.plots.NODE_STYLE.NORMAL - default node stying
    - causalnex.plots.NODE_STYLE.WEAK - intended for less important nodes in structure
    - causalnex.plots.NODE_STYLE.STRONG - intended for more important nodes in structure

    - causalnex.plots.EDGE_STYLE.NORMAL - default edge stying
    - causalnex.plots.EDGE_STYLE.wEAK - intended for less important edges in structure
    - causalnex.plots.EDGE_STYLE.STRONG - intended for more important edges in structure

    - causalnex.plots.GRAPH_STYLE - default graph styling

    Example:
    ::
        >>> from causalnex.plots import plot_structure
        >>> plot = plot_structure(structure_model)
        >>> plot.draw("plot.png")

    Args:
        sm: structure to plot
        prog: Name of Graphviz layout program
        all_node_attributes: attributes to apply to all nodes
        all_edge_attributes: attrinbutes to apply to all edges
        node_attributes: attributes to apply to specific nodes
        edge_attributes: attributes to apply to specific edges
        graph_attributes: attributes to apply to the graph

    Returns:
        a styled pygraphgiz graph that can be rendered as an image

    Raises:
        Warning: Suggests mitigation strategies when ``pygraphviz`` is not installed.
    """

    # apply node and edge attributes
    _sm = _add_attributes(
        sm, all_node_attributes, all_edge_attributes, node_attributes, edge_attributes
    )

    # create plot
    try:
        a_graph = nx.nx_agraph.to_agraph(_sm)
    except ImportError as error_msg:
        raise Warning(
            """
            Pygraphviz not installed. Also make sure you have the system-level
            ``graphviz`` requirement installed.

            Alternatively, you can visualise your graph using the networkx.draw
            functionality:
            >>> sm = StructureModel()
            >>> fig, ax = plt.subplots()
            >>> nx.draw_circular(sm, ax=ax)
            >>> fig.show()
            """
        ) from error_msg

    # apply graph attributes
    a_graph.graph_attr.update(GRAPH_STYLE)
    if graph_attributes:
        a_graph.graph_attr.update(graph_attributes)

    # layout and return
    a_graph.layout(prog=prog)
    return a_graph


def color_gradient_string(from_color: str, to_color: str, steps: int) -> str:
    """
    Create a pygraphgiz compatible color gradient string.

    This string can be used when setting colors for nodes,
    edges, and graph attributes.

    Example:
    ::
        >>> node_attributes = {
        >>>    "color": color_gradient_string(
        >>>        from_color="#000000", to_color="#FFFFFF", steps=30
        >>>    )
        >>> }

    Args:
        from_color: rgb(a) string of color to start gradient from
        to_color: rgb(a) string of color to end gradient at
        steps: number of steps in the gradient string. steps=1 produces from_color:to_color
        without any intermediary steps

    Returns:
        a pygraphviz color gradient string
    """

    color_regex = re.compile(
        r"(#)([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})*"
    )

    from_colors = [
        int(v, 16) if v else 0 for v in color_regex.match(from_color).groups()[1:]
    ]
    to_colors = [
        int(v, 16) if v else 0 for v in color_regex.match(to_color).groups()[1:]
    ]

    delta_colors = [(t - f) / steps for f, t in zip(from_colors, to_colors)]

    gradient_colors = [
        "#"
        + "".join(
            [format(int(f + d * i), "02x") for f, d in zip(from_colors, delta_colors)]
        )
        for i in range(steps + 1)
    ]

    return ":".join(
        [f"{gradient_colors[i]};{1 / (steps + 1):.2f}" for i in range(steps + 1)]
    )


def _add_attributes(
    sm: nx.DiGraph,
    all_node_attributes: Dict[str, str] = None,
    all_edge_attributes: Dict[str, str] = None,
    node_attributes: Dict[str, Dict[str, str]] = None,
    edge_attributes: Dict[str, Dict[str, str]] = None,
) -> nx.DiGraph:
    _sm = deepcopy(sm)

    # shift labels to be above nodes
    for node in _sm.nodes:
        _sm.nodes[node]["label"] = f"{node}\n\n "

    # apply node attributes (start with default, then apply any custom)
    _all_node_attr = {**NODE_STYLE.NORMAL}
    if all_node_attributes:
        _all_node_attr.update(all_node_attributes)

    for k, v in _all_node_attr.items():
        nx.set_node_attributes(_sm, v, k)

    # apply edge attributes (start with default, then apply any custom)
    _all_edge_attr = {**EDGE_STYLE.NORMAL}
    if all_edge_attributes:
        _all_edge_attr.update(all_edge_attributes)

    for k, v in _all_edge_attr.items():
        nx.set_edge_attributes(_sm, v, k)

    # apply specific node and edge attributes
    if node_attributes:
        nx.set_node_attributes(_sm, node_attributes)
    if edge_attributes:
        nx.set_edge_attributes(_sm, edge_attributes)

    return _sm


GRAPH_STYLE = {
    "bgcolor": "#001521",
    "fontcolor": "#FFFFFFD9",
    "fontname": "Helvetica",
    "splines": True,
    "overlap": "scale",
    "scale": 2.0,
    "pad": "0.8,0.3",
    "dpi": 300,
}

_style = namedtuple("Style", ["WEAK", "NORMAL", "STRONG"])

NODE_STYLE = _style(
    {
        "fontcolor": "#FFFFFF8c",
        "fontname": "Helvetica",
        "shape": "circle",
        "fixedsize": True,
        "style": "filled",
        "fillcolor": "#4a90e2d9",
        "color": "#FFFFFFD9",
        "width": 0.05,
        "penwidth": "1",
        "fontsize": 10,
    },
    {
        "fontcolor": "#FFFFFFD9",
        "fontname": "Helvetica",
        "shape": "circle",
        "fixedsize": True,
        "style": "filled",
        "fillcolor": "#4a90e2d9",
        "color": "#4a90e220",
        "width": 0.15,
        "penwidth": "20",
        "fontsize": 15,
    },
    {
        "fontcolor": "#4a90e2",
        "fontname": "Helvetica",
        "shape": "circle",
        "fixedsize": True,
        "style": "filled",
        "fillcolor": "#4a90e2d9",
        "color": "#4a90e2",
        "width": 0.15,
        "penwidth": "4",
        "fontsize": 20,
    },
)

EDGE_STYLE = _style(
    {
        "color": color_gradient_string("#FFFFFF33", "#ffffffaa", 30),
        "arrowhead": "normal",
        "penwidth": 0.25,
        "arrowsize": 0.4,
    },
    {
        "color": color_gradient_string("#FFFFFF33", "#ffffffaa", 30),
        "arrowhead": "normal",
        "penwidth": 1,
        "arrowsize": 0.8,
    },
    {
        "color": color_gradient_string("#FFFFFF33", "#1F78B4aa", 30),
        "arrowhead": "normal",
        "penwidth": 3,
        "arrowsize": 1,
    },
)
