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
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from causalnex.structure.structuremodel import StructureModel


def _setup_plot(ax: plt.Axes = None, title: str = None) -> (plt.Figure, plt.Axes):
    """Initial setup of fig and ax to plot to."""

    if not ax:
        fig = plt.figure()  # type: plt.Figure
        ax = fig.add_subplot(1, 1, 1)  # type: plt.Axes

    if title:
        ax.set_title(title)

    return ax.get_figure(), ax


def plot_structure(
    g: StructureModel,
    ax: plt.Axes = None,
    title: str = None,
    show_labels: bool = True,
    node_color: str = "r",
    edge_color: str = "k",
    label_color: str = "k",
    node_positions: Dict[str, List[float]] = None,
) -> Tuple[Figure, Axes, Dict[str, List[float]]]:
    """Plot the structure model to visualise the relationships between nodes.

    Args:
        g: the structure model to plot.
        ax: if provided then figure will be drawn to this Axes, otherwise a new Axes will be created.
        title: if provided then the title will be drawn on the plot.
        show_labels: if True then node labels will be drawn.
        node_color: a single color format string, for example 'r' or '#ff0000'. default "r".
        edge_color: a single color format string, for example 'r' or '#ff0000'. default "k".
        label_color: a single color format string, for example 'r' or '#ff0000'. default "k".
        node_positions: coordinates for node positions, ie {"node_a": [0, 0]}.

    Returns:
        fig, ax, node_positions.

    Example:
    ::
        >>> # Create a Bayesian Network with a manually defined DAG.
        >>> from causalnex.structure import StructureModel
        >>> from causalnex.network import BayesianNetwork
        >>>
        >>> sm = StructureModel()
        >>> sm.add_edges_from([
        >>>                    ('rush_hour', 'traffic'),
        >>>                    ('weather', 'traffic')
        >>>                    ])
        >>> from causalnex.plots import plot_structure
        >>> plot_structure(sm)
    """

    fig, ax = _setup_plot(ax, title)

    if not node_positions:
        node_positions = nx.circular_layout(g)

    node_color = node_color if node_color else "r"
    edge_color = edge_color if edge_color else "k"
    label_color = label_color if label_color else "k"

    nx.draw_networkx_nodes(
        g, node_positions, ax=ax, nodelist=g.nodes, node_color=node_color
    )

    for u, v in g.edges:
        nx.draw_networkx_edges(
            g, node_positions, ax=ax, edgelist=[(u, v)], edge_color=edge_color
        )

    if show_labels:
        nx.draw_networkx_labels(g, node_positions, ax=ax, font_color=label_color)

    ax.set_axis_off()
    plt.tight_layout()

    return fig, ax, node_positions
