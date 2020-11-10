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
"""Methods to display styled pygraphgiz plots."""

import io
from typing import Any, Tuple

try:
    from pygraphviz.agraph import AGraph
except ImportError:
    AGraph = Any

try:
    from IPython.display import Image
except ImportError:
    Image = Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError:
    Axes = Any
    Figure = Any


def display_plot_ipython(viz: AGraph, prog: str = "neato") -> Image:
    """
    Displays a pygraphviz object using ipython.

    Args:
        viz: pygraphviz object to render.

        prog: The graph layout. Avaliable are:
        dot, neato, fdp, sfdp, twopi and circo

    Returns:
        IPython Image object. Renders in a notebook.

    Raises:
        ImportError: if IPython is not installed (optional dependency).
    """

    if Image is Any:
        raise ImportError("display_plot_ipython method requires IPython installed.")

    return Image(viz.draw(format="png", prog=prog))


def display_plot_mpl(
    viz: AGraph,
    prog: str = "neato",
    ax: Axes = None,
    pixel_size_in: float = 0.01,
) -> Tuple[Figure, Axes]:
    """
    Displays a pygraphviz object using matplotlib.

    Args:
        viz: pygraphviz object to render.

        prog: The graph layout. Avaliable are:
        dot, neato, fdp, sfdp, twopi and circo

        ax: Optional matplotlib axes to plot on.

        pixel_size_in: Scaling multiple for the plot.

    Returns:
        IPython Image object. Renders in a notebook.

    Raises:
        ImportError: if matplotlib is not installed (optional dependency).
    """

    if Figure is Any:
        raise ImportError("display_plot_mpl method requires matplotlib installed.")

    # bytes:
    s = viz.draw(format="png", prog=prog)
    # convert to numpy array
    array = plt.imread(io.BytesIO(s))
    x_dim, y_dim, _ = array.shape

    # handle passed axis
    if ax is not None:
        ax.imshow(array)
        ax.axis("off")
        return None, ax

    # handle new axis
    f, ax = plt.subplots(1, 1, figsize=(y_dim * pixel_size_in, x_dim * pixel_size_in))
    ax.imshow(array)
    ax.axis("off")
    f.tight_layout(pad=0.0)
    return f, ax
