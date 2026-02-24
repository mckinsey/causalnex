"""
Impact heatmap visualization for the global relationship graph.

Maps JSD impact scores to node color and size in an interactive PyVis
graph rendered via ``causalnex.plots.plot_structure``.

Color scheme:

* **Intervened nodes** — bright red (#FF4444)
* **High impact (JSD > 0.3)** — warm red/orange gradient
* **Medium impact (0.1–0.3)** — yellow
* **Low / no impact (< 0.1)** — cool blue (default)
"""

from typing import Dict, List, Optional, Tuple

from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure import StructureModel


def _jsd_to_hex(jsd: float, max_jsd: float) -> str:
    """Map a JSD score to a hex colour on a blue→yellow→red gradient.

    Falls back to a simple linear interpolation so that ``matplotlib`` is
    not a hard requirement.
    """
    if max_jsd <= 0:
        return "#4a90e2"

    t = min(jsd / max_jsd, 1.0)  # normalise to [0, 1]

    if t < 0.5:
        # Blue (#4a90e2) → Yellow (#e2d84a)
        s = t / 0.5
        r = int(0x4A + (0xE2 - 0x4A) * s)
        g = int(0x90 + (0xD8 - 0x90) * s)
        b = int(0xE2 + (0x4A - 0xE2) * s)
    else:
        # Yellow (#e2d84a) → Red (#e24a4a)
        s = (t - 0.5) / 0.5
        r = int(0xE2)
        g = int(0xD8 + (0x4A - 0xD8) * s)
        b = int(0x4A)

    return f"#{r:02x}{g:02x}{b:02x}"


def visualize_impact_heatmap(
    sm: StructureModel,
    impact_scores: List[Tuple[str, float]],
    intervened_nodes: List[str],
    output_path: str = "global_ripple.html",
    node_registry: Optional[Dict[str, Dict]] = None,
) -> object:
    """Render the causal graph with nodes coloured by JSD impact score.

    Args:
        sm: the ``StructureModel`` defining the DAG edges.
        impact_scores: list of ``(node_name, jsd_score)`` as returned by
            ``compute_ripple_effect`` or ``compound_ripple_effect``.
        intervened_nodes: node names that were the intervention triggers.
        output_path: file path for the generated HTML visualisation.
        node_registry: optional node metadata dict for richer labels.

    Returns:
        The ``pyvis.network.Network`` object (also saved to *output_path*).
    """
    node_registry = node_registry or {}
    impact_dict = dict(impact_scores)
    max_jsd = max(impact_dict.values()) if impact_dict else 1.0

    node_attrs: Dict[str, dict] = {}
    for node in sm.nodes:
        label = node_registry.get(node, {}).get("label", node)

        if node in intervened_nodes:
            node_attrs[node] = {
                "color": {"border": "#FF0000", "background": "#FF4444"},
                "size": 45,
                "font": {"color": "#FFFFFF", "face": "Helvetica", "size": 60},
                "label": f"[EVENT] {label}",
            }
        elif node in impact_dict:
            jsd = impact_dict[node]
            hex_color = _jsd_to_hex(jsd, max_jsd)
            size = int(20 + 20 * (jsd / max_jsd if max_jsd > 0 else 0))
            node_attrs[node] = {
                "color": {"border": hex_color, "background": hex_color},
                "size": size,
                "font": {"color": "#FFFFFF", "face": "Helvetica", "size": 35},
                "label": label,
                "title": f"{label}\nJSD Impact: {jsd:.4f}",
            }
        else:
            node_attrs[node] = {
                **NODE_STYLE.WEAK,
                "label": label,
            }

    edge_attrs: Dict[Tuple[str, str], dict] = {}
    for u, v, data in sm.edges(data=True):
        if data.get("origin") == "expert":
            edge_attrs[(u, v)] = EDGE_STYLE.STRONG
        elif data.get("origin") == "learned":
            edge_attrs[(u, v)] = EDGE_STYLE.NORMAL
        else:
            edge_attrs[(u, v)] = EDGE_STYLE.WEAK

    viz = plot_structure(
        sm,
        all_node_attributes=NODE_STYLE.NORMAL,
        all_edge_attributes=EDGE_STYLE.NORMAL,
        node_attributes=node_attrs,
        edge_attributes=edge_attrs,
        plot_options={
            "height": "800px",
            "width": "100%",
            "bgcolor": "#0a0a1a",
        },
    )
    viz.show(output_path)
    return viz
