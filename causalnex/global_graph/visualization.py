"""
Impact heatmap visualization for the global relationship graph.

Maps JSD impact scores to node color and size in an interactive PyVis
graph rendered via ``causalnex.plots.plot_structure``.

When an :class:`~causalnex.global_graph.registry.OntologyRegistry` is
provided, nodes are shaped and base-coloured by entity group.

Color scheme:

* **Intervened nodes** — bright red (#FF4444)
* **High impact (JSD > 0.3)** — warm red/orange gradient
* **Medium impact (0.1–0.3)** — yellow
* **Low / no impact (< 0.1)** — cool blue (default)
"""

from typing import Dict, List, Optional, Tuple

from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure import StructureModel
from causalnex.global_graph.registry import OntologyRegistry


# Default colour palette for entity group categories.
# Keys match the ``category`` field in ``node_types.yaml``.
_CATEGORY_COLORS = {
    "geopolitics": "#5B8DEF",   # blue
    "economics": "#4ECB71",     # green
    "infrastructure": "#F5A623", # orange
    "security": "#D0021B",      # red
    "social": "#BD10E0",        # purple
    "governance": "#7ED321",    # lime
    "events": "#F8E71C",        # yellow
    "environment": "#50E3C2",   # teal
}

# Map entity group categories to pyvis node shapes.
_CATEGORY_SHAPES = {
    "geopolitics": "dot",
    "economics": "diamond",
    "infrastructure": "square",
    "security": "triangle",
    "social": "star",
    "governance": "hexagon",
    "events": "triangleDown",
    "environment": "ellipse",
}


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


def _get_category(node_id: str, registry: Optional[OntologyRegistry]) -> str:
    """Look up the category for a node via its type in the registry."""
    if not registry:
        return ""
    try:
        node_meta = registry.get_node(node_id)
        type_id = node_meta.get("type", "")
        type_meta = registry.get_node_type(type_id)
        return type_meta.get("category", "")
    except KeyError:
        return ""


def visualize_impact_heatmap(
    sm: StructureModel,
    impact_scores: List[Tuple[str, float]],
    intervened_nodes: List[str],
    output_path: str = "global_ripple.html",
    node_registry: Optional[Dict[str, Dict]] = None,
    registry: Optional[OntologyRegistry] = None,
) -> object:
    """Render the causal graph with nodes coloured by JSD impact score.

    Args:
        sm: the ``StructureModel`` defining the DAG edges.
        impact_scores: list of ``(node_name, jsd_score)`` as returned by
            ``compute_ripple_effect`` or ``compound_ripple_effect``.
        intervened_nodes: node names that were the intervention triggers.
        output_path: file path for the generated HTML visualisation.
        node_registry: **deprecated** — use *registry* instead.
        registry: an :class:`OntologyRegistry` for richer labels and
            category-based styling.

    Returns:
        The ``pyvis.network.Network`` object (also saved to *output_path*).
    """
    node_registry = node_registry or {}
    impact_dict = dict(impact_scores)
    max_jsd = max(impact_dict.values()) if impact_dict else 1.0

    node_attrs: Dict[str, dict] = {}
    for node in sm.nodes:
        # Resolve label
        label = node
        if registry:
            try:
                label = registry.get_node(node).get("label", node)
            except KeyError:
                pass
        else:
            label = node_registry.get(node, {}).get("label", node)

        category = _get_category(node, registry)
        shape = _CATEGORY_SHAPES.get(category, "dot")

        if node in intervened_nodes:
            node_attrs[node] = {
                "color": {"border": "#FF0000", "background": "#FF4444"},
                "size": 45,
                "shape": shape,
                "font": {"color": "#FFFFFF", "face": "Helvetica", "size": 60},
                "label": f"[EVENT] {label}",
            }
        elif node in impact_dict:
            jsd = impact_dict[node]
            hex_color = _jsd_to_hex(jsd, max_jsd)
            size = int(20 + 20 * (jsd / max_jsd if max_jsd > 0 else 0))

            # Build tooltip with type info
            type_label = ""
            if registry:
                try:
                    ntype = registry.get_node(node).get("type", "")
                    type_label = registry.get_node_type(ntype).get("label", "")
                except KeyError:
                    pass

            tooltip = f"{label}\nJSD Impact: {jsd:.4f}"
            if type_label:
                tooltip = f"[{type_label}]\n{tooltip}"

            node_attrs[node] = {
                "color": {"border": hex_color, "background": hex_color},
                "size": size,
                "shape": shape,
                "font": {"color": "#FFFFFF", "face": "Helvetica", "size": 35},
                "label": label,
                "title": tooltip,
            }
        else:
            node_attrs[node] = {
                **NODE_STYLE.WEAK,
                "shape": shape,
                "label": label,
            }

    # Edge styling: typed edges get labels
    edge_attrs: Dict[Tuple[str, str], dict] = {}
    for u, v, data in sm.edges(data=True):
        style = dict(EDGE_STYLE.NORMAL)
        if data.get("origin") == "expert":
            style = dict(EDGE_STYLE.STRONG)
        elif data.get("origin") == "learned":
            style = dict(EDGE_STYLE.NORMAL)
        else:
            style = dict(EDGE_STYLE.WEAK)

        # Add edge type as a label/tooltip if available
        edge_type = data.get("edge_type", "")
        if edge_type and registry:
            try:
                et_label = registry.get_edge_type(edge_type).get("label", edge_type)
            except KeyError:
                et_label = edge_type
            style["title"] = et_label
            desc = data.get("description", "")
            if desc:
                style["title"] = f"{et_label}: {desc}"

        edge_attrs[(u, v)] = style

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
