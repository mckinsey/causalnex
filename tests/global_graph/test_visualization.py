"""Tests for causalnex.global_graph.visualization."""

import pytest

from causalnex.structure import StructureModel
from causalnex.global_graph.visualization import _jsd_to_hex, visualize_impact_heatmap


class TestJsdToHex:
    def test_zero_jsd_returns_blue(self):
        color = _jsd_to_hex(0.0, 1.0)
        assert color.startswith("#")
        # Should be the cool blue end of the gradient
        assert color == "#4a90e2"

    def test_max_jsd_returns_red(self):
        color = _jsd_to_hex(1.0, 1.0)
        assert color.startswith("#")
        # Should be the warm red end
        assert color == "#e24a4a"

    def test_mid_jsd_returns_yellow_ish(self):
        color = _jsd_to_hex(0.5, 1.0)
        assert color.startswith("#")
        # At t=0.5 we hit the yellow peak
        assert color == "#e2d84a"

    def test_zero_max_jsd_returns_blue(self):
        color = _jsd_to_hex(0.0, 0.0)
        assert color == "#4a90e2"


class TestVisualizeImpactHeatmap:
    def test_produces_pyvis_network(self, tmp_path):
        sm = StructureModel()
        sm.add_edge("a", "b", origin="expert")
        sm.add_edge("b", "c", origin="learned")

        impact = [("b", 0.5), ("c", 0.2)]
        output = str(tmp_path / "test.html")

        viz = visualize_impact_heatmap(
            sm,
            impact_scores=impact,
            intervened_nodes=["a"],
            output_path=output,
        )
        # pyvis Network object should have all three nodes
        node_ids = [n["id"] for n in viz.nodes]
        assert set(node_ids) == {"a", "b", "c"}

    def test_intervened_node_styled_red(self, tmp_path):
        sm = StructureModel()
        sm.add_edge("a", "b", origin="expert")

        output = str(tmp_path / "test.html")
        viz = visualize_impact_heatmap(
            sm,
            impact_scores=[("b", 0.3)],
            intervened_nodes=["a"],
            output_path=output,
        )
        a_node = next(n for n in viz.nodes if n["id"] == "a")
        assert a_node["color"]["background"] == "#FF4444"
