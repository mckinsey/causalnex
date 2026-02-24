"""Tests for causalnex.global_graph.structure_builder."""

import networkx as nx
import pytest

from causalnex.structure import StructureModel
from causalnex.global_graph.structure_builder import (
    DEFAULT_EXPERT_EDGES,
    build_expert_skeleton,
    merge_temporal_into_static,
    enforce_dag,
)


class TestBuildExpertSkeleton:
    def test_returns_structure_model(self):
        sm = build_expert_skeleton()
        assert isinstance(sm, StructureModel)

    def test_default_edges_are_present(self):
        sm = build_expert_skeleton()
        for u, v in DEFAULT_EXPERT_EDGES:
            assert sm.has_edge(u, v), f"Missing edge {u} -> {v}"

    def test_all_edges_are_expert_origin(self):
        sm = build_expert_skeleton()
        for u, v, data in sm.edges(data=True):
            assert data["origin"] == "expert"

    def test_default_skeleton_is_dag(self):
        sm = build_expert_skeleton()
        assert nx.is_directed_acyclic_graph(sm)

    def test_custom_edges(self):
        edges = [("a", "b"), ("b", "c")]
        sm = build_expert_skeleton(edges=edges)
        assert list(sm.edges) == [("a", "b"), ("b", "c")]

    def test_invalid_node_name_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            build_expert_skeleton(edges=[("bad-name", "ok_name")])


class TestMergeTemporalIntoStatic:
    def test_adds_learned_edges(self):
        expert = StructureModel()
        expert.add_edge("a", "b", origin="expert")
        expert.add_edge("b", "c", origin="expert")

        temporal = StructureModel()
        temporal.add_edge("a_lag1", "c_lag0")

        merged = merge_temporal_into_static(expert, temporal)
        assert merged.has_edge("a", "c")
        assert merged["a"]["c"]["origin"] == "learned"

    def test_does_not_add_intra_slice(self):
        expert = StructureModel()
        expert.add_edge("a", "b", origin="expert")

        temporal = StructureModel()
        temporal.add_edge("a_lag0", "b_lag0")

        merged = merge_temporal_into_static(expert, temporal)
        # Edge already existed as expert — intra-slice should not duplicate
        assert merged["a"]["b"]["origin"] == "expert"

    def test_does_not_add_reverse_edge(self):
        expert = StructureModel()
        expert.add_edge("a", "b", origin="expert")

        temporal = StructureModel()
        temporal.add_edge("b_lag1", "a_lag0")

        merged = merge_temporal_into_static(expert, temporal)
        assert not merged.has_edge("b", "a")

    def test_ignores_nodes_not_in_expert(self):
        expert = StructureModel()
        expert.add_edge("a", "b", origin="expert")

        temporal = StructureModel()
        temporal.add_edge("x_lag1", "y_lag0")

        merged = merge_temporal_into_static(expert, temporal)
        assert "x" not in merged.nodes
        assert "y" not in merged.nodes


class TestEnforceDag:
    def test_already_dag_unchanged(self):
        sm = StructureModel()
        sm.add_edge("a", "b", origin="expert")
        sm.add_edge("b", "c", origin="expert")

        result = enforce_dag(sm)
        assert nx.is_directed_acyclic_graph(result)
        assert result.has_edge("a", "b")
        assert result.has_edge("b", "c")

    def test_removes_learned_edge_in_cycle(self):
        sm = StructureModel()
        sm.add_edge("a", "b", origin="expert")
        sm.add_edge("b", "c", origin="expert")
        sm.add_edge("c", "a", origin="learned", weight=0.5)

        result = enforce_dag(sm)
        assert nx.is_directed_acyclic_graph(result)
        # Expert edges should survive
        assert result.has_edge("a", "b")
        assert result.has_edge("b", "c")

    def test_handles_all_expert_cycle(self):
        sm = StructureModel()
        sm.add_edge("a", "b", origin="expert", weight=1.0)
        sm.add_edge("b", "c", origin="expert", weight=0.5)
        sm.add_edge("c", "a", origin="expert", weight=0.1)

        result = enforce_dag(sm)
        assert nx.is_directed_acyclic_graph(result)
