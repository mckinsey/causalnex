"""Tests for the config-driven ontology registry and expanded global graph."""

import os
import shutil

import pytest

from causalnex.global_graph.registry import (
    OntologyRegistry,
    YAMLBackend,
    InMemoryBackend,
    get_default_registry,
    validate_node_name,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def default_registry():
    """Load the registry from the shipped YAML configs."""
    return get_default_registry()


@pytest.fixture
def empty_registry():
    """An empty in-memory registry."""
    return OntologyRegistry(InMemoryBackend())


@pytest.fixture
def tmp_yaml_dir(tmp_path):
    """Copy the default configs into a temp dir for write tests."""
    config_src = os.path.join(
        os.path.dirname(__file__),
        "..",
        "causalnex",
        "global_graph",
        "config",
    )
    config_dst = tmp_path / "config"
    shutil.copytree(config_src, str(config_dst))
    return str(config_dst)


# -------------------------------------------------------------------------
# Tests: validate_node_name
# -------------------------------------------------------------------------


class TestValidateNodeName:
    def test_valid_names(self):
        assert validate_node_name("sov_usa") is True
        assert validate_node_name("fig_biden") is True
        assert validate_node_name("res_oil_brent") is True
        assert validate_node_name("node123") is True

    def test_invalid_names(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_node_name("has space")
        with pytest.raises(ValueError, match="invalid"):
            validate_node_name("has-hyphen")
        with pytest.raises(ValueError, match="invalid"):
            validate_node_name("")


# -------------------------------------------------------------------------
# Tests: Loading default config
# -------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_loads_node_types(self, default_registry):
        nt = default_registry.node_types
        assert "sovereign_state" in nt
        assert "commodity" in nt
        assert "climate" in nt
        assert "social_movement" in nt
        assert len(nt) >= 20

    def test_node_type_has_required_fields(self, default_registry):
        for type_id, meta in default_registry.node_types.items():
            assert "label" in meta, f"{type_id} missing label"
            assert "prefix" in meta, f"{type_id} missing prefix"
            assert "category" in meta, f"{type_id} missing category"

    def test_loads_edge_types(self, default_registry):
        et = default_registry.edge_types
        assert "exports_to" in et
        assert "disrupts" in et
        assert "triggers" in et
        assert "sets_rate_for" in et
        assert len(et) >= 20

    def test_loads_nodes(self, default_registry):
        nodes = default_registry.nodes
        assert "sov_usa" in nodes
        assert "fig_biden" in nodes
        assert "res_oil_brent" in nodes
        assert "cur_usd" in nodes
        assert "cb_fed" in nodes
        assert "route_hormuz" in nodes
        assert "mil_us_forces" in nodes
        assert "clim_el_nino" in nodes
        assert len(nodes) >= 80

    def test_loads_edges(self, default_registry):
        edges = default_registry.edges
        assert len(edges) >= 80
        # Check a known edge exists
        sources = {e["source"] for e in edges}
        targets = {e["target"] for e in edges}
        assert "fig_putin" in sources
        assert "evt_ukraine_war" in targets

    def test_validation_passes(self, default_registry):
        warnings = default_registry.validate()
        assert warnings == [], f"Validation warnings: {warnings}"

    def test_repr(self, default_registry):
        r = repr(default_registry)
        assert "OntologyRegistry(" in r
        assert "node types" in r
        assert "edge types" in r


# -------------------------------------------------------------------------
# Tests: Queries
# -------------------------------------------------------------------------


class TestRegistryQueries:
    def test_get_node_type(self, default_registry):
        meta = default_registry.get_node_type("sovereign_state")
        assert meta["prefix"] == "sov"

    def test_get_node_type_missing(self, default_registry):
        with pytest.raises(KeyError):
            default_registry.get_node_type("nonexistent_type")

    def test_prefix_for_type(self, default_registry):
        assert default_registry.prefix_for_type("commodity") == "res"
        assert default_registry.prefix_for_type("currency") == "cur"

    def test_type_for_prefix(self, default_registry):
        assert default_registry.type_for_prefix("sov") == "sovereign_state"
        assert default_registry.type_for_prefix("fig") == "political_figure"
        assert default_registry.type_for_prefix("zzz") is None

    def test_get_nodes_by_type(self, default_registry):
        states = default_registry.get_nodes_by_type("sovereign_state")
        assert "sov_usa" in states
        assert "sov_china" in states
        assert "fig_biden" not in states

    def test_get_nodes_by_domain(self, default_registry):
        forex = default_registry.get_nodes_by_domain("forex")
        assert "cur_usd" in forex
        assert "cur_eur" in forex

    def test_get_edges_from(self, default_registry):
        edges = default_registry.get_edges_from("fig_putin")
        assert len(edges) >= 1
        targets = {e["target"] for e in edges}
        assert "evt_ukraine_war" in targets

    def test_get_edges_to(self, default_registry):
        edges = default_registry.get_edges_to("res_oil_brent")
        assert len(edges) >= 1

    def test_get_edges_by_type(self, default_registry):
        disrupts = default_registry.get_edges_by_type("disrupts")
        assert len(disrupts) >= 5


# -------------------------------------------------------------------------
# Tests: CRUD mutations
# -------------------------------------------------------------------------


class TestRegistryCRUD:
    def test_add_node_type(self, empty_registry):
        empty_registry.add_node_type(
            "custom_group",
            label="Custom Group",
            prefix="cust",
            category="test",
        )
        assert "custom_group" in empty_registry.node_types
        assert empty_registry.prefix_for_type("custom_group") == "cust"

    def test_add_duplicate_node_type_raises(self, empty_registry):
        empty_registry.add_node_type("a", label="A", prefix="aa")
        with pytest.raises(ValueError, match="already exists"):
            empty_registry.add_node_type("a", label="A2", prefix="bb")

    def test_add_duplicate_prefix_raises(self, empty_registry):
        empty_registry.add_node_type("a", label="A", prefix="aa")
        with pytest.raises(ValueError, match="Prefix"):
            empty_registry.add_node_type("b", label="B", prefix="aa")

    def test_remove_node_type(self, empty_registry):
        empty_registry.add_node_type("temp", label="Temp", prefix="tmp")
        empty_registry.remove_node_type("temp")
        assert "temp" not in empty_registry.node_types

    def test_add_edge_type(self, empty_registry):
        empty_registry.add_edge_type(
            "custom_rel",
            label="Custom Relationship",
            category="test",
        )
        assert "custom_rel" in empty_registry.edge_types

    def test_add_node(self, empty_registry):
        empty_registry.add_node(
            "test_node_1",
            type_id="sovereign_state",
            label="Test Node",
            domain="test",
        )
        assert "test_node_1" in empty_registry.nodes
        assert empty_registry.get_node("test_node_1")["label"] == "Test Node"

    def test_add_node_invalid_name(self, empty_registry):
        with pytest.raises(ValueError, match="invalid"):
            empty_registry.add_node(
                "bad name",
                type_id="test",
                label="Bad",
            )

    def test_add_duplicate_node_raises(self, empty_registry):
        empty_registry.add_node("n1", type_id="t", label="N1")
        with pytest.raises(ValueError, match="already exists"):
            empty_registry.add_node("n1", type_id="t", label="N1 dup")

    def test_remove_node(self, empty_registry):
        empty_registry.add_node("n1", type_id="t", label="N1")
        empty_registry.remove_node("n1")
        assert "n1" not in empty_registry.nodes

    def test_add_edge(self, empty_registry):
        empty_registry.add_node("a", type_id="t", label="A")
        empty_registry.add_node("b", type_id="t", label="B")
        empty_registry.add_edge("a", "b", edge_type="influences")
        assert len(empty_registry.edges) == 1
        assert empty_registry.edges[0]["source"] == "a"

    def test_remove_edge(self, empty_registry):
        empty_registry.add_edge("x", "y", edge_type="triggers")
        empty_registry.add_edge("x", "y", edge_type="disrupts")
        removed = empty_registry.remove_edge("x", "y", edge_type="triggers")
        assert removed == 1
        assert len(empty_registry.edges) == 1
        assert empty_registry.edges[0]["edge_type"] == "disrupts"

    def test_remove_all_edges_between(self, empty_registry):
        empty_registry.add_edge("x", "y", edge_type="a")
        empty_registry.add_edge("x", "y", edge_type="b")
        removed = empty_registry.remove_edge("x", "y")
        assert removed == 2
        assert len(empty_registry.edges) == 0


# -------------------------------------------------------------------------
# Tests: InMemoryBackend
# -------------------------------------------------------------------------


class TestInMemoryBackend:
    def test_round_trip(self):
        backend = InMemoryBackend()
        registry = OntologyRegistry(backend)

        registry.add_node_type("t1", label="T1", prefix="t")
        registry.add_node("n1", type_id="t1", label="N1")
        registry.add_edge("n1", "n1", edge_type="self_loop")
        registry.save()

        # Load fresh from same backend
        registry2 = OntologyRegistry(backend)
        assert "t1" in registry2.node_types
        assert "n1" in registry2.nodes
        assert len(registry2.edges) == 1


# -------------------------------------------------------------------------
# Tests: YAMLBackend persistence
# -------------------------------------------------------------------------


class TestYAMLBackendPersistence:
    def test_save_and_reload(self, tmp_yaml_dir):
        backend = YAMLBackend(tmp_yaml_dir)
        reg = OntologyRegistry(backend)

        # Add a custom node type
        reg.add_node_type(
            "test_type_999",
            label="Test Type 999",
            prefix="t999",
            category="test",
        )
        reg.add_node("t999_instance", type_id="test_type_999", label="Instance")
        reg.save()

        # Reload
        reg2 = OntologyRegistry(YAMLBackend(tmp_yaml_dir))
        assert "test_type_999" in reg2.node_types
        assert "t999_instance" in reg2.nodes


# -------------------------------------------------------------------------
# Tests: Backward compatibility
# -------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_to_node_registry_dict(self, default_registry):
        legacy = default_registry.to_node_registry_dict()
        assert isinstance(legacy, dict)
        assert "sov_usa" in legacy
        assert "label" in legacy["sov_usa"]

    def test_to_expert_edges(self, default_registry):
        edges = default_registry.to_expert_edges()
        assert isinstance(edges, list)
        assert len(edges) >= 80
        # Each entry is a (source, target) tuple
        for e in edges:
            assert len(e) == 2

    def test_legacy_node_registry_import(self):
        from causalnex.global_graph import NODE_REGISTRY, NodeType

        assert isinstance(NODE_REGISTRY, dict)
        assert len(NODE_REGISTRY) >= 80
        # NodeType enum should have extended members
        assert hasattr(NodeType, "SOVEREIGN_STATE")
        assert hasattr(NodeType, "CLIMATE")

    def test_legacy_validate_node_name_import(self):
        from causalnex.global_graph import validate_node_name

        assert validate_node_name("test_node") is True


# -------------------------------------------------------------------------
# Tests: build_expert_skeleton integration
# -------------------------------------------------------------------------


class TestBuildExpertSkeleton:
    def test_from_registry(self, default_registry):
        from causalnex.global_graph.structure_builder import build_expert_skeleton

        sm = build_expert_skeleton(registry=default_registry)
        assert len(sm.edges) >= 80
        # Check that edge metadata is preserved
        edge_data = sm["fig_putin"]["evt_ukraine_war"]
        assert edge_data["origin"] == "expert"
        assert edge_data.get("edge_type") == "triggers"

    def test_legacy_tuple_list_still_works(self):
        from causalnex.global_graph.structure_builder import build_expert_skeleton

        edges = [("a", "b"), ("b", "c")]
        sm = build_expert_skeleton(edges=edges)
        assert len(sm.edges) == 2

    def test_default_loads_from_yaml(self):
        """Calling with no args should load from YAML."""
        from causalnex.global_graph.structure_builder import build_expert_skeleton

        sm = build_expert_skeleton()
        assert len(sm.edges) >= 80


# -------------------------------------------------------------------------
# Tests: Validation
# -------------------------------------------------------------------------


class TestRegistryValidation:
    def test_default_config_is_consistent(self, default_registry):
        warnings = default_registry.validate()
        assert warnings == [], f"Default config has warnings: {warnings}"

    def test_warns_on_unknown_type(self, empty_registry):
        empty_registry.add_node("bad", type_id="nonexistent", label="Bad")
        warnings = empty_registry.validate()
        assert any("unknown type" in w for w in warnings)

    def test_warns_on_dangling_edge(self, empty_registry):
        empty_registry.add_edge("ghost_src", "ghost_tgt", edge_type="test")
        warnings = empty_registry.validate()
        assert any("ghost_src" in w for w in warnings)
        assert any("ghost_tgt" in w for w in warnings)
