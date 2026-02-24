"""Tests for causalnex.global_graph.node_registry."""

import pytest

from causalnex.global_graph.node_registry import (
    NODE_REGISTRY,
    NodeType,
    validate_node_name,
)


class TestValidateNodeName:
    def test_valid_names(self):
        for name in ["fig_biden", "evt_ukraine_war", "res_oil_brent", "node123"]:
            assert validate_node_name(name) is True

    def test_invalid_name_with_hyphen(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_node_name("saudi-arabia")

    def test_invalid_name_with_space(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_node_name("bad name")

    def test_invalid_name_with_parens(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_node_name("node(1)")


class TestNodeRegistry:
    def test_all_names_are_valid(self):
        for name in NODE_REGISTRY:
            assert validate_node_name(name) is True

    def test_all_entries_have_required_keys(self):
        for name, meta in NODE_REGISTRY.items():
            assert "type" in meta, f"{name} missing 'type'"
            assert "label" in meta, f"{name} missing 'label'"
            assert "domain" in meta, f"{name} missing 'domain'"

    def test_type_values_are_node_type_enum(self):
        for name, meta in NODE_REGISTRY.items():
            assert isinstance(meta["type"], NodeType), f"{name} has bad type"

    def test_registry_has_all_five_types(self):
        types_present = {meta["type"] for meta in NODE_REGISTRY.values()}
        assert types_present == {
            NodeType.FIGURE,
            NodeType.EVENT,
            NodeType.RESOURCE,
            NodeType.CURRENCY,
            NodeType.COUNTRY,
        }
