"""
Node taxonomy and naming conventions for the global relationship graph.

.. deprecated::
    The hardcoded ``NodeType`` enum and ``NODE_REGISTRY`` dict are retained
    for backward compatibility.  New code should use
    :class:`~causalnex.global_graph.registry.OntologyRegistry` which loads
    node types and instances from YAML config files (or any pluggable
    backend).

All node names must match the regex ``^[0-9a-zA-Z_]+$`` (enforced by
``InferenceEngine``).
"""

import re
from enum import Enum
from typing import Dict

from causalnex.global_graph.registry import (  # noqa: F401 — re-exported
    OntologyRegistry,
    get_default_registry,
    validate_node_name,
)


_NODE_NAME_RE = re.compile(r"^[0-9a-zA-Z_]+$")


class NodeType(str, Enum):
    """Legacy enumeration of node types.

    Kept for backward compatibility.  The canonical source of truth is now
    the ``node_types.yaml`` config file, loaded via
    :class:`~causalnex.global_graph.registry.OntologyRegistry`.
    """

    # Original 5 types
    FIGURE = "figure"
    EVENT = "event"
    RESOURCE = "resource"
    CURRENCY = "currency"
    COUNTRY = "country"

    # Extended types (mapped from config for convenience)
    SOVEREIGN_STATE = "sovereign_state"
    POLITICAL_FIGURE = "political_figure"
    INTL_ORGANIZATION = "intl_organization"
    COMMODITY = "commodity"
    FINANCIAL_MARKET = "financial_market"
    CORPORATION = "corporation"
    INDUSTRY = "industry"
    CENTRAL_BANK = "central_bank"
    TRADE_ROUTE = "trade_route"
    MILITARY = "military"
    ENERGY_INFRA = "energy_infra"
    TECH_INFRA = "tech_infra"
    DEMOGRAPHICS = "demographics"
    MEDIA = "media"
    REGULATORY = "regulatory"
    GEOPOLITICAL_EVENT = "geopolitical_event"
    CLIMATE = "climate"
    PUBLIC_HEALTH = "public_health"
    SOCIAL_MOVEMENT = "social_movement"


def _build_legacy_registry() -> Dict[str, Dict]:
    """Build the legacy ``NODE_REGISTRY`` dict from the YAML config.

    Maps the config-based ``type`` field to a :class:`NodeType` enum value
    where possible, falling back to the raw string.
    """
    registry = get_default_registry()
    result: Dict[str, Dict] = {}
    for node_id, meta in registry.nodes.items():
        entry = dict(meta)
        raw_type = entry.get("type", "")
        try:
            entry["type"] = NodeType(raw_type)
        except ValueError:
            pass  # keep the raw string if no enum match
        result[node_id] = entry
    return result


# Populated lazily from YAML on first access via module import.
NODE_REGISTRY: Dict[str, Dict] = _build_legacy_registry()
