"""
Node taxonomy and naming conventions for the global relationship graph.

All node names must match the regex ``^[0-9a-zA-Z_]+$`` (enforced by
``InferenceEngine``). A prefixed naming convention encodes the node type:

- ``fig_`` — world figures (politicians, CEOs, central bankers)
- ``evt_`` — news events (wars, sanctions, policy decisions)
- ``res_`` — resources / commodities (oil, wheat, semiconductors)
- ``cur_`` — currencies (USD, EUR, CNY strength indices)
- ``ctr_`` — country economic metrics (GDP, trade volume)

Node states are ordinal integers: 0 = low/none, 1 = medium/minor, 2 = high/major.
"""

import re
from enum import Enum
from typing import Dict


_NODE_NAME_RE = re.compile(r"^[0-9a-zA-Z_]+$")


class NodeType(str, Enum):
    """Enumeration of supported node types."""

    FIGURE = "figure"
    EVENT = "event"
    RESOURCE = "resource"
    CURRENCY = "currency"
    COUNTRY = "country"


# Default set of nodes for the global relationship graph.
# Users can extend this registry and pass it into ``build_expert_skeleton``.
NODE_REGISTRY: Dict[str, Dict] = {
    # World figures
    "fig_biden": {
        "type": NodeType.FIGURE,
        "label": "Biden",
        "domain": "politics",
    },
    "fig_powell": {
        "type": NodeType.FIGURE,
        "label": "Jerome Powell",
        "domain": "finance",
    },
    "fig_mbs": {
        "type": NodeType.FIGURE,
        "label": "MBS",
        "domain": "politics",
    },
    # News events
    "evt_ukraine_war": {
        "type": NodeType.EVENT,
        "label": "Ukraine War",
        "domain": "geopolitical",
    },
    "evt_fed_rate_hike": {
        "type": NodeType.EVENT,
        "label": "Fed Rate Hike",
        "domain": "economic",
    },
    "evt_opec_cut": {
        "type": NodeType.EVENT,
        "label": "OPEC Cut",
        "domain": "economic",
    },
    # Resources / commodities
    "res_oil_brent": {
        "type": NodeType.RESOURCE,
        "label": "Brent Oil",
        "domain": "energy",
    },
    "res_semiconductor": {
        "type": NodeType.RESOURCE,
        "label": "Semiconductors",
        "domain": "tech",
    },
    "res_wheat": {
        "type": NodeType.RESOURCE,
        "label": "Wheat",
        "domain": "agriculture",
    },
    # Currencies
    "cur_usd_strength": {
        "type": NodeType.CURRENCY,
        "label": "USD Strength",
        "domain": "forex",
    },
    "cur_eur_strength": {
        "type": NodeType.CURRENCY,
        "label": "EUR Strength",
        "domain": "forex",
    },
    "cur_cny_strength": {
        "type": NodeType.CURRENCY,
        "label": "CNY Strength",
        "domain": "forex",
    },
    # Country economic metrics
    "ctr_russia_gdp": {
        "type": NodeType.COUNTRY,
        "label": "Russia GDP",
        "domain": "economics",
    },
    "ctr_usa_gdp": {
        "type": NodeType.COUNTRY,
        "label": "USA GDP",
        "domain": "economics",
    },
    "ctr_china_trade": {
        "type": NodeType.COUNTRY,
        "label": "China Trade",
        "domain": "economics",
    },
}


def validate_node_name(name: str) -> bool:
    """Return ``True`` if *name* satisfies the ``InferenceEngine`` regex.

    Raises:
        ValueError: when *name* does not match ``^[0-9a-zA-Z_]+$``.
    """
    if not _NODE_NAME_RE.match(name):
        raise ValueError(
            f"Node name '{name}' is invalid. "
            "Names must match ^[0-9a-zA-Z_]+$ (no spaces, hyphens, etc.)."
        )
    return True
