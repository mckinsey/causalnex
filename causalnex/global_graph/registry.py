"""
Config-driven ontology registry for the global relationship graph.

The registry stores four collections:

* **node types** — entity group definitions (e.g. ``sovereign_state``,
  ``commodity``).
* **edge types** — relationship type definitions (e.g. ``exports_to``,
  ``disrupts``).
* **nodes** — concrete node instances (e.g. ``sov_usa``, ``res_oil_brent``).
* **edges** — concrete causal edges between nodes.

Storage is delegated to a :class:`RegistryBackend`.  The default
:class:`YAMLBackend` reads from the ``config/`` directory shipped with
this package; swap in :class:`SQLiteBackend` or your own implementation
for production-scale deployments with hundreds of entity groups and
thousands of nodes.
"""

import copy
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_NODE_NAME_RE = re.compile(r"^[0-9a-zA-Z_]+$")

_DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class RegistryBackend(ABC):
    """Abstract interface for ontology storage.

    Implement this to back the registry with a database, remote service, or
    any other storage system.
    """

    # -- reads --

    @abstractmethod
    def load_node_types(self) -> Dict[str, Dict]:
        """Return ``{type_id: {label, prefix, category, ...}}``."""

    @abstractmethod
    def load_edge_types(self) -> Dict[str, Dict]:
        """Return ``{edge_type_id: {label, category, ...}}``."""

    @abstractmethod
    def load_nodes(self) -> Dict[str, Dict]:
        """Return ``{node_id: {type, label, domain, ...}}``."""

    @abstractmethod
    def load_edges(self) -> List[Dict]:
        """Return list of edge dicts with keys source, target, edge_type, …"""

    # -- writes --

    @abstractmethod
    def save_node_types(self, data: Dict[str, Dict]) -> None:
        ...

    @abstractmethod
    def save_edge_types(self, data: Dict[str, Dict]) -> None:
        ...

    @abstractmethod
    def save_nodes(self, data: Dict[str, Dict]) -> None:
        ...

    @abstractmethod
    def save_edges(self, data: List[Dict]) -> None:
        ...


# ---------------------------------------------------------------------------
# YAML backend (default)
# ---------------------------------------------------------------------------

class YAMLBackend(RegistryBackend):
    """Read/write ontology from YAML files in a config directory.

    Expected files::

        <config_dir>/
            node_types.yaml
            edge_types.yaml
            default_nodes.yaml
            default_edges.yaml
    """

    def __init__(self, config_dir: str = _DEFAULT_CONFIG_DIR):
        self.config_dir = Path(config_dir)

    def _read(self, filename: str) -> Any:
        import yaml

        path = self.config_dir / filename
        is_list_file = not filename.endswith("types.yaml") and filename != "default_nodes.yaml"
        empty_default: Any = [] if is_list_file else {}
        if not path.exists():
            return empty_default
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if data is not None else empty_default

    def _write(self, filename: str, data: Any) -> None:
        import yaml

        path = self.config_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)

    # -- reads --

    def load_node_types(self) -> Dict[str, Dict]:
        return self._read("node_types.yaml")

    def load_edge_types(self) -> Dict[str, Dict]:
        return self._read("edge_types.yaml")

    def load_nodes(self) -> Dict[str, Dict]:
        return self._read("default_nodes.yaml")

    def load_edges(self) -> List[Dict]:
        data = self._read("default_edges.yaml")
        return data if isinstance(data, list) else []

    # -- writes --

    def save_node_types(self, data: Dict[str, Dict]) -> None:
        self._write("node_types.yaml", data)

    def save_edge_types(self, data: Dict[str, Dict]) -> None:
        self._write("edge_types.yaml", data)

    def save_nodes(self, data: Dict[str, Dict]) -> None:
        self._write("default_nodes.yaml", data)

    def save_edges(self, data: List[Dict]) -> None:
        self._write("default_edges.yaml", data)


# ---------------------------------------------------------------------------
# In-memory backend (useful for tests and ephemeral sessions)
# ---------------------------------------------------------------------------

class InMemoryBackend(RegistryBackend):
    """Purely in-memory backend — nothing is persisted to disk."""

    def __init__(self):
        self._node_types: Dict[str, Dict] = {}
        self._edge_types: Dict[str, Dict] = {}
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Dict] = []

    def load_node_types(self):
        return copy.deepcopy(self._node_types)

    def load_edge_types(self):
        return copy.deepcopy(self._edge_types)

    def load_nodes(self):
        return copy.deepcopy(self._nodes)

    def load_edges(self):
        return copy.deepcopy(self._edges)

    def save_node_types(self, data):
        self._node_types = copy.deepcopy(data)

    def save_edge_types(self, data):
        self._edge_types = copy.deepcopy(data)

    def save_nodes(self, data):
        self._nodes = copy.deepcopy(data)

    def save_edges(self, data):
        self._edges = copy.deepcopy(data)


# ---------------------------------------------------------------------------
# OntologyRegistry
# ---------------------------------------------------------------------------

class OntologyRegistry:
    """Central registry for the global graph ontology.

    Loads node types, edge types, nodes, and edges from a
    :class:`RegistryBackend` and provides query / mutation helpers.

    Args:
        backend: storage backend.  Defaults to :class:`YAMLBackend` reading
            from the ``config/`` directory shipped with this package.
    """

    def __init__(self, backend: Optional[RegistryBackend] = None):
        self._backend = backend or YAMLBackend()
        self._node_types: Dict[str, Dict] = self._backend.load_node_types()
        self._edge_types: Dict[str, Dict] = self._backend.load_edge_types()
        self._nodes: Dict[str, Dict] = self._backend.load_nodes()
        self._edges: List[Dict] = self._backend.load_edges()

    # ------------------------------------------------------------------
    # Node type queries
    # ------------------------------------------------------------------

    @property
    def node_types(self) -> Dict[str, Dict]:
        """All registered node (entity group) types."""
        return self._node_types

    def get_node_type(self, type_id: str) -> Dict:
        """Return metadata for a single node type.

        Raises:
            KeyError: if *type_id* is not registered.
        """
        return self._node_types[type_id]

    def prefix_for_type(self, type_id: str) -> str:
        """Return the naming prefix for *type_id* (e.g. ``"sov"``).

        Raises:
            KeyError: if *type_id* is not registered.
        """
        return self._node_types[type_id]["prefix"]

    def type_for_prefix(self, prefix: str) -> Optional[str]:
        """Reverse-lookup: return the type_id that owns *prefix*, or None."""
        for tid, meta in self._node_types.items():
            if meta.get("prefix") == prefix:
                return tid
        return None

    # ------------------------------------------------------------------
    # Edge type queries
    # ------------------------------------------------------------------

    @property
    def edge_types(self) -> Dict[str, Dict]:
        """All registered edge (relationship) types."""
        return self._edge_types

    def get_edge_type(self, edge_type_id: str) -> Dict:
        """Return metadata for a single edge type.

        Raises:
            KeyError: if *edge_type_id* is not registered.
        """
        return self._edge_types[edge_type_id]

    # ------------------------------------------------------------------
    # Node queries
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Dict[str, Dict]:
        """All registered node instances."""
        return self._nodes

    def get_node(self, node_id: str) -> Dict:
        """Return metadata for a single node.

        Raises:
            KeyError: if *node_id* is not registered.
        """
        return self._nodes[node_id]

    def get_nodes_by_type(self, type_id: str) -> Dict[str, Dict]:
        """Return all nodes whose ``type`` matches *type_id*."""
        return {
            nid: meta
            for nid, meta in self._nodes.items()
            if meta.get("type") == type_id
        }

    def get_nodes_by_domain(self, domain: str) -> Dict[str, Dict]:
        """Return all nodes whose ``domain`` matches *domain*."""
        return {
            nid: meta
            for nid, meta in self._nodes.items()
            if meta.get("domain") == domain
        }

    # ------------------------------------------------------------------
    # Edge queries
    # ------------------------------------------------------------------

    @property
    def edges(self) -> List[Dict]:
        """All registered edge instances."""
        return self._edges

    def get_edges_from(self, source_id: str) -> List[Dict]:
        """Return edges originating from *source_id*."""
        return [e for e in self._edges if e["source"] == source_id]

    def get_edges_to(self, target_id: str) -> List[Dict]:
        """Return edges pointing to *target_id*."""
        return [e for e in self._edges if e["target"] == target_id]

    def get_edges_by_type(self, edge_type: str) -> List[Dict]:
        """Return all edges of a given relationship type."""
        return [e for e in self._edges if e.get("edge_type") == edge_type]

    # ------------------------------------------------------------------
    # Mutations (CRUD)
    # ------------------------------------------------------------------

    def add_node_type(self, type_id: str, *, label: str, prefix: str,
                      category: str = "", description: str = "",
                      attributes: Optional[List[str]] = None,
                      **extra: Any) -> None:
        """Register a new entity group.

        Raises:
            ValueError: if *type_id* or *prefix* already exists.
        """
        if type_id in self._node_types:
            raise ValueError(f"Node type '{type_id}' already exists.")
        if any(m.get("prefix") == prefix for m in self._node_types.values()):
            raise ValueError(f"Prefix '{prefix}' already in use.")
        self._node_types[type_id] = {
            "label": label,
            "prefix": prefix,
            "category": category,
            "description": description,
            "attributes": attributes or [],
            **extra,
        }

    def remove_node_type(self, type_id: str) -> None:
        """Remove an entity group. Does NOT cascade-delete nodes."""
        self._node_types.pop(type_id, None)

    def add_edge_type(self, edge_type_id: str, *, label: str,
                      category: str = "", directed: bool = True,
                      description: str = "",
                      typical_pairs: Optional[List] = None,
                      **extra: Any) -> None:
        """Register a new relationship type.

        Raises:
            ValueError: if *edge_type_id* already exists.
        """
        if edge_type_id in self._edge_types:
            raise ValueError(f"Edge type '{edge_type_id}' already exists.")
        self._edge_types[edge_type_id] = {
            "label": label,
            "category": category,
            "directed": directed,
            "description": description,
            "typical_pairs": typical_pairs or [],
            **extra,
        }

    def remove_edge_type(self, edge_type_id: str) -> None:
        """Remove a relationship type.  Does NOT cascade-delete edges."""
        self._edge_types.pop(edge_type_id, None)

    def add_node(self, node_id: str, *, type_id: str, label: str,
                 domain: str = "", **extra: Any) -> None:
        """Register a concrete node instance.

        Raises:
            ValueError: if *node_id* already exists or name is invalid.
        """
        validate_node_name(node_id)
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists.")
        self._nodes[node_id] = {"type": type_id, "label": label,
                                "domain": domain, **extra}

    def remove_node(self, node_id: str) -> None:
        """Remove a node instance.  Does NOT cascade-delete edges."""
        self._nodes.pop(node_id, None)

    def add_edge(self, source: str, target: str, edge_type: str,
                 origin: str = "expert", weight: float = 1.0,
                 description: str = "", **extra: Any) -> None:
        """Register a concrete causal edge."""
        self._edges.append({
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "origin": origin,
            "weight": weight,
            "description": description,
            **extra,
        })

    def remove_edge(self, source: str, target: str,
                    edge_type: Optional[str] = None) -> int:
        """Remove edges matching *source* → *target* (and optionally *edge_type*).

        Returns:
            Number of edges removed.
        """
        before = len(self._edges)
        self._edges = [
            e for e in self._edges
            if not (
                e["source"] == source
                and e["target"] == target
                and (edge_type is None or e.get("edge_type") == edge_type)
            )
        ]
        return before - len(self._edges)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist current state back through the backend."""
        self._backend.save_node_types(self._node_types)
        self._backend.save_edge_types(self._edge_types)
        self._backend.save_nodes(self._nodes)
        self._backend.save_edges(self._edges)

    # ------------------------------------------------------------------
    # Backward-compatibility helpers
    # ------------------------------------------------------------------

    def to_node_registry_dict(self) -> Dict[str, Dict]:
        """Export nodes in the legacy ``NODE_REGISTRY`` format.

        Each value is ``{"type": <type_id_string>, "label": ..., "domain": ...}``.
        """
        return copy.deepcopy(self._nodes)

    def to_expert_edges(self) -> List[Tuple[str, str]]:
        """Export edges as ``[(source, target), ...]`` for ``build_expert_skeleton``.

        Only includes edges with ``origin == "expert"``.
        """
        return [
            (e["source"], e["target"])
            for e in self._edges
            if e.get("origin") == "expert"
        ]

    def to_typed_expert_edges(self) -> List[Dict]:
        """Export expert edges with full metadata for the enhanced structure builder."""
        return [
            e for e in self._edges
            if e.get("origin") == "expert"
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Run consistency checks.  Returns a list of warning strings.

        Checks:
        * Every node references a known node type.
        * Every edge references known source/target nodes.
        * Every edge references a known edge type.
        * Node names match the ``InferenceEngine`` regex.
        """
        warnings = []

        for node_id, meta in self._nodes.items():
            if not _NODE_NAME_RE.match(node_id):
                warnings.append(f"Invalid node name: '{node_id}'")
            ntype = meta.get("type")
            if ntype and ntype not in self._node_types:
                warnings.append(
                    f"Node '{node_id}' references unknown type '{ntype}'"
                )

        known_nodes = set(self._nodes.keys())
        for i, edge in enumerate(self._edges):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            etype = edge.get("edge_type", "")
            if src not in known_nodes:
                warnings.append(
                    f"Edge #{i} source '{src}' not in node registry"
                )
            if tgt not in known_nodes:
                warnings.append(
                    f"Edge #{i} target '{tgt}' not in node registry"
                )
            if etype and etype not in self._edge_types:
                warnings.append(
                    f"Edge #{i} type '{etype}' not in edge type registry"
                )

        return warnings

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OntologyRegistry("
            f"{len(self._node_types)} node types, "
            f"{len(self._edge_types)} edge types, "
            f"{len(self._nodes)} nodes, "
            f"{len(self._edges)} edges)"
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def validate_node_name(name: str) -> bool:
    """Return True if *name* satisfies the InferenceEngine regex.

    Raises:
        ValueError: when *name* does not match ``^[0-9a-zA-Z_]+$``.
    """
    if not _NODE_NAME_RE.match(name):
        raise ValueError(
            f"Node name '{name}' is invalid. "
            "Names must match ^[0-9a-zA-Z_]+$ (no spaces, hyphens, etc.)."
        )
    return True


def get_default_registry() -> OntologyRegistry:
    """Return a registry loaded from the shipped YAML config files."""
    return OntologyRegistry(YAMLBackend(_DEFAULT_CONFIG_DIR))
