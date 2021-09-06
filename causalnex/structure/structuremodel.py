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
"""
This module contains the implementation of ``StructureModel``.

``StructureModel`` is a class that describes relationships between variables as a graph.
"""

from typing import Hashable, List, Set, Union

import networkx as nx
import numpy as np
from networkx.exception import NodeNotFound


def _validate_origin(origin: str) -> None:
    """
    Checks that origin has a valid value. One of:
        - unknown: edge exists for an unknown reason;
        - learned: edge was created as the output of a machine-learning process;
        - expert: edge was created by a domain expert.

    Args:
        origin: the value to validate.

    Raises:
        ValueError: if origin is not valid.
    """

    allowed = {"unknown", "learned", "expert"}

    if origin not in allowed:
        raise ValueError(f"Unknown origin: must be one of {allowed} - got `{origin}`.")


class StructureModel(nx.DiGraph):
    """
    Base class for structure models, which are an extension of ``networkx.DiGraph``.

    A ``StructureModel`` stores nodes and edges with optional data, or attributes.

    Edges have one required attribute, "origin", which describes how the edge was created.
    Origin can be one of either unknown, learned, or expert.

    StructureModel hold directed edges, describing a cause -> effect relationship.
    Cycles are permitted within a ``StructureModel``.

    Nodes can be arbitrary (hashable) Python objects with optional key/value attributes.
    By convention None is not used as a node.

    Edges are represented as links between nodes with optional key/value attributes.
    """

    def __init__(self, incoming_graph_data=None, origin="unknown", **attr):
        """
        Create a ``StructureModel`` with incoming_graph_data, which has come from some origin.

        Args:
            incoming_graph_data (Optional): input graph (optional, default: None)
                                 Data to initialize graph. If None (default) an empty graph is created.
                                 The data can be any format that is supported by the to_networkx_graph()
                                 function, currently including edge list, dict of dicts, dict of lists,
                                 NetworkX graph, NumPy matrix or 2d ndarray, SciPy sparse matrix, or PyGraphviz graph.

            origin (str): label for how the edges were created. Can be one of:
                        - unknown: edges exist for an unknown reason;
                        - learned: edges were created as the output of a machine-learning process;
                        - expert: edges were created by a domain expert.

            attr : Attributes to add to graph as key/value pairs (no attributes by default).
        """

        _validate_origin(origin)
        super().__init__(incoming_graph_data, **attr)
        for u_of_edge, v_of_edge in self.edges:
            self[u_of_edge][v_of_edge]["origin"] = origin

    def to_directed_class(self):
        """
        Returns the class to use for directed copies.
        See :func:`networkx.DiGraph.to_directed()`.
        """
        return StructureModel

    def to_undirected_class(self):
        """
        Returns the class to use for undirected copies.
        See :func:`networkx.DiGraph.to_undirected()`.
        """
        return nx.Graph

    # disabled: W0221: Parameters differ from overridden 'add_edge' method (arguments-differ)
    # this has been disabled because origin tracking is required for CausalGraphs
    # implementing it in this way allows all 3rd party libraries and applications to
    # integrate seamlessly, where edges will be given origin="unknown" where not provided
    def add_edge(
        self, u_of_edge: str, v_of_edge: str, origin: str = "unknown", **attr
    ):  # pylint: disable=W0221
        """
        Adds a causal relationship from u to v.

        If u or v do not currently exists in the ``StructureModel`` then they will be created.

        By default a relationship will be given origin="unknown", but
        may also be given "learned" or "expert" origin.

        Adding an edge that already exists will replace the existing edge.
        See :func:`networkx.DiGraph.add_edge`.

        Args:
            u_of_edge: causal node.
            v_of_edge: effect node.
            origin: label for how the edge was created. Can be one of:
                        - unknown: edge exists for an unknown reason;
                        - learned: edge was created as the output of a machine-learning process;
                        - expert: edge was created by a domain expert.
            **attr:  Attributes to add to edge as key/value pairs (no attributes by default).
        """
        _validate_origin(origin)

        attr.update({"origin": origin})
        super().add_edge(u_of_edge, v_of_edge, **attr)

    # disabled: W0221: Parameters differ from overridden 'add_edge' method (arguments-differ)
    # this has been disabled because origin tracking is required for CausalGraphs
    # implementing it in this way allows all 3rd party libraries and applications to
    # integrate seamlessly, where edges will be given origin="unknown" where not provided
    def add_edges_from(
        self,
        ebunch_to_add: Union[Set[tuple], List[tuple]],
        origin: str = "unknown",
        **attr,
    ):  # pylint: disable=W0221
        """
        Adds a bunch of causal relationships, u -> v.

        If u or v do not currently exists in the ``StructureModel`` then they will be created.

        By default relationships will be given origin="unknown",
        but may also be given "learned" or "expert" origin.

        Notes:
            Adding an edge that already exists will replace the existing edge.
            See :func:`networkx.DiGraph.add_edges_from`.

        Args:
            ebunch_to_add: container of edges.
                           Each edge given in the container will be added to the graph.
                           The edges must be given as 2-tuples (u, v) or
                           3-tuples (u, v, d) where d is a dictionary containing edge data.
            origin: label for how the edges were created. One of:
                        - unknown: edges exist for an unknown reason.
                        - learned: edges were created as the output of a machine-learning process.
                        - expert: edges were created by a domain expert.
            **attr:  Attributes to add to edge as key/value pairs (no attributes by default).
        """

        _validate_origin(origin)

        attr.update({"origin": origin})
        super().add_edges_from(ebunch_to_add, **attr)

    # disabled: W0221: Parameters differ from overridden 'add_edge' method (arguments-differ)
    # this has been disabled because origin tracking is required for CausalGraphs
    # implementing it in this way allows all 3rd party libraries and applications to
    # integrate seamlessly, where edges will be given origin="unknown" where not provided
    def add_weighted_edges_from(
        self,
        ebunch_to_add: Union[Set[tuple], List[tuple]],
        weight: str = "weight",
        origin: str = "unknown",
        **attr,
    ):  # pylint: disable=W0221
        """
        Adds a bunch of weighted causal relationships, u -> v.

        If u or v do not currently exists in the ``StructureModel`` then they will be created.

        By default relationships will be given origin="unknown",
        but may also be given "learned" or "expert" origin.

        Notes:
            Adding an edge that already exists will replace the existing edge.
            See :func:`networkx.DiGraph.add_edges_from`.

        Args:
            ebunch_to_add: container of edges.
                           Each edge given in the container will be added to the graph.
                           The edges must be given as 2-tuples (u, v) or
                           3-tuples (u, v, d) where d is a dictionary containing edge data.
            weight : string, optional (default='weight').
                     The attribute name for the edge weights to be added.
            origin: label for how the edges were created. One of:
                - unknown: edges exist for an unknown reason;
                - learned: edges were created as the output of a machine-learning process;
                - expert: edges were created by a domain expert.
            **attr: Attributes to add to edge as key/value pairs (no attributes by default).
        """
        _validate_origin(origin)

        attr.update({"origin": origin})
        super().add_weighted_edges_from(ebunch_to_add, weight=weight, **attr)

    def edges_with_origin(self, origin) -> list:
        """
        List of edges created with given origin attribute.

        Returns:
            A list of edges with the given origin.
        """

        return [(u, v) for u, v in self.edges if self[u][v]["origin"] == origin]

    def remove_edges_below_threshold(self, threshold: float):
        """
        Remove edges whose absolute weights are less than a defined threshold.

        Args:
            threshold: edges whose absolute weight is less than this value are removed.
        """

        self.remove_edges_from(
            [(u, v) for u, v, w in self.edges(data="weight") if np.abs(w) < threshold]
        )

    def get_largest_subgraph(self) -> "StructureModel":
        """
        Get the largest subgraph of the Structure Model.

        Returns:
            The largest subgraph of the Structure Model. If no subgraph exists, None is returned.
        """
        largest_n_edges = 0
        largest_subgraph = None

        for subgraph in (
            self.subgraph(c).copy() for c in nx.weakly_connected_components(self)
        ):
            if len(subgraph.edges) > largest_n_edges:
                largest_n_edges = len(subgraph.edges)
                largest_subgraph = subgraph

        return largest_subgraph

    def get_target_subgraph(self, node: Hashable) -> "StructureModel":
        """
        Get the subgraph with the specified node.

        Args:
            node: the name of the node.

        Returns:
            The subgraph with the target node.

        Raises:
            NodeNotFound: if the node is not found in the graph.
        """
        if node in self.nodes:
            for subgraph in (
                self.subgraph(c).copy() for c in nx.weakly_connected_components(self)
            ):
                if node in subgraph.nodes:
                    return subgraph

        raise NodeNotFound(f"Node {node} not found in the graph.")

    def threshold_till_dag(self):
        """
        Remove edges with smallest weight until the graph is a DAG.
        Not recommended if the weights have different units.
        """
        while not nx.algorithms.is_directed_acyclic_graph(self):
            i, j, _ = min(self.edges(data="weight"), key=lambda x: abs(x[2]))
            self.remove_edge(i, j)
