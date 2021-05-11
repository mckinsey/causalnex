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
This module contains the helper functions for interacting with Bayesian Network
"""

from copy import deepcopy

from causalnex.network import BayesianNetwork


def get_markov_blanket(bn: BayesianNetwork, target_node: str) -> "BayesianNetwork":
    """
    Generate the markov blanket of a node in the network
    Args:
        bn (BayesianNetwork): A BayesianNetwork object that contains the structure of the full graph
        target_node (str): Name of the target node that we want the markov boundary for
    Returns:
        A Bayesian Network object containing the structure of the input's markov blanket
    Raises:
        KeyError: if target_node is not in the network
    """

    if target_node not in bn.nodes:
        raise KeyError(f"{target_node} is not found in the network")

    mb_graph = deepcopy(bn)
    keep_nodes = set()
    for node in mb_graph.nodes:
        if node in mb_graph.structure.predecessors(target_node):
            keep_nodes.add(node)
        if node in mb_graph.structure.successors(target_node):
            keep_nodes.add(node)
            for parent in mb_graph.structure.predecessors(node):
                keep_nodes.add(parent)
    for node in mb_graph.nodes:
        if node not in keep_nodes and node != target_node:
            mb_graph.structure.remove_node(node)

    return BayesianNetwork(mb_graph.structure)
