# Copyright 2019-2020 QuantumBlack Visual Analytics Limited
#
# The methods found in this file are adapted from a repository under Apache 2.0:
# eBay's Pythonic Bayesian Belief Network Framework.
# @online{
#     author = {Neville Newey,Anzar Afaq},
#     title = {bayesian-belief-networks},
#     organisation = {eBay},
#     codebase = {https://github.com/eBay/bayesian-belief-networks},
# }
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
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Generic Graph Classes"""


class Node(object):
    def __init__(self, name, parents=[], children=[]):
        self.name = name
        self.parents = parents[:]
        self.children = children[:]

    def __repr__(self):
        return "<Node %s>" % self.name


class UndirectedNode(object):
    def __init__(self, name, neighbours=[]):
        self.name = name
        self.neighbours = neighbours[:]

    def __repr__(self):
        return "<UndirectedNode %s>" % self.name


class UndirectedGraph(object):
    def __init__(self, nodes, name=None):
        self.nodes = nodes
        self.name = name


def connect(parent, child):
    """
    Make an edge between a parent
    node and a child node.
    a - parent
    b - child
    """
    parent.children.append(child)
    child.parents.append(parent)
