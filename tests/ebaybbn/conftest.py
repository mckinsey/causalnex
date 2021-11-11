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

import pytest

from causalnex.ebaybbn import (
    BBN,
    Node,
    build_bbn,
    build_join_tree,
    combinations,
    make_moralized_copy,
    make_undirected_copy,
)
from causalnex.ebaybbn.utils import make_key


@pytest.fixture
def sprinkler_graph():
    """The Sprinkler Example as a BBN
    to be used in tests.
    """
    cloudy = Node("Cloudy")
    sprinkler = Node("Sprinkler")
    rain = Node("Rain")
    wet_grass = Node("WetGrass")
    cloudy.children = [sprinkler, rain]
    sprinkler.parents = [cloudy]
    sprinkler.children = [wet_grass]
    rain.parents = [cloudy]
    rain.children = [wet_grass]
    wet_grass.parents = [sprinkler, rain]
    bbn = BBN(dict(cloudy=cloudy, sprinkler=sprinkler, rain=rain, wet_grass=wet_grass))
    return bbn


@pytest.fixture
def sprinkler_bbn():
    """Sprinkler BBN built with build_bbn."""

    def f_rain(rain):
        if rain is True:
            return 0.2
        return 0.8

    def f_sprinkler(rain, sprinkler):
        sprinkler_dict = {
            (False, True): 0.4,
            (False, False): 0.6,
            (True, True): 0.01,
            (True, False): 0.99,
        }
        return sprinkler_dict[(rain, sprinkler)]

    def f_grass_wet(sprinkler, rain, grass_wet):
        table = {}
        table["fft"] = 0.0
        table["fff"] = 1.0
        table["ftt"] = 0.8
        table["ftf"] = 0.2
        table["tft"] = 0.9
        table["tff"] = 0.1
        table["ttt"] = 0.99
        table["ttf"] = 0.01
        return table[make_key(sprinkler, rain, grass_wet)]

    return build_bbn(f_rain, f_sprinkler, f_grass_wet)


@pytest.fixture
def huang_darwiche_nodes():
    """The nodes for the Huang Darwich example"""

    def f_a(a):
        if a:
            return 1 / 2
        return 1 / 2

    def f_b(a, b):
        tt = dict(tt=0.5, ft=0.4, tf=0.5, ff=0.6)
        return tt[make_key(a, b)]

    def f_c(a, c):
        tt = dict(tt=0.7, ft=0.2, tf=0.3, ff=0.8)
        return tt[make_key(a, c)]

    def f_d(b, d):
        tt = dict(tt=0.9, ft=0.5, tf=0.1, ff=0.5)
        return tt[make_key(b, d)]

    def f_e(c, e):
        tt = dict(tt=0.3, ft=0.6, tf=0.7, ff=0.4)
        return tt[make_key(c, e)]

    def f_f(d, e, f):
        tt = dict(
            ttt=0.01,
            ttf=0.99,
            tft=0.01,
            tff=0.99,
            ftt=0.01,
            ftf=0.99,
            fft=0.99,
            fff=0.01,
        )
        return tt[make_key(d, e, f)]

    def f_g(c, g):
        tt = dict(tt=0.8, tf=0.2, ft=0.1, ff=0.9)
        return tt[make_key(c, g)]

    def f_h(e, g, h):
        tt = dict(
            ttt=0.05,
            ttf=0.95,
            tft=0.95,
            tff=0.05,
            ftt=0.95,
            ftf=0.05,
            fft=0.95,
            fff=0.05,
        )
        return tt[make_key(e, g, h)]

    return [f_a, f_b, f_c, f_d, f_e, f_f, f_g, f_h]


@pytest.fixture
def huang_darwiche_dag(huang_darwiche_nodes):

    nodes = huang_darwiche_nodes
    return build_bbn(nodes)


@pytest.fixture
def huang_darwiche_moralized(huang_darwiche_dag):

    dag = huang_darwiche_dag
    gu = make_undirected_copy(dag)
    gm = make_moralized_copy(gu, dag)

    return gm


@pytest.fixture
def huang_darwiche_jt(huang_darwiche_dag):
    def priority_func_override(node):
        introduced_arcs = 0
        cluster = [node] + node.neighbours
        for node_a, node_b in combinations(cluster, 2):
            if node_a not in node_b.neighbours:
                assert node_b not in node_a.neighbours
                introduced_arcs += 1
        introduced_arcs_dict = {
            "f_h": [introduced_arcs, 0],
            "f_g": [introduced_arcs, 1],
            "f_c": [introduced_arcs, 2],
            "f_b": [introduced_arcs, 3],
            "f_d": [introduced_arcs, 4],
            "f_e": [introduced_arcs, 5],
            "others": [introduced_arcs, 10],
        }
        if node.name in introduced_arcs_dict:
            return introduced_arcs_dict[node.name]

        return introduced_arcs_dict["others"]

    dag = huang_darwiche_dag
    jt = build_join_tree(dag, priority_func_override)
    return jt
