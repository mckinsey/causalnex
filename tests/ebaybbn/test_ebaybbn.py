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

from __future__ import division

import copy
from collections import Counter

import pytest

from causalnex.ebaybbn import (
    BBNNode,
    JoinTree,
    JoinTreeCliqueNode,
    SepSet,
    build_bbn,
    build_bbn_from_conditionals,
    build_join_tree,
    combinations,
    make_moralized_copy,
    make_node_func,
    make_undirected_copy,
    priority_func,
    triangulate,
)
from causalnex.ebaybbn.exceptions import (
    VariableNotInGraphError,
    VariableValueNotInDomainError,
)
from causalnex.ebaybbn.graph import Node, UndirectedNode
from causalnex.ebaybbn.utils import get_args, get_original_factors, make_key


def r3(x):
    return round(x, 3)


def r5(x):
    return round(x, 5)


class TestBBN:
    def test_get_graphviz_source(self, sprinkler_graph):
        gv_src = """digraph G {
  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];
  Cloudy [ shape="ellipse" color="blue"];
  Rain [ shape="ellipse" color="blue"];
  Sprinkler [ shape="ellipse" color="blue"];
  WetGrass [ shape="ellipse" color="blue"];
  Cloudy -> Rain;
  Cloudy -> Sprinkler;
  Rain -> WetGrass;
  Sprinkler -> WetGrass;
}
"""
        assert sprinkler_graph.get_graphviz_source() == gv_src

    def test_get_original_factors(self, huang_darwiche_nodes):

        original_factors = get_original_factors(huang_darwiche_nodes)
        assert original_factors["a"] == huang_darwiche_nodes[0]
        assert original_factors["b"] == huang_darwiche_nodes[1]
        assert original_factors["c"] == huang_darwiche_nodes[2]
        assert original_factors["d"] == huang_darwiche_nodes[3]
        assert original_factors["e"] == huang_darwiche_nodes[4]
        assert original_factors["f"] == huang_darwiche_nodes[5]
        assert original_factors["g"] == huang_darwiche_nodes[6]
        assert original_factors["h"] == huang_darwiche_nodes[7]

    def test_build_graph(self, huang_darwiche_nodes):
        bbn = build_bbn(huang_darwiche_nodes)
        nodes = {node.name: node for node in bbn.nodes}
        assert nodes["f_a"].parents == []
        assert nodes["f_b"].parents == [nodes["f_a"]]
        assert nodes["f_c"].parents == [nodes["f_a"]]
        assert nodes["f_d"].parents == [nodes["f_b"]]
        assert nodes["f_e"].parents == [nodes["f_c"]]
        assert nodes["f_f"].parents == [nodes["f_d"], nodes["f_e"]]
        assert nodes["f_g"].parents == [nodes["f_c"]]
        assert nodes["f_h"].parents == [nodes["f_e"], nodes["f_g"]]

    def test_make_undirecred_copy(self, huang_darwiche_dag):
        ug = make_undirected_copy(huang_darwiche_dag)
        nodes = {node.name: node for node in ug.nodes}
        assert set(nodes["f_a"].neighbours) == set([nodes["f_b"], nodes["f_c"]])
        assert set(nodes["f_b"].neighbours) == set([nodes["f_a"], nodes["f_d"]])
        assert set(nodes["f_c"].neighbours) == set(
            [nodes["f_a"], nodes["f_e"], nodes["f_g"]]
        )
        assert set(nodes["f_d"].neighbours) == set([nodes["f_b"], nodes["f_f"]])
        assert set(nodes["f_e"].neighbours) == set(
            [nodes["f_c"], nodes["f_f"], nodes["f_h"]]
        )
        assert set(nodes["f_f"].neighbours) == set([nodes["f_d"], nodes["f_e"]])
        assert set(nodes["f_g"].neighbours) == set([nodes["f_c"], nodes["f_h"]])
        assert set(nodes["f_h"].neighbours) == set([nodes["f_e"], nodes["f_g"]])

    def test_make_moralized_copy(self, huang_darwiche_dag):
        gu = make_undirected_copy(huang_darwiche_dag)
        gm = make_moralized_copy(gu, huang_darwiche_dag)
        nodes = {node.name: node for node in gm.nodes}
        assert set(nodes["f_a"].neighbours) == set([nodes["f_b"], nodes["f_c"]])
        assert set(nodes["f_b"].neighbours) == set([nodes["f_a"], nodes["f_d"]])
        assert set(nodes["f_c"].neighbours) == set(
            [nodes["f_a"], nodes["f_e"], nodes["f_g"]]
        )
        assert set(nodes["f_d"].neighbours) == set(
            [nodes["f_b"], nodes["f_f"], nodes["f_e"]]
        )
        assert set(nodes["f_e"].neighbours) == set(
            [nodes["f_c"], nodes["f_f"], nodes["f_h"], nodes["f_d"], nodes["f_g"]]
        )
        assert set(nodes["f_f"].neighbours) == set([nodes["f_d"], nodes["f_e"]])
        assert set(nodes["f_g"].neighbours) == set(
            [nodes["f_c"], nodes["f_h"], nodes["f_e"]]
        )
        assert set(nodes["f_h"].neighbours) == set([nodes["f_e"], nodes["f_g"]])

    def test_triangulate(self, huang_darwiche_moralized):

        # Because of ties in the priority q we will
        # override the priority function here to
        # insert tie breakers to ensure the same
        # elimination ordering as Darwich Huang.
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

        cliques, elimination_ordering = triangulate(
            huang_darwiche_moralized, priority_func_override
        )
        nodes = {node.name: node for node in huang_darwiche_moralized.nodes}
        assert len(cliques) == 6
        assert cliques[0].nodes == set([nodes["f_e"], nodes["f_g"], nodes["f_h"]])
        assert cliques[1].nodes == set([nodes["f_c"], nodes["f_e"], nodes["f_g"]])
        assert cliques[2].nodes == set([nodes["f_d"], nodes["f_e"], nodes["f_f"]])
        assert cliques[3].nodes == set([nodes["f_a"], nodes["f_c"], nodes["f_e"]])
        assert cliques[4].nodes == set([nodes["f_a"], nodes["f_b"], nodes["f_d"]])
        assert cliques[5].nodes == set([nodes["f_a"], nodes["f_d"], nodes["f_e"]])

        assert elimination_ordering == [
            "f_h",
            "f_g",
            "f_f",
            "f_c",
            "f_b",
            "f_d",
            "f_e",
            "f_a",
        ]
        # Now lets ensure the triangulated graph is
        # the same as Darwiche Huang fig. 2 pg. 13
        nodes = {node.name: node for node in huang_darwiche_moralized.nodes}
        assert set(nodes["f_a"].neighbours) == set(
            [nodes["f_b"], nodes["f_c"], nodes["f_d"], nodes["f_e"]]
        )
        assert set(nodes["f_b"].neighbours) == set([nodes["f_a"], nodes["f_d"]])
        assert set(nodes["f_c"].neighbours) == set(
            [nodes["f_a"], nodes["f_e"], nodes["f_g"]]
        )
        assert set(nodes["f_d"].neighbours) == set(
            [nodes["f_b"], nodes["f_f"], nodes["f_e"], nodes["f_a"]]
        )
        assert set(nodes["f_e"].neighbours) == set(
            [
                nodes["f_c"],
                nodes["f_f"],
                nodes["f_h"],
                nodes["f_d"],
                nodes["f_g"],
                nodes["f_a"],
            ]
        )
        assert set(nodes["f_f"].neighbours) == set([nodes["f_d"], nodes["f_e"]])
        assert set(nodes["f_g"].neighbours) == set(
            [nodes["f_c"], nodes["f_h"], nodes["f_e"]]
        )
        assert set(nodes["f_h"].neighbours) == set([nodes["f_e"], nodes["f_g"]])

    def test_triangulate_no_tie_break(self, huang_darwiche_moralized):
        # Now lets see what happens if
        # we dont enforce the tie-breakers...
        # It seems the triangulated graph is
        # different adding edges from d to c
        # and b to c
        # Will be interesting to see whether
        # inference will still be correct.
        triangulate(huang_darwiche_moralized)
        nodes = {node.name: node for node in huang_darwiche_moralized.nodes}
        assert set(nodes["f_a"].neighbours) == set([nodes["f_b"], nodes["f_c"]])
        assert set(nodes["f_b"].neighbours) == set(
            [nodes["f_a"], nodes["f_d"], nodes["f_c"]]
        )
        assert set(nodes["f_c"].neighbours) == set(
            [nodes["f_a"], nodes["f_e"], nodes["f_g"], nodes["f_b"], nodes["f_d"]]
        )
        assert set(nodes["f_d"].neighbours) == set(
            [nodes["f_b"], nodes["f_f"], nodes["f_e"], nodes["f_c"]]
        )
        assert set(nodes["f_e"].neighbours) == set(
            [nodes["f_c"], nodes["f_f"], nodes["f_h"], nodes["f_d"], nodes["f_g"]]
        )
        assert set(nodes["f_f"].neighbours) == set([nodes["f_d"], nodes["f_e"]])
        assert set(nodes["f_g"].neighbours) == set(
            [nodes["f_c"], nodes["f_h"], nodes["f_e"]]
        )
        assert set(nodes["f_h"].neighbours) == set([nodes["f_e"], nodes["f_g"]])

    def test_build_join_tree(self, huang_darwiche_dag):
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

        jt = build_join_tree(huang_darwiche_dag, priority_func_override)
        for node in jt.sepset_nodes:
            assert {n.clique for n in node.neighbours} == {node.sepset.X, node.sepset.Y}
        # clique nodes.

    def test_initialize_potentials(self, huang_darwiche_jt, huang_darwiche_dag):
        # Seems like there can be multiple assignments so
        # for this test we will set the assignments explicitely
        cliques = {node.name: node for node in huang_darwiche_jt.nodes}
        bbn_nodes = {node.name: node for node in huang_darwiche_dag.nodes}
        assignments = {
            cliques["Clique_ACE"]: [bbn_nodes["f_c"], bbn_nodes["f_e"]],
            cliques["Clique_ABD"]: [
                bbn_nodes["f_a"],
                bbn_nodes["f_b"],
                bbn_nodes["f_d"],
            ],
        }
        huang_darwiche_jt.initialize_potentials(assignments, huang_darwiche_dag)
        for node in huang_darwiche_jt.sepset_nodes:
            for v in node.potential_tt.values():
                assert v == 1

        # Note that in H&D there are two places that show
        # initial potentials, one is for ABD and AD
        # and the second is for ACE and CE
        # We should test both here but we must enforce
        # the assignments above because alternate and
        # equally correct Junction Trees will give
        # different potentials.
        def r(x):
            return round(x, 3)

        tt = cliques["Clique_ACE"].potential_tt
        assert r(tt[("a", True), ("c", True), ("e", True)]) == 0.21
        assert r(tt[("a", True), ("c", True), ("e", False)]) == 0.49
        assert r(tt[("a", True), ("c", False), ("e", True)]) == 0.18
        assert r(tt[("a", True), ("c", False), ("e", False)]) == 0.12
        assert r(tt[("a", False), ("c", True), ("e", True)]) == 0.06
        assert r(tt[("a", False), ("c", True), ("e", False)]) == 0.14
        assert r(tt[("a", False), ("c", False), ("e", True)]) == 0.48
        assert r(tt[("a", False), ("c", False), ("e", False)]) == 0.32

        tt = cliques["Clique_ABD"].potential_tt
        assert r(tt[("a", True), ("b", True), ("d", True)]) == 0.225
        assert r(tt[("a", True), ("b", True), ("d", False)]) == 0.025
        assert r(tt[("a", True), ("b", False), ("d", True)]) == 0.125
        assert r(tt[("a", True), ("b", False), ("d", False)]) == 0.125
        assert r(tt[("a", False), ("b", True), ("d", True)]) == 0.180
        assert r(tt[("a", False), ("b", True), ("d", False)]) == 0.020
        assert r(tt[("a", False), ("b", False), ("d", True)]) == 0.150
        assert r(tt[("a", False), ("b", False), ("d", False)]) == 0.150

    def test_jtclique_node_variable_names(self, huang_darwiche_jt):
        for node in huang_darwiche_jt.clique_nodes:
            if "ADE" in node.name:
                assert set(node.variable_names) == set(["a", "d", "e"])

    def test_propagate(self, huang_darwiche_jt, huang_darwiche_dag):
        jt_cliques = {node.name: node for node in huang_darwiche_jt.clique_nodes}
        assignments = huang_darwiche_jt.assign_clusters(huang_darwiche_dag)
        huang_darwiche_jt.initialize_potentials(assignments, huang_darwiche_dag)

        huang_darwiche_jt.propagate(starting_clique=jt_cliques["Clique_ACE"])
        tt = jt_cliques["Clique_DEF"].potential_tt
        assert r5(tt[(("d", False), ("e", True), ("f", True))]) == 0.00150
        assert r5(tt[(("d", True), ("e", False), ("f", True))]) == 0.00365
        assert r5(tt[(("d", False), ("e", False), ("f", True))]) == 0.16800
        assert r5(tt[(("d", True), ("e", True), ("f", True))]) == 0.00315
        assert r5(tt[(("d", False), ("e", False), ("f", False))]) == 0.00170
        assert r5(tt[(("d", True), ("e", True), ("f", False))]) == 0.31155
        assert r5(tt[(("d", False), ("e", True), ("f", False))]) == 0.14880
        assert r5(tt[(("d", True), ("e", False), ("f", False))]) == 0.36165

    def test_marginal(self, huang_darwiche_jt, huang_darwiche_dag):
        # The remaining marginals here come
        # from the module itself, however they
        # have been corrobarted by running
        # inference using the sampling inference
        # engine and the same results are
        # achieved.
        """
        +------+-------+----------+
        | Node | Value | Marginal |
        +------+-------+----------+
        | a    | False | 0.500000 |
        | a    | True  | 0.500000 |
        | b    | False | 0.550000 |
        | b    | True  | 0.450000 |
        | c    | False | 0.550000 |
        | c    | True  | 0.450000 |
        | d    | False | 0.320000 |
        | d    | True  | 0.680000 |
        | e    | False | 0.535000 |
        | e    | True  | 0.465000 |
        | f    | False | 0.823694 |
        | f    | True  | 0.176306 |
        | g    | False | 0.585000 |
        | g    | True  | 0.415000 |
        | h    | False | 0.176900 |
        | h    | True  | 0.823100 |
        +------+-------+----------+
        """
        bbn_nodes = {node.name: node for node in huang_darwiche_dag.nodes}
        assignments = huang_darwiche_jt.assign_clusters(huang_darwiche_dag)
        huang_darwiche_jt.initialize_potentials(assignments, huang_darwiche_dag)
        huang_darwiche_jt.propagate()

        # These test values come directly from
        # pg. 22 of H & D
        p_A = huang_darwiche_jt.marginal(bbn_nodes["f_a"])
        assert r3(p_A[(("a", True),)]) == 0.5
        assert r3(p_A[(("a", False),)]) == 0.5

        p_D = huang_darwiche_jt.marginal(bbn_nodes["f_d"])
        assert r3(p_D[(("d", True),)]) == 0.68
        assert r3(p_D[(("d", False),)]) == 0.32

        p_B = huang_darwiche_jt.marginal(bbn_nodes["f_b"])
        assert r3(p_B[(("b", True),)]) == 0.45
        assert r3(p_B[(("b", False),)]) == 0.55

        p_C = huang_darwiche_jt.marginal(bbn_nodes["f_c"])
        assert r3(p_C[(("c", True),)]) == 0.45
        assert r3(p_C[(("c", False),)]) == 0.55

        p_E = huang_darwiche_jt.marginal(bbn_nodes["f_e"])
        assert r3(p_E[(("e", True),)]) == 0.465
        assert r3(p_E[(("e", False),)]) == 0.535

        p_F = huang_darwiche_jt.marginal(bbn_nodes["f_f"])
        assert r3(p_F[(("f", True),)]) == 0.176
        assert r3(p_F[(("f", False),)]) == 0.824

        p_G = huang_darwiche_jt.marginal(bbn_nodes["f_g"])
        assert r3(p_G[(("g", True),)]) == 0.415
        assert r3(p_G[(("g", False),)]) == 0.585

        p_H = huang_darwiche_jt.marginal(bbn_nodes["f_h"])
        assert r3(p_H[(("h", True),)]) == 0.823
        assert r3(p_H[(("h", False),)]) == 0.177


def test_make_node_func():
    UPDATE = {
        "prize_door": [
            # For nodes that have no parents
            # use the empty list to specify
            # the conditioned upon variables
            # ie conditioned on the empty set
            [[], {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}]
        ],
        "guest_door": [[[], {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}]],
        "monty_door": [
            [[["prize_door", "A"], ["guest_door", "A"]], {"A": 0, "B": 0.5, "C": 0.5}],
            [[["prize_door", "A"], ["guest_door", "B"]], {"A": 0, "B": 0, "C": 1}],
            [[["prize_door", "A"], ["guest_door", "C"]], {"A": 0, "B": 1, "C": 0}],
            [[["prize_door", "B"], ["guest_door", "A"]], {"A": 0, "B": 0, "C": 1}],
            [[["prize_door", "B"], ["guest_door", "B"]], {"A": 0.5, "B": 0, "C": 0.5}],
            [[["prize_door", "B"], ["guest_door", "C"]], {"A": 1, "B": 0, "C": 0}],
            [[["prize_door", "C"], ["guest_door", "A"]], {"A": 0, "B": 1, "C": 0}],
            [[["prize_door", "C"], ["guest_door", "B"]], {"A": 1, "B": 0, "C": 0}],
            [[["prize_door", "C"], ["guest_door", "C"]], {"A": 0.5, "B": 0.5, "C": 0}],
        ],
    }

    node_func = make_node_func("prize_door", UPDATE["prize_door"])
    assert get_args(node_func) == ["prize_door"]
    assert node_func("A") == 1 / 3
    assert node_func("B") == 1 / 3
    assert node_func("C") == 1 / 3

    node_func = make_node_func("guest_door", UPDATE["guest_door"])
    assert get_args(node_func) == ["guest_door"]
    assert node_func("A") == 1 / 3
    assert node_func("B") == 1 / 3
    assert node_func("C") == 1 / 3

    node_func = make_node_func("monty_door", UPDATE["monty_door"])
    assert get_args(node_func) == ["guest_door", "prize_door", "monty_door"]
    assert node_func("A", "A", "A") == 0
    assert node_func("A", "A", "B") == 0.5
    assert node_func("A", "A", "C") == 0.5
    assert node_func("A", "B", "A") == 0
    assert node_func("A", "B", "B") == 0
    assert node_func("A", "B", "C") == 1
    assert node_func("A", "C", "A") == 0
    assert node_func("A", "C", "B") == 1
    assert node_func("A", "C", "C") == 0
    assert node_func("B", "A", "A") == 0
    assert node_func("B", "A", "B") == 0
    assert node_func("B", "A", "C") == 1
    assert node_func("B", "B", "A") == 0.5
    assert node_func("B", "B", "B") == 0
    assert node_func("B", "B", "C") == 0.5
    assert node_func("B", "C", "A") == 1
    assert node_func("B", "C", "B") == 0
    assert node_func("B", "C", "C") == 0
    assert node_func("C", "A", "A") == 0
    assert node_func("C", "A", "B") == 1
    assert node_func("C", "A", "C") == 0
    assert node_func("C", "B", "A") == 1
    assert node_func("C", "B", "B") == 0
    assert node_func("C", "B", "C") == 0
    assert node_func("C", "C", "A") == 0.5
    assert node_func("C", "C", "B") == 0.5
    assert node_func("C", "C", "C") == 0


def close_enough(x, y, r=3):
    return round(x, r) == round(y, r)


def test_build_bbn_from_conditionals():
    UPDATE = {
        "prize_door": [
            # For nodes that have no parents
            # use the empty list to specify
            # the conditioned upon variables
            # ie conditioned on the empty set
            [[], {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}]
        ],
        "guest_door": [[[], {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}]],
        "monty_door": [
            [[["prize_door", "A"], ["guest_door", "A"]], {"A": 0, "B": 0.5, "C": 0.5}],
            [[["prize_door", "A"], ["guest_door", "B"]], {"A": 0, "B": 0, "C": 1}],
            [[["prize_door", "A"], ["guest_door", "C"]], {"A": 0, "B": 1, "C": 0}],
            [[["prize_door", "B"], ["guest_door", "A"]], {"A": 0, "B": 0, "C": 1}],
            [[["prize_door", "B"], ["guest_door", "B"]], {"A": 0.5, "B": 0, "C": 0.5}],
            [[["prize_door", "B"], ["guest_door", "C"]], {"A": 1, "B": 0, "C": 0}],
            [[["prize_door", "C"], ["guest_door", "A"]], {"A": 0, "B": 1, "C": 0}],
            [[["prize_door", "C"], ["guest_door", "B"]], {"A": 1, "B": 0, "C": 0}],
            [[["prize_door", "C"], ["guest_door", "C"]], {"A": 0.5, "B": 0.5, "C": 0}],
        ],
    }
    g = build_bbn_from_conditionals(UPDATE)
    result = g.query()
    assert close_enough(result[("guest_door", "A")], 0.333)
    assert close_enough(result[("guest_door", "B")], 0.333)
    assert close_enough(result[("guest_door", "C")], 0.333)
    assert close_enough(result[("monty_door", "A")], 0.333)
    assert close_enough(result[("monty_door", "B")], 0.333)
    assert close_enough(result[("monty_door", "C")], 0.333)
    assert close_enough(result[("prize_door", "A")], 0.333)
    assert close_enough(result[("prize_door", "B")], 0.333)
    assert close_enough(result[("prize_door", "C")], 0.333)

    result = g.query(guest_door="A", monty_door="B")
    assert close_enough(result[("guest_door", "A")], 1)
    assert close_enough(result[("guest_door", "B")], 0)
    assert close_enough(result[("guest_door", "C")], 0)
    assert close_enough(result[("monty_door", "A")], 0)
    assert close_enough(result[("monty_door", "B")], 1)
    assert close_enough(result[("monty_door", "C")], 0)
    assert close_enough(result[("prize_door", "A")], 0.333)
    assert close_enough(result[("prize_door", "B")], 0)
    assert close_enough(result[("prize_door", "C")], 0.667)


def valid_sample(samples, query_result):
    """For a group of samples from
    a query result ensure that
    the sample is approximately equivalent
    to the query_result which is the
    true distribution."""
    counts = Counter()
    for sample in samples:
        for var, val in sample.items():
            counts[(var, val)] += 1
    # Now lets normalize for each count...
    differences = []
    for k, v in counts.items():
        counts[k] = v / len(samples)
        difference = abs(counts.get(k, 0) - query_result[k])
        differences.append(difference)
    return all(not round(difference, 2) > 0.01 for difference in differences)


def test_draw_sample_sprinkler(sprinkler_bbn):

    query_result = sprinkler_bbn.query()
    samples = sprinkler_bbn.draw_samples({}, 10000)
    assert valid_sample(samples, query_result)


def test_repr():

    assert repr(Node("test")) == "<Node test>"
    assert repr(UndirectedNode("test")) == "<UndirectedNode test>"
    assert (
        repr(BBNNode(get_original_factors))
        == "<BBNNode get_original_factors (['factors'])>"
    )
    assert (
        repr(JoinTreeCliqueNode(UndirectedNode("test")))
        == "<JoinTreeCliqueNode: <UndirectedNode test>>"
    )


def test_exception(sprinkler_bbn):
    with pytest.raises(VariableValueNotInDomainError):
        sprinkler_bbn.query(rain="No")
    with pytest.raises(VariableNotInGraphError):
        sprinkler_bbn.query(sunny="True")


def test_make_key():
    class DummyTest:
        def __init__(self, value):

            self.value = value

        def dummy_method(self, value):  # Add this method to by pass linting
            self.value = value

    test = DummyTest(8)
    test.dummy_method(10)
    with pytest.raises(ValueError, match=r"Unexpected type"):
        make_key(test)


def test_insert_duplicate_clique(huang_darwiche_moralized):

    cliques, _ = triangulate(huang_darwiche_moralized, priority_func)

    forest = set()
    for clique in cliques:
        jt_node = JoinTreeCliqueNode(clique)
        clique.node = jt_node
        tree = JoinTree([jt_node])
        forest.add(tree)

    s = SepSet(cliques[0], cliques[0])
    assert s.insertable(forest) is False
    s_copy = copy.deepcopy(s)
    s.insert(forest)
    assert len(s.X.node.neighbours) > len(s_copy.X.node.neighbours)
