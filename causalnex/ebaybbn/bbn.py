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
"""Data Structures to represent a BBN as a DAG."""

import copy
import heapq
import logging
from collections import defaultdict
from io import StringIO
from itertools import combinations, product
from random import choice, random

from .exceptions import VariableNotInGraphError, VariableValueNotInDomainError
from .graph import Node, UndirectedGraph, UndirectedNode, connect
from .utils import get_args, get_original_factors

# from .bayesian import GREEN, NORMAL
GREEN = "\033[92m"
NORMAL = "\033[0m"


class BBNNode(Node):
    def __init__(self, factor):
        super(BBNNode, self).__init__(factor.__name__)
        self.func = factor
        self.argspec = get_args(factor)

    def __repr__(self):
        return "<BBNNode %s (%s)>" % (self.name, self.argspec)


class BBN:
    """A Directed Acyclic Graph"""

    def __init__(self, nodes_dict, name=None, domains={}):
        self.nodes = list(nodes_dict.values())
        self.vars_to_nodes = nodes_dict
        self.domains = domains
        # For each node we want
        # to explicitly record which
        # variable it 'introduced'.
        # Note that we cannot record
        # this duing Node instantiation
        # becuase at that point we do
        # not yet know *which* of the
        # variables in the argument
        # list is the one being modeled
        # by the function. (Unless there
        # is only one argument)
        for variable_name, node in list(nodes_dict.items()):
            node.variable_name = variable_name

    def get_graphviz_source(self):
        fh = StringIO()
        fh.write("digraph G {\n")
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in sorted(self.nodes, key=lambda x: x.name):
            fh.write('  %s [ shape="ellipse" color="blue"];\n' % node.name)
            for child in node.children:
                edge = (node.name, child.name)
                edges.add(edge)
        for source, target in sorted(edges, key=lambda x: (x[0], x[1])):
            fh.write("  %s -> %s;\n" % (source, target))
        fh.write("}\n")
        return fh.getvalue()

    def build_join_tree(self):
        jt = build_join_tree(self)
        return jt

    def validate_keyvals(self, **kwds):
        """
        When evidence in the form of
        keyvals are provided to the .query() method
        validate that all keys match a variable name
        and that all vals are in the domain of
        the variable
        """
        vars = set([n.variable_name for n in self.nodes])
        for k, v in list(kwds.items()):
            if k not in vars:
                raise VariableNotInGraphError(k)
            domain = self.domains.get(k, (True, False))
            if v not in domain:
                raise VariableValueNotInDomainError(f"{k}={v}")
        return True

    def query(self, **kwds):
        # First check that the keyvals
        # provided are valid for this graph...
        self.validate_keyvals(**kwds)
        jt = self.build_join_tree()
        assignments = jt.assign_clusters(self)
        jt.initialize_potentials(assignments, self, kwds)

        jt.propagate()
        marginals = {}
        normalizers = defaultdict(float)

        for node in self.nodes:
            for k, v in list(jt.marginal(node).items()):
                # For a single node the
                # key for the marginal tt always
                # has just one argument so we
                # will unpack it here
                marginals[k[0]] = v
                # If we had any evidence then we
                # need to normalize all the variables
                # not evidenced.
                if kwds:
                    normalizers[k[0][0]] += v

        if kwds:
            for k, v in marginals.items():
                if normalizers[k[0]] != 0:
                    marginals[k] /= normalizers[k[0]]

        return marginals

    def draw_samples(self, query={}, n=1):
        """query is a dict of currently evidenced
        variables and is none by default."""
        samples = []
        result_cache = {}
        # We need to add evidence variables to the sample...
        while len(samples) < n:
            sample = dict(query)
            while len(sample) < len(self.nodes):
                next_node = choice(
                    [node for node in self.nodes if node.variable_name not in sample]
                )
                key = tuple(sorted(sample.items()))
                if key not in result_cache:
                    result_cache[key] = self.query(**sample)
                result = result_cache[key]
                var_density = [
                    r
                    for r in list(result.items())
                    if r[0][0] == next_node.variable_name
                ]
                cumulative_density = var_density[:1]
                for key, mass in var_density[1:]:
                    cumulative_density.append((key, cumulative_density[-1][1] + mass))
                r = random()
                i = 0
                while r > cumulative_density[i][1]:
                    i += 1
                sample[next_node.variable_name] = cumulative_density[i][0][1]
            samples.append(sample)
        return samples


class JoinTree(UndirectedGraph):
    def __init__(self, nodes, name=None):
        super(JoinTree, self).__init__(nodes, name)
        self._sensitivity_flag = False

    @property
    def sepset_nodes(self):
        return [n for n in self.nodes if isinstance(n, JoinTreeSepSetNode)]

    @property
    def clique_nodes(self):
        return [n for n in self.nodes if isinstance(n, JoinTreeCliqueNode)]

    def initialize_potentials(self, assignments, bbn, evidence={}):
        # Step 1, assign 1 to each cluster and sepset
        for node in self.nodes:
            tt = {}

            vals = []
            variables = node.variable_names
            # Lets sort the variables here so that
            # the variable names in the keys in
            # the tt are always sorted.
            variables.sort()
            for variable in variables:
                domain = bbn.domains.get(variable, [True, False])
                vals.append(list(product([variable], domain)))
            permutations = product(*vals)
            for permutation in permutations:
                tt[permutation] = 1
            node.potential_tt = tt

        # Step 2: Note that in H&D the assignments are
        # done as part of step 2 however we have
        # seperated the assignment algorithm out and
        # done these prior to step 1.
        # Now for each assignment we want to
        # generate a truth-table from the
        # values of the bbn truth-tables that are
        # assigned to the clusters...

        for clique, bbn_nodes in assignments.items():
            tt = {}
            vals = []
            variables = list(clique.variable_names)
            variables.sort()
            for variable in variables:
                domain = bbn.domains.get(variable, [True, False])
                vals.append(list(product([variable], domain)))
            permutations = product(*vals)
            for permutation in permutations:
                argvals = dict(permutation)
                potential = 1
                for bbn_node in bbn_nodes:
                    bbn_node.clique = clique
                    # We could handle evidence here
                    # by altering the potential_tt.
                    # This is slightly different to
                    # the way that H&D do it.

                    arg_list = []
                    for arg_name in get_args(bbn_node.func):
                        arg_list.append(argvals[arg_name])

                    potential *= bbn_node.func(*arg_list)
                tt[permutation] = potential
            clique.potential_tt = tt

        if not evidence:
            # We dont need to deal with likelihoods
            # if we dont have any evidence.
            return

        # Step 2b: Set each liklihood element ^V(v) to 1
        self.initial_likelihoods(assignments, bbn)
        for clique, bbn_nodes in assignments.items():
            for node in bbn_nodes:
                if node.variable_name in evidence:
                    for k, v in list(clique.potential_tt.items()):
                        # Encode the evidence in
                        # the clique potential...
                        for variable, value in k:
                            if variable == node.variable_name:
                                if value != evidence[variable]:
                                    clique.potential_tt[k] = 0

    def initial_likelihoods(self, assignments, bbn):
        # TODO: Since this is the same every time we should probably
        # cache it.
        likelihood = defaultdict(dict)
        for clique, bbn_nodes in assignments.items():
            for node in bbn_nodes:
                for value in bbn.domains.get(node.variable_name, [True, False]):
                    likelihood[(node.variable_name, value)] = 1
        return likelihood

    def assign_clusters(self, bbn):
        assignments_by_family = {}
        assignments_by_clique = defaultdict(list)
        assigned = set()
        for node in bbn.nodes:
            args = get_args(node.func)
            if len(args) == 1:
                # If the func has only 1 arg
                # it means that it does not
                # specify a conditional probability
                # This is where H&D is a bit vague
                # but it seems to imply that we
                # do not assign it to any
                # clique.
                # Revising this for now as I dont
                # think its correct, I think
                # all CPTs need to be assigned
                # once and once only. The example
                # in H&D just happens to be a clique
                # that f_a could have been assigned
                # to but wasnt presumably because
                # it got assigned somewhere else.
                pass
                # continue
            # Now we need to find a cluster that
            # is a superset of the Family(v)
            # Family(v) is defined by D&H to
            # be the union of v and parents(v)
            family = set(args)
            # At this point we need to know which *variable*
            # a BBN node represents. Up to now we have
            # not *explicitely* specified this, however
            # we have been following some conventions
            # so we could just use this convention for
            # now. Need to come back to this to
            # perhaps establish the variable at
            # build bbn time...

            containing_cliques = [
                clique_node
                for clique_node in self.clique_nodes
                if (set(clique_node.variable_names).issuperset(family))
            ]
            assert len(containing_cliques) >= 1
            for clique in containing_cliques:
                if node in assigned:
                    # Make sure we assign all original
                    # PMFs only once each
                    break
                assignments_by_clique[clique].append(node)
                assigned.add(node)
            assignments_by_family[tuple(family)] = containing_cliques
        return assignments_by_clique

    def propagate(self, starting_clique=None):
        """Refer to H&D pg. 20"""

        # Step 1 is to choose an arbitrary clique cluster
        # as starting cluster
        if starting_clique is None:
            starting_clique = self.clique_nodes[0]
        logging.debug("Starting propagating messages from: %s", starting_clique.name)
        # Step 2: Unmark all clusters, call collect_evidence(X)
        for node in self.clique_nodes:
            node.marked = False
            logging.debug("Marking node as not visited Node: %s", node.name)
        self.collect_evidence(sender=starting_clique)

        # Step 3: Unmark all clusters, call distribute_evidence(X)
        for node in self.clique_nodes:
            node.marked = False

        self.distribute_evidence(starting_clique)

    def collect_evidence(self, sender=None, receiver=None):

        logging.debug("Collect evidence from %s", sender.name)
        # Step 1, Mark X
        sender.marked = True

        # Step 2, call collect_evidence on Xs unmarked
        # neighbouring clusters.
        for neighbouring_clique in sender.neighbouring_cliques:
            if not neighbouring_clique.marked:
                logging.debug(
                    "Collect evidence from %s to %s",
                    neighbouring_clique.name,
                    sender.name,
                )
                self.collect_evidence(sender=neighbouring_clique, receiver=sender)
        # Step 3, pass message from sender to receiver
        if receiver is not None:
            sender.pass_message(receiver)

    def distribute_evidence(self, sender=None, receiver=None):
        logging.debug("Distribute evidence from: %s", sender.name)
        # Step 1, Mark X
        sender.marked = True

        # Step 2, pass a messagee from X to each of its
        # unmarked neighbouring clusters
        for neighbouring_clique in sender.neighbouring_cliques:
            if not neighbouring_clique.marked:
                logging.debug(
                    "Pass message from: %s to %s", sender.name, neighbouring_clique.name
                )
                sender.pass_message(neighbouring_clique)

        # Step 3, call distribute_evidence on Xs unmarked neighbours
        for neighbouring_clique in sender.neighbouring_cliques:
            if not neighbouring_clique.marked:
                logging.debug(
                    "Distribute evidence from: %s to %s",
                    neighbouring_clique.name,
                    sender.name,
                )
                self.distribute_evidence(sender=neighbouring_clique, receiver=sender)

    def marginal(self, bbn_node):
        """Remember that the original
        variables that we are interested in
        are actually in the bbn. However
        when we constructed the JT we did
        it out of the moralized graph.
        This means the cliques refer to
        the nodes in the moralized graph
        and not the nodes in the BBN.
        For efficiency we should come back
        to this and add some pointers
        or an index.
        """

        # First we will find the JT nodes that
        # contain the bbn_node ie all the nodes
        # that are either cliques or sepsets
        # that contain the bbn_node
        # Note that for efficiency we
        # should probably have an index
        # cached in the bbn and/or the jt.
        containing_nodes = []

        for node in self.clique_nodes:
            if bbn_node.name in [n.name for n in node.clique.nodes]:
                containing_nodes.append(node)
                # In theory it doesnt matter which one we
                # use so we could bale out after we
                # find the first one
                # TODO: With some better indexing we could
                # avoid searching for this node every time...
        clique_node = containing_nodes[0]
        tt = defaultdict(float)
        for k, v in list(clique_node.potential_tt.items()):
            entry = transform(k, clique_node.variable_names, [bbn_node.variable_name])
            tt[entry] += v

        # Now if this node was evidenced we need to normalize
        # over the values...
        # TODO: It will be safer to copy the defaultdict to a regular dict
        return tt


class Clique(object):
    def __init__(self, cluster):
        self.nodes = cluster

    def __repr__(self):
        vars = sorted([n.variable_name for n in self.nodes])
        return "Clique_%s" % "".join([v.upper() for v in vars])


def transform(x, X, R):
    """Transform a Potential Truth Table
    Entry into a different variable space.
    For example if we have the
    entry [True, True, False] representing
    values of variable [A, B, C] in X
    and we want to transform into
    R which has variables [C, A] we
    will return the entry [False, True].
    Here X represents the argument list
    for the clique set X and R represents
    the argument list for the sepset.
    This implies that R is always a subset
    of X"""
    entry = []
    for r in R:
        pos = X.index(r)
        entry.append(x[pos])
    return tuple(entry)


class JoinTreeCliqueNode(UndirectedNode):
    def __init__(self, clique):
        super(JoinTreeCliqueNode, self).__init__(clique.__repr__())
        self.clique = clique
        self.potential_psi = None

        # Now we create a pointer to
        # this clique node as the "parent" clique
        # node of each node in the cluster.
        # for node in self.clique.nodes:
        #    node.parent_clique = self
        # This is not quite correct, the
        # parent cluster as defined by H&D
        # is *a* cluster than is a superset
        # of Family(v)

    @property
    def variable_names(self):
        """Return the set of variable names
        that this clique represents"""
        var_names = []
        for node in self.clique.nodes:
            var_names.append(node.variable_name)
        return sorted(var_names)

    @property
    def neighbouring_cliques(self):
        """Return the neighbouring cliques
        this is used during the propagation algorithm.

        """
        neighbours = set()
        for sepset_node in self.neighbours:
            # All *immediate* neighbours will
            # be sepset nodes, its the neighbours of
            # these sepsets that form the nodes
            # clique neighbours (excluding itself)
            for clique_node in sepset_node.neighbours:
                if clique_node is not self:
                    neighbours.add(clique_node)
        return neighbours

    def pass_message(self, target):
        """Pass a message from this node to the
        recipient node during propagation.

        NB: It may turnout at this point that
        after initializing the potential
        Truth table on the JT we could quite
        simply construct a factor graph
        from the JT and use the factor
        graph sum product propagation.
        In theory this should be the same
        and since the semantics are already
        worked out it would be easier."""

        # Find the sepset node between the
        # source and target nodes.
        sepset_node = list(set(self.neighbours).intersection(target.neighbours))[0]

        logging.debug("Pass message from: %s to: %s", self.name, target.name)
        # Step 1: projection
        logging.debug("Project into the Sepset node: %s", str(sepset_node))
        self.project(sepset_node)

        logging.debug(" Send the summed marginals to the target: %s ", str(sepset_node))

        # Step 2 absorbtion
        self.absorb(sepset_node, target)

    def project(self, sepset_node):
        """See page 20 of PPTC.
        We assign a new potential tt to
        the sepset which consists of the
        potential of the source node
        with all variables not in R marginalized.
        """
        assert sepset_node in self.neighbours
        # First we make a copy of the
        # old potential tt

        # Now we assign a new potential tt
        # to the sepset by marginalizing
        # out the variables from X that are not
        # in the sepset
        # ToDO test and check this function
        # Todo check on the old sepset potentials and when they will be set

        sepset_node.potential_tt_old = copy.deepcopy(sepset_node.potential_tt)
        tt = defaultdict(float)
        for k, v in self.potential_tt.items():
            entry = transform(k, self.variable_names, sepset_node.variable_names)
            tt[entry] += v
        sepset_node.potential_tt = tt

    def absorb(self, sepset, target):
        # Assign a new potential tt to
        # Y (the target)
        logging.debug(
            "Absorb potentails from sepset node %s into clique %s",
            sepset.name,
            target.name,
        )
        tt = {}

        for k, v in list(target.potential_tt.items()):
            # For each entry we multiply by
            # sepsets new value and divide
            # by sepsets old value...
            # Note that nowhere in H&D is
            # division on potentials defined.
            # However in Barber page 12
            # an equation implies that
            # the the division is equivalent
            # to the original assignment.
            # For now we will assume entry-wise
            # division which seems logical.
            entry = transform(k, target.variable_names, sepset.variable_names)
            if target.potential_tt[k] == 0:
                tt[k] = 0
            else:
                tt[k] = target.potential_tt[k] * (
                    sepset.potential_tt[entry] / sepset.potential_tt_old[entry]
                )
        # assign the new potentials to the node
        target.potential_tt = tt

    def __repr__(self):
        return "<JoinTreeCliqueNode: %s>" % self.clique


class SepSet(object):
    def __init__(self, X, Y):
        """X and Y are cliques represented as sets."""
        self.X = X
        self.Y = Y
        self.label = list(X.nodes.intersection(Y.nodes))

    @property
    def mass(self):
        return len(self.label)

    @property
    def cost(self):
        """Since cost is used as a tie-breaker
        and is an optimization for inference time
        we will punt on it for now. Instead we
        will just use the assumption that all
        variables in X and Y are binary and thus
        use a weight of 2.
        TODO: come back to this and compute
        actual weights
        """
        return 2 ** len(self.X.nodes) + 2 ** len(self.Y.nodes)

    def insertable(self, forest):
        """A sepset can only be inserted
        into the JT if the cliques it
        separates are NOT already on
        the same tree.
        NOTE: For efficiency we should
        add an index that indexes cliques
        into the trees in the forest."""
        X_trees = [t for t in forest if self.X in [n.clique for n in t.clique_nodes]]
        Y_trees = [t for t in forest if self.Y in [n.clique for n in t.clique_nodes]]
        assert len(X_trees) == 1
        assert len(Y_trees) == 1
        if X_trees[0] is not Y_trees[0]:
            return True
        return False

    def insert(self, forest):
        """Inserting this sepset into
        a forest, providing the two
        cliques are in different trees,
        means that effectively we are
        collapsing the two trees into
        one. We will explicitely perform
        this collapse by adding the
        sepset node into the tree
        and adding edges between itself
        and its clique node neighbours.
        Finally we must remove the
        second tree from the forest
        as it is now joined to the
        first.
        """
        X_tree = [t for t in forest if self.X in [n.clique for n in t.clique_nodes]][0]
        Y_tree = [t for t in forest if self.Y in [n.clique for n in t.clique_nodes]][0]

        # Now create and insert a sepset node into the Xtree
        ss_node = JoinTreeSepSetNode(self, self)
        X_tree.nodes.append(ss_node)

        # And connect them
        self.X.node.neighbours.append(ss_node)
        ss_node.neighbours.append(self.X.node)

        # Now lets keep the X_tree and drop the Y_tree
        # this means we need to copy all the nodes
        # in the Y_tree that are not already in the X_tree
        for node in Y_tree.nodes:
            if node in X_tree.nodes:
                continue
            X_tree.nodes.append(node)

        # Now connect the sepset node to the
        # Y_node (now residing in the X_tree)
        self.Y.node.neighbours.append(ss_node)
        ss_node.neighbours.append(self.Y.node)

        # And finally we must remove the Y_tree from
        # the forest...
        forest.remove(Y_tree)

    def __repr__(self):
        return "SepSet_%s" % "".join(
            # [x.name[2:].upper() for x in list(self.label)])
            [x.variable_name.upper() for x in list(self.label)]
        )


class JoinTreeSepSetNode(UndirectedNode):
    def __init__(self, name, sepset):
        super(JoinTreeSepSetNode, self).__init__(name)
        self.sepset = sepset
        self.potential_psi = None

    @property
    def variable_names(self):
        """Return the set of variable names
        that this sepset represents"""
        # TODO: we are assuming here
        # that X and Y are each separate
        # variables from the BBN which means
        # we are assuming that the sepsets
        # always contain only 2 nodes.
        # Need to check whether this is
        # the case.
        return sorted([x.variable_name for x in self.sepset.label])

    def __repr__(self):
        return "<JoinTreeSepSetNode: %s>" % self.sepset


def build_bbn(*args, **kwds):
    """Builds a BBN Graph from
    a list of functions and domains"""
    variables = set()
    domains = kwds.get("domains", {})
    name = kwds.get("name")
    factor_nodes = {}

    if isinstance(args[0], list):
        # Assume the functions were all
        # passed in a list in the first
        # argument. This makes it possible
        # to build very large graphs with
        # more than 255 functions, since
        # Python functions are limited to
        # 255 arguments.
        args = args[0]

    for factor in args:
        factor_args = get_args(factor)
        variables.update(factor_args)
        bbn_node = BBNNode(factor)
        factor_nodes[factor.__name__] = bbn_node

    # Now lets create the connections
    # To do this we need to find the
    # factor node representing the variables
    # in a child factors argument and connect
    # it to the child node.

    # Note that calling original_factors
    # here can break build_bbn if the
    # factors do not correctly represent
    # a BBN.
    original_factors = get_original_factors(list(factor_nodes.values()))
    for factor_node in list(factor_nodes.values()):
        factor_args = get_args(factor_node)
        parents = [
            original_factors[arg]
            for arg in factor_args
            if original_factors[arg] != factor_node
        ]
        for parent in parents:
            connect(parent, factor_node)
    bbn = BBN(original_factors, name=name)
    bbn.domains = domains

    return bbn


def make_node_func(variable_name, conditions):
    # We will enforce the following
    # convention.
    # The ordering of arguments will
    # be firstly the parent variables
    # in alphabetical order, followed
    # always by the child variable
    tt = {}
    domain = set()
    for givens, conditionals in conditions:
        key = []
        for parent_name, val in sorted(givens):
            key.append((parent_name, val))
        # Now we will sort the
        # key before we add the child
        # node.
        # key.sort(key=lambda x: x[0])

        # Now for each value in
        # the conditional probabilities
        # we will add a new key
        for value, prob in list(conditionals.items()):
            key_ = tuple(key + [(variable_name, value)])
            domain.add(value)
            tt[key_] = prob

    argspec = [k[0] for k in key_]

    def node_func(*args):
        key = []
        for arg, val in zip(argspec, args):
            key.append((arg, val))
        return tt[tuple(key)]

    node_func.argspec = argspec
    node_func._domain = domain
    node_func.__name__ = "f_" + variable_name
    return node_func


def build_bbn_from_conditionals(conds):
    node_funcs = []
    domains = {}
    for variable_name, cond_tt in list(conds.items()):
        node_func = make_node_func(variable_name, cond_tt)
        node_funcs.append(node_func)
        domains[variable_name] = node_func._domain
    return build_bbn(*node_funcs, domains=domains)


def make_undirected_copy(dag):
    """Returns an exact copy of the dag
    except that direction of edges are dropped."""
    nodes = {}

    for node in dag.nodes:
        undirected_node = UndirectedNode(name=node.name)
        undirected_node.func = node.func
        undirected_node.argspec = node.argspec
        undirected_node.variable_name = node.variable_name
        nodes[node.name] = undirected_node
    # Now we need to traverse the original
    # nodes once more and add any parents
    # or children as neighbours.
    for node in dag.nodes:
        for parent in node.parents:
            nodes[node.name].neighbours.append(nodes[parent.name])
            nodes[parent.name].neighbours.append(nodes[node.name])

    g = UndirectedGraph(list(nodes.values()))
    return g


def make_moralized_copy(gu, dag):
    """gu is an undirected graph being
    a copy of dag."""
    gm = copy.deepcopy(gu)
    gm_nodes = dict([(node.name, node) for node in gm.nodes])
    for node in dag.nodes:
        for parent_1, parent_2 in combinations(node.parents, 2):
            if gm_nodes[parent_1.name] not in gm_nodes[parent_2.name].neighbours:
                gm_nodes[parent_2.name].neighbours.append(gm_nodes[parent_1.name])
            if gm_nodes[parent_2.name] not in gm_nodes[parent_1.name].neighbours:
                gm_nodes[parent_1.name].neighbours.append(gm_nodes[parent_2.name])
    return gm


def priority_func(node):
    """Specify the rules for computing
    priority of a node. See Harwiche and Wang pg 12.
    """
    # We need to calculate the number of edges
    # that would be added.
    # For each node, we need to connect all
    # of the nodes in itself and its neighbours
    # (the "cluster") which are not already
    # connected. This will be the primary
    # key value in the heap.
    # We need to fix the secondary key, right
    # now its just 2 (because mostly the variables
    # will be discrete binary)
    introduced_arcs = 0
    cluster = [node] + node.neighbours
    for node_a, node_b in combinations(cluster, 2):
        if node_a not in node_b.neighbours:
            assert node_b not in node_a.neighbours
            introduced_arcs += 1
    return [introduced_arcs, 2]  # TODO: Fix this to look at domains


def construct_priority_queue(nodes, priority_func=priority_func):
    pq = []
    for node_name, node in nodes.items():
        entry = priority_func(node) + [node.name]
        heapq.heappush(pq, entry)
    return pq


def record_cliques(cliques, cluster):
    """We only want to save the cluster
    if it is not a subset of any clique
    already saved.
    Argument cluster must be a set"""
    if any([cluster.issubset(c.nodes) for c in cliques]):
        return
    cliques.append(Clique(cluster))


def triangulate(gm, priority_func=priority_func):
    """Triangulate the moralized Graph. (in Place)
    and return the cliques of the triangulated
    graph as well as the elimination ordering."""

    # First we will make a copy of gm...
    gm_ = copy.deepcopy(gm)

    # Now we will construct a priority q using
    # the standard library heapq module.
    # See docs for example of priority q tie
    # breaking. We will use a 3 element list
    # with entries as follows:
    #   - Number of edges added if V were selected
    #   - Weight of V (or cluster)
    #   - Pointer to node in gm_
    # Note that its unclear from Huang and Darwiche
    # what is meant by the "number of values of V"
    gmnodes = dict([(node.name, node) for node in gm.nodes])
    elimination_ordering = []
    cliques = []
    while True:
        gm_nodes = dict([(node.name, node) for node in gm_.nodes])
        if not gm_nodes:
            break
        pq = construct_priority_queue(gm_nodes, priority_func)
        # Now we select the first node in
        # the priority q and any arcs that
        # should be added in order to fully connect
        # the cluster should be added to both
        # gm and gm_
        v = gm_nodes[pq[0][2]]
        cluster = [v] + v.neighbours
        for node_a, node_b in combinations(cluster, 2):
            if node_a not in node_b.neighbours:
                node_b.neighbours.append(node_a)
                node_a.neighbours.append(node_b)
                # Now also add this new arc to gm...
                gmnodes[node_b.name].neighbours.append(gmnodes[node_a.name])
                gmnodes[node_a.name].neighbours.append(gmnodes[node_b.name])
        gmcluster = set([gmnodes[c.name] for c in cluster])
        record_cliques(cliques, gmcluster)
        # Now we need to remove v from gm_...
        # This means we also have to remove it from all
        # of its neighbours that reference it...
        for neighbour in v.neighbours:
            neighbour.neighbours.remove(v)
        gm_.nodes.remove(v)
        elimination_ordering.append(v.name)
    return cliques, elimination_ordering


def build_join_tree(dag, clique_priority_func=priority_func):
    # First we will create an undirected copy
    # of the dag
    gu = make_undirected_copy(dag)

    # Now we create a copy of the undirected graph
    # and connect all pairs of parents that are
    # not already parents called the 'moralized' graph.
    gm = make_moralized_copy(gu, dag)

    # Now we triangulate the moralized graph...
    cliques, elimination_ordering = triangulate(gm, clique_priority_func)

    # Now we initialize the forest and sepsets
    # Its unclear from Darwiche Huang whether we
    # track a sepset for each tree or whether its
    # a global list????
    # We will implement the Join Tree as an undirected
    # graph for now...

    # First initialize a set of graphs where
    # each graph initially consists of just one
    # node for the clique. As these graphs get
    # populated with sepsets connecting them
    # they should collapse into a single tree.
    forest = set()
    for clique in cliques:
        jt_node = JoinTreeCliqueNode(clique)
        # Track a reference from the clique
        # itself to the node, this will be
        # handy later... (alternately we
        # could just collapse clique and clique
        # node into one class...
        clique.node = jt_node
        tree = JoinTree([jt_node])
        forest.add(tree)

    # Initialize the SepSets
    S = set()  # track the sepsets
    for X, Y in combinations(cliques, 2):
        if X.nodes.intersection(Y.nodes):
            S.add(SepSet(X, Y))
    sepsets_inserted = 0
    while sepsets_inserted < (len(cliques) - 1):
        # Adding in name to make this sort deterministic
        deco = [(s, -1 * s.mass, s.cost, s.__repr__()) for s in S]
        deco.sort(key=lambda x: x[1:])
        candidate_sepset = deco[0][0]
        for candidate_sepset, _, _, _ in deco:
            if candidate_sepset.insertable(forest):
                # Insert into forest and remove the sepset
                candidate_sepset.insert(forest)
                S.remove(candidate_sepset)
                sepsets_inserted += 1
                break

    assert len(forest) == 1
    jt = list(forest)[0]
    return jt
