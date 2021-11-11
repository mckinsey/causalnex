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
This module contains the implementation of ``InferenceEngine``.

``InferenceEngine`` provides tools to make inferences based on interventions and observations.
"""
import copy
import inspect
import re
import types
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Union

import networkx as nx
import pandas as pd
from pathos import multiprocessing

from causalnex.ebaybbn import build_bbn
from causalnex.network import BayesianNetwork


class InferenceEngine:
    """
    An ``InferenceEngine`` provides methods to query marginals based on observations and
    make interventions (Do-Calculus) on a ``BayesianNetwork``.

    Example:
    ::
        >>> # Create a Bayesian Network with a manually defined DAG
        >>> from causalnex.structure.structuremodel import StructureModel
        >>> from causalnex.network import BayesianNetwork
        >>> from causalnex.inference import InferenceEngine
        >>>
        >>> sm = StructureModel()
        >>> sm.add_edges_from([
        >>>                    ('rush_hour', 'traffic'),
        >>>                    ('weather', 'traffic')
        >>>                    ])
        >>> data = pd.DataFrame({
        >>>                      'rush_hour': [True, False, False, False, True, False, True],
        >>>                      'weather': ['Terrible', 'Good', 'Bad', 'Good', 'Bad', 'Bad', 'Good'],
        >>>                      'traffic': ['heavy', 'light', 'heavy', 'light', 'heavy', 'heavy', 'heavy']
        >>>                      })
        >>> bn = BayesianNetwork(sm)
        >>> # Inference can only be performed on the `BayesianNetwork` with learned nodes states and CPDs
        >>> bn = bn.fit_node_states_and_cpds(data)
        >>>
        >>> # Create an `InferenceEngine` to query marginals and make interventions
        >>> ie = InferenceEngine(bn)
        >>> # Query the marginals as learned from data
        >>> ie.query()['traffic']
        {'heavy': 0.7142857142857142, 'light': 0.2857142857142857}
        >>> # Query the marginals given observations
        >>> ie.query({'rush_hour': True, 'weather': 'Terrible'})['traffic']
        {'heavy': 1.0, 'light': 0.0}
        >>> # Make an intervention on the `BayesianNetwork`
        >>> ie.do_intervention('rush_hour', False)
        >>> # Query marginals on the intervened `BayesianNetwork`
        >>> ie.query()['traffic']
        {'heavy': 0.5, 'light': 0.5}
        >>> # Reset interventions
        >>> ie.reset_do('rush_hour')
        >>> ie.query()['traffic']
        {'heavy': 0.7142857142857142, 'light': 0.2857142857142857}
    """

    def __init__(self, bn: BayesianNetwork):
        """
        Creates a new ``InferenceEngine`` from an existing ``BayesianNetwork``.

        It is expected that structure and probability distribution has already been learned
        for the ``BayesianNetwork`` that is to be used for inference.
        This Bayesian Network cannot contain any isolated nodes.

        Args:
            bn: Bayesian Network that inference will act on.

        Raises:
            ValueError: if the Bayesian Network contains isolates, or if a variable name is invalid,
                        or if the CPDs have not been learned yet.
        """
        bad_nodes = [node for node in bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]

        if bad_nodes:
            raise ValueError(
                "Variable names must match ^[0-9a-zA-Z_]+$ - please fix the "
                f"following nodes: {bad_nodes}"
            )

        if not bn.cpds:
            raise ValueError(
                "Bayesian Network does not contain any CPDs. You should fit CPDs "
                "before doing inference (see `BayesianNetwork.fit_cpds`)."
            )

        self._cpds = None
        self._detached_cpds = {}
        self._baseline_marginals = None

        self._create_cpds_dict_bn(bn)
        self._generate_domains_bn(bn)
        self._generate_bbn()

    def _single_query(
        self,
        observations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[Hashable, float]]:
        """
        Queries the ``BayesianNetwork`` for marginals given some observations.

        Args:
            observations: observed states of nodes in the Bayesian Network.
                          For instance, query({"node_a": 1, "node_b": 3})
                          If None or {}, the marginals for all nodes in the ``BayesianNetwork`` are returned.

        Returns:
            A dictionary of marginal probabilities of the network.
            For instance, :math:`P(a=1) = 0.3, P(a=2) = 0.7` -> {a: {1: 0.3, 2: 0.7}}
        """
        bbn_results = (
            self._bbn.query(**observations) if observations else self._bbn.query()
        )
        results = {node: {} for node in self._cpds}

        for (node, state), prob in bbn_results.items():
            results[node][state] = prob

        # the detached nodes are set to the baseline marginals based on original CPDs
        for node in self._detached_cpds:
            results[node] = self._baseline_marginals[node]

        return results

    def query(
        self,
        observations: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        parallel: bool = False,
        num_cores: Optional[int] = None,
    ) -> Union[
        Dict[str, Dict[Hashable, float]],
        List[Dict[str, Dict[Hashable, float]]],
    ]:
        """
        Queries the ``BayesianNetwork`` for marginals given one or more observations.

        Args:
            observations: one or more observations of states of nodes in the Bayesian Network.
            parallel: if True, run the query using multiprocessing
            num_cores: only applicable if parallel=True. The number of cores used during multiprocessing.
                       If num_cores is not provided, number of processors will be autodetected and used

        Returns:
            A dictionary or a list of dictionaries of marginal probabilities of the network.

        Raises:
            TypeError: if observations is neither None nor a dictionary nor a list
        """
        if observations is not None and not isinstance(observations, (dict, list)):
            raise TypeError("Expecting observations to be a dict, list or None")

        if isinstance(observations, list):
            if parallel:
                with multiprocessing.Pool(num_cores) as p:
                    result = p.map(self._single_query, observations)
            else:
                result = [self._single_query(obs) for obs in observations]
        else:  # dictionary or None
            result = self._single_query(observations)

        return result

    def _do(self, observation: str, state: Dict[Hashable, float]):
        """
        Makes an intervention on the Bayesian Network.

        Args:
            observation: observation that the intervention is on.
            state: mapping of state -> probability.

        Raises:
            ValueError: if states do not match original states of the node, or probabilities do not sum to 1.
        """
        if sum(state.values()) != 1.0:
            raise ValueError("The cpd for the provided observation must sum to 1")

        if max(state.values()) > 1.0 or min(state.values()) < 0:
            raise ValueError(
                "The cpd for the provided observation must be between 0 and 1"
            )

        if not set(state.keys()) == set(self._cpds_original[observation]):
            expected = set(self._cpds_original[observation])
            found = set(state.keys())
            raise ValueError(
                f"The cpd states do not match expected states: expected {expected}, found {found}"
            )

        self._cpds[observation] = {s: {(): p} for s, p in state.items()}

    def do_intervention(
        self,
        node: str,
        state: Optional[Union[Hashable, Dict[Hashable, float]]] = None,
    ):
        """
        Makes an intervention on the Bayesian Network.

        For instance,
            `do_intervention('X', 'x')` will set :math:`P(X=x)` to 1, and :math:`P(X=y)` to 0
            `do_intervention('X', {'x': 0.2, 'y': 0.8})` will set :math:`P(X=x)` to 0.2, and :math:`P(X=y)` to 0.8

        Args:
            node: the node that the intervention acts upon.
            state: state to update node it.
                - if Hashable: the intervention updates the state to 1, and all other states to 0;
                - if Dict[Hashable, float]: update states to all state -> probabilitiy in the dict.

        Raises:
            ValueError: if performing intervention would create an isolated node.
        """
        if not any(
            node in inspect.getargs(f.__code__)[0][1:]
            for _, f in self._node_functions.items()
        ):
            raise ValueError(
                "Do calculus cannot be applied because it would result in an isolate"
            )

        # initialise baseline marginals if not done previously
        if self._baseline_marginals is None:
            self._baseline_marginals = self._single_query(None)

        if isinstance(state, int):
            state = {s: float(s == state) for s in self._cpds[node]}

        self._do(node, state)

        # check for presence of separate subgraph after do-intervention
        self._remove_disconnected_nodes(node)
        self._generate_bbn()

    def reset_do(self, observation: str):
        """
        Resets any do_interventions that have been applied to the observation.

        Args:
            observation: observation that will be reset.
        """
        self._cpds[observation] = self._cpds_original[observation]

        for node, cpd in self._detached_cpds.items():
            self._cpds[node] = cpd

        self._detached_cpds = {}
        self._generate_bbn()

    def _generate_bbn(self):
        """Re-creates the _bbn."""
        self._node_functions = self._create_node_functions()
        self._bbn = build_bbn(
            list(self._node_functions.values()),
            domains=self._domains,
        )

    def _generate_domains_bn(self, bn: BayesianNetwork):
        """
        Generates domains from Bayesian network

        Args:
            bn: Bayesian network
        """
        self._domains = {
            variable: list(cpd.index.values) for variable, cpd in bn.cpds.items()
        }

    def _create_cpds_dict_bn(self, bn: BayesianNetwork):
        """
        Maps CPDs in the ``BayesianNetwork`` to required format:

        Args:
            bn: Bayesian network

        >>> {"observation":
        >>>     {"state":
        >>>         {(("condition1_observation", "condition1_state"), ("conditionN_observation", "conditionN_state")):
        >>>             "probability"
        >>>     }
        >>> }

        For example, :math:`P( Colour=red | Make=fender, Model=stratocaster) = 0.4`:
        >>> {"colour":
        >>>     {"red":
        >>>         {(("make", "fender"), ("model", "stratocaster")):
        >>>             0.4
        >>>         }
        >>>     }
        >>> }
        """
        lookup = {
            variable: {
                state: {
                    tuple(zip(cpd.columns.names, parent_value)): cpd.loc[state][
                        parent_value
                    ]
                    for parent_value in pd.MultiIndex.from_frame(cpd).names
                }
                for state in cpd.index.values
            }
            for variable, cpd in bn.cpds.items()
        }
        self._cpds = lookup
        self._cpds_original = copy.deepcopy(self._cpds)

    def _create_node_function(self, name: str, args: Tuple[str]):
        """Creates a new function that describes a node in the ``BayesianNetwork``."""

        def template() -> float:
            """Template node function."""
            # use inspection to determine arguments to the function
            # initially there are none present, but caller will add appropriate arguments to the function
            # getargvalues was "inadvertently marked as deprecated in Python 3.5"
            # https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
            arg_spec = inspect.getargvalues(inspect.currentframe())

            return self._cpds[arg_spec.args[0]][  # target name
                arg_spec.locals[arg_spec.args[0]]
            ][  # target state
                tuple((arg, arg_spec.locals[arg]) for arg in arg_spec.args[1:])
            ]  # conditions

        code = template.__code__
        pos_count = (
            [code.co_posonlyargcount] if hasattr(code, "co_posonlyargcount") else []
        )
        template.__code__ = types.CodeType(
            len(args),
            *pos_count,
            code.co_kwonlyargcount,
            len(args),
            code.co_stacksize,
            code.co_flags,
            code.co_code,
            code.co_consts,
            code.co_names,
            args,
            code.co_filename,
            name,
            code.co_firstlineno,
            code.co_lnotab,
            code.co_freevars,
            code.co_cellvars,
        )
        template.__name__ = name
        return template

    def _create_node_functions(self) -> Dict[str, Callable]:
        """
        Creates all functions required to create a ``BayesianNetwork``.

        Returns:
            Dictionary of node functions
        """
        node_functions = {}

        for node, states in self._cpds.items():
            # since we only need condition names, which are consistent across all states,
            # then we can inspect the 0th element
            states_conditions = next(iter(states.values()))

            # take any state, and get its conditions
            state_conditions = next(iter(states_conditions.keys()))
            condition_nodes = [n for n, v in state_conditions]

            node_args = tuple([node] + condition_nodes)  # type: Tuple[str]
            node_function = self._create_node_function(f"f_{node}", node_args)
            node_functions[node] = node_function

        return node_functions

    def _remove_disconnected_nodes(self, var: str):
        """
        Identifies and removes from the _cpds the nodes of the bbn which are
        part of one or more upstream subgraphs that could have been formed
        after a do-intervention.

        Uses the attribute _cpds to determine the parents of each node.
        Leverages networkX `weakly_connected_component` method to identify the
        subgraphs.

        For instance, the network A -> B -> C -> D -> E  would be split into
        two sub networks (A -> B) and (C -> D -> E) if we intervene on
        node C.

        Args:
            var: variable we have intervened on
        """
        # construct graph from CPDs
        g = nx.DiGraph()

        for node, states in self._cpds.items():
            sample_state = next(iter(states.values()))
            parents = next(iter(sample_state.keys()))
            g.add_node(node)  # add nodes as there could be isolates

            for parent, _ in parents:
                g.add_edge(parent, node)

        # remove nodes in subgraphs which do not contain the intervention node
        for sub_graph in nx.weakly_connected_components(g):
            if var not in sub_graph:
                for node in sub_graph:
                    self._detached_cpds[node] = self._cpds[node]
                    self._cpds.pop(node)
