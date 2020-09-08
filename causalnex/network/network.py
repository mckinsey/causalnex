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
This module contains the implementation of ``BayesianNetwork``.

``BayesianNetwork`` is a class that represents a probabilistic, weighted, directed acyclic graph (DAG)
describing causal relationships between variables and their distribution in a factorised way.
"""

import re
from copy import deepcopy
from typing import Dict, Hashable, List, Set, Tuple

import networkx as nx
import pandas as pd
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel

from causalnex.structure import StructureModel


class BayesianNetwork:
    """
    Base class for Bayesian Network (BN), a probabilistic weighted DAG where nodes represent variables,
    edges represent the causal relationships between variables.

    ``BayesianNetwork`` stores nodes with their possible states, edges and
    conditional probability distributions (CPDs) of each node.

    ``BayesianNetwork`` is built on top of the ``StructureModel``, which is an extension of ``networkx.DiGraph``
    (see :func:`causalnex.structure.structuremodel.StructureModel`).

    In order to define the ``BayesianNetwork``, users should provide a relevant ``StructureModel``.
    Once ``BayesianNetwork`` is initialised, no changes to the ``StructureModel`` can be made
    and CPDs can be learned from the data.

    The learned CPDs can be then used for likelihood estimation and predictions.

    Example:
    ::
        >>> # Create a Bayesian Network with a manually defined DAG.
        >>> from causalnex.structure import StructureModel
        >>> from causalnex.network import BayesianNetwork
        >>>
        >>> sm = StructureModel()
        >>> sm.add_edges_from([
        >>>                    ('rush_hour', 'traffic'),
        >>>                    ('weather', 'traffic')
        >>>                    ])
        >>> bn = BayesianNetwork(sm)
        >>> # A created ``BayesianNetwork`` stores nodes and edges defined by the ``StructureModel``
        >>> bn.nodes
        ['rush_hour', 'traffic', 'weather']
        >>>
        >>> bn.edges
        [('rush_hour', 'traffic'), ('weather', 'traffic')]
        >>> # A ``BayesianNetwork`` doesn't store any CPDs yet
        >>> bn.cpds
        >>> {}
        >>>
        >>> # Learn the nodes' states from the data
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        >>>                      'rush_hour': [True, False, False, False, True, False, True],
        >>>                      'weather': ['Terrible', 'Good', 'Bad', 'Good', 'Bad', 'Bad', 'Good'],
        >>>                      'traffic': ['heavy', 'light', 'heavy', 'light', 'heavy', 'heavy', 'heavy']
        >>>                      })
        >>> bn = bn.fit_node_states(data)
        >>> bn.node_states
        {'rush_hour': {False, True}, 'weather': {'Bad', 'Good', 'Terrible'}, 'traffic': {'heavy', 'light'}}
        >>> # Learn the CPDs from the data
        >>> bn = bn.fit_cpds(data)
        >>> # Use the learned CPDs to make predictions on the unseen data
        >>> test_data = pd.DataFrame({
        >>>                           'rush_hour': [False, False, True, True],
        >>>                           'weather': ['Good', 'Bad', 'Good', 'Bad']
        >>>                           })
        >>> bn.predict(test_data, "traffic").to_dict()
        >>> {'traffic_prediction': {0: 'light', 1: 'heavy', 2: 'heavy', 3: 'heavy'}}
        >>> bn.predict_probability(test_data, "traffic").to_dict()
        {'traffic_prediction': {0: 'light', 1: 'heavy', 2: 'heavy', 3: 'heavy'}}
        {'traffic_light': {0: 0.75, 1: 0.25, 2: 0.3333333333333333, 3: 0.3333333333333333},
         'traffic_heavy': {0: 0.25, 1: 0.75, 2: 0.6666666666666666, 3: 0.6666666666666666}}
    """

    def __init__(self, structure: StructureModel):
        """
        Create a ``BayesianNetwork`` with a DAG defined by ``StructureModel``.

        Args:
            structure: a graph representing a causal relationship between variables.
                       In the structure
                           - cycles are not allowed;
                           - multiple (parallel) edges are not allowed;
                           - isolated nodes and multiple components are not allowed.

        Raises:
            ValueError: If the structure is not a connected DAG.
        """
        n_components = nx.number_weakly_connected_components(structure)

        if n_components > 1:
            raise ValueError(
                "The given structure has {n_components} separated graph components. "
                "Please make sure it has only one.".format(n_components=n_components)
            )

        if not nx.is_directed_acyclic_graph(structure):
            cycle = nx.find_cycle(structure)
            raise ValueError(
                "The given structure is not acyclic. Please review the following cycle: {cycle}".format(
                    cycle=cycle
                )
            )

        # _node_states is a Dict in the form `dict: {node: dict: {state: index}}`.
        # Underlying libraries expect all states to be integers from zero, and
        # thus this dict is used to convert from state -> idx, and then back from idx -> state as required
        self._node_states = None  # type: Dict[str: Dict[Hashable, int]]
        self._structure = structure

        # _model is a pgmpy Bayesian Model.
        # It is used for:
        #                - probability fitting
        #                - predictions
        self._model = BayesianModel()
        self._model.add_edges_from(structure.edges)

    @property
    def structure(self) -> StructureModel:
        """
        ``StructureModel`` defining the DAG of the Bayesian Network.

        Returns:
            A ``StructureModel`` of the Bayesian Network.
        """
        return self._structure

    @property
    def nodes(self) -> List[str]:
        """
        List of all nodes contained within the Bayesian Network.

        Returns:
            A list of node names.
        """
        return list(self._model.nodes)

    @property
    def node_states(self) -> Dict[str, Set[Hashable]]:
        """
        Dictionary of all states that each node can take.

        Returns:
            A dictionary of node and its possible states, in format of `dict: {node: state}`.
        """
        return {node: set(states.keys()) for node, states in self._node_states.items()}

    @node_states.setter
    def node_states(self, nodes: Dict[str, Set[Hashable]]):
        """
        Set the list of nodes that are contained within the Bayesian Network.
        The states of all nodes must be provided.

        Args:
            nodes: A dictionary of node and its possible states, in format of `dict: {node: state}`.

        Raises:
            ValueError: if a node contains a None state.
            KeyError: if a node is missing.
        """
        missing_feature = set(self.nodes).difference(set(nodes.keys()))
        if missing_feature:
            raise KeyError(
                "The data does not cover all the features found in the Bayesian Network. "
                "Please check the following features: {nodes}".format(
                    nodes=missing_feature
                )
            )

        for node, states in nodes.items():
            if any(pd.isnull(list(states))):
                raise ValueError("node '{node}' contains None state".format(node=node))
        self._node_states = {
            n: {v: k for k, v in enumerate(sorted(nodes[n]))} for n in nodes
        }

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """
        List of all edges contained within the Bayesian Network, as a Tuple(from_node, to_node).

        Returns:
            A list of all edges.
        """
        return list(self._model.edges)

    @property
    def cpds(self) -> Dict[str, pd.DataFrame]:
        """
        Conditional Probability Distributions of each node within the Bayesian Network.

        The row-index of each dataframe is all possible states for the node.
        The col-index of each dataframe is a MultiIndex that describes all possible permutations of parent states.

        For example, for a node :math:`P(A | B, D)`, where
        .. math::
            - A \\in \\text{{"a", "b", "c", "d"}}
            - B \\in \\text{{"x", "y", "z"}}
            - C \\in \\text{{False, True}}

        >>> b         x                   y               z
        >>> d     False     True      False True      False     True
        >>> a
        >>> a  0.265306  0.214286  0.066667  0.25  0.444444  0.000000
        >>> b  0.183673  0.214286  0.200000  0.25  0.222222  0.666667
        >>> c  0.285714  0.285714  0.400000  0.25  0.333333  0.333333
        >>> d  0.265306  0.285714  0.333333  0.25  0.000000  0.000000

        Returns:
            Conditional Probability Distributions of each node within the Bayesian Network.
        """
        cpds = dict()
        for cpd in self._model.cpds:

            iterables = [
                sorted(self._node_states[var].keys()) for var in cpd.variables[1:]
            ]
            cols = [""]
            if iterables:
                cols = pd.MultiIndex.from_product(iterables, names=cpd.variables[1:])

            cpds[cpd.variable] = pd.DataFrame(
                cpd.values.reshape(
                    len(self._node_states[cpd.variable]), max(1, len(cols))
                )
            )
            cpds[cpd.variable][cpd.variable] = sorted(
                self._node_states[cpd.variable].keys()
            )
            cpds[cpd.variable].set_index([cpd.variable], inplace=True)
            cpds[cpd.variable].columns = cols

        return cpds

    def fit_node_states(self, df: pd.DataFrame) -> "BayesianNetwork":
        """
        Fit all states of nodes that can appear in the data.
        The dataframe provided should contain every possible state (values that can be taken) for every column.

        Args:
            df: data to fit node states from. Each column indicates a node and each row
                an observed combination of states.

        Returns:
            self

        Raises:
            ValueError: if dataframe contains any missing data.
        """
        self.node_states = {c: set(df[c].unique()) for c in df.columns}

        return self

    def _state_to_index(
        self, df: pd.DataFrame, nodes: List[str] = None
    ) -> pd.DataFrame:
        """
        Transforms all values in df to an integer, as defined by the mapping from fit_node_states.

        Args:
            df: data to transform
            nodes: list of nodes to map to index. None means all.

        Returns:
            The transformed dataframe.

        Raises:
            ValueError: if nodes have not been fit, or if column names do not match node names.
        """

        df.is_copy = False
        cols = nodes if nodes else df.columns
        for col in cols:
            df[col] = df[col].map(self._node_states[col])
        df.is_copy = True
        return df

    def fit_cpds(
        self,
        data: pd.DataFrame,
        method: str = "MaximumLikelihoodEstimator",
        bayes_prior: str = None,
        equivalent_sample_size: int = None,
    ) -> "BayesianNetwork":
        """
        Learn conditional probability distributions for all nodes in the Bayesian Network, conditioned on
        their incoming edges (parents).

        Args:
            data: dataframe containing one column per node in the Bayesian Network.
            method: how to fit probabilities. One of:
                    - "MaximumLikelihoodEstimator": fit probabilities using Maximum Likelihood Estimation;
                    - "BayesianEstimator": fit probabilities using Bayesian Parameter Estimation. Use bayes_prior.
            bayes_prior: how to construct the Bayesian prior used by method="BayesianEstimator". One of:
                         - "K2": shorthand for dirichlet where all pseudo_counts are 1
                                 regardless of variable cardinality;
                         - "BDeu": equivalent of using Dirichlet and using uniform 'pseudo_counts' of
                                   `equivalent_sample_size / (node_cardinality * np.prod(parents_cardinalities))`
                                   for each node. Use equivelant_sample_size.
            equivalent_sample_size: used by BDeu bayes_prior to compute pseudo_counts.

        Returns:
            self

        Raises:
            ValueError: if an invalid method or bayes_prior is specified.

        """

        state_names = {k: list(v.values()) for k, v in self._node_states.items()}

        transformed_data = data.copy(deep=True)  # type: pd.DataFrame
        transformed_data = self._state_to_index(transformed_data[self.nodes])

        if method == "MaximumLikelihoodEstimator":
            self._model.fit(
                data=transformed_data,
                estimator=MaximumLikelihoodEstimator,
                state_names=state_names,
            )

        elif method == "BayesianEstimator":
            valid_bayes_priors = ["BDeu", "K2"]
            if bayes_prior not in valid_bayes_priors:
                raise ValueError(
                    "unrecognised bayes_prior, please use on of %s"
                    % " ".join(valid_bayes_priors)
                )

            self._model.fit(
                data=transformed_data,
                estimator=BayesianEstimator,
                prior_type=bayes_prior,
                equivalent_sample_size=equivalent_sample_size,
                state_names=state_names,
            )
        else:
            valid_methods = ["MaximumLikelihoodEstimator", "BayesianEstimator"]
            raise ValueError(
                "unrecognised method, please use on of %s" % " ".join(valid_methods)
            )

        return self

    def fit_node_states_and_cpds(
        self,
        data: pd.DataFrame,
        method: str = "MaximumLikelihoodEstimator",
        bayes_prior: str = None,
        equivalent_sample_size: int = None,
    ) -> "BayesianNetwork":
        """
        Call `fit_node_states` and then `fit_cpds`.

        Args:
            data: dataframe containing one column per node in the Bayesian Network.
            method: how to fit probabilities. One of:
                    - "MaximumLikelihoodEstimator": fit probabilities using Maximum Likelihood Estimation;
                    - "BayesianEstimator": fit probabilities using Bayesian Parameter Estimation. Use bayes_prior.
            bayes_prior: how to construct the Bayesian prior used by method="BayesianEstimator". One of:
                         - "K2": shorthand for dirichlet where all pseudo_counts are 1
                                 regardless of variable cardinality;
                         - "BDeu": equivalent of using dirichlet and using uniform 'pseudo_counts' of
                                   `equivalent_sample_size / (node_cardinality * np.prod(parents_cardinalities))`
                                   for each node. Use equivelant_sample_size.
            equivalent_sample_size: used by BDeu bayes_prior to compute pseudo_counts.

        Returns:
            self
        """

        return self.fit_node_states(data).fit_cpds(
            data, method, bayes_prior, equivalent_sample_size
        )

    def predict(self, data: pd.DataFrame, node: str) -> pd.DataFrame:
        """
        Predict the state of a node based on some input data, using the Bayesian Network.

        Args:
            data: data to make prediction.
            node: the node to predict.

        Returns:
            A dataframe of predictions, containing a single column name {node}_prediction.
        """

        if all(parent in data.columns for parent in self._model.get_parents(node)):
            return self._predict_from_complete_data(data, node)

        return self._predict_from_incomplete_data(data, node)

    def _predict_from_complete_data(
        self, data: pd.DataFrame, node: str
    ) -> pd.DataFrame:
        """
        Predicts state of node given all parents of node exist within data.
        This method inspects the CPD of node directly, since all parent states are known.
        This avoids traversing the full network to compute marginals.
        This method is fast.

        Args:
            data: data to make prediction.
            node: the node to predict.

        Returns:
            A dataframe of predictions, containing a single column named {node}_prediction.
        """
        transformed_data = data.copy(deep=True)  # type: pd.DataFrame

        parents = sorted(self._model.get_parents(node))
        cpd = self.cpds[node]

        transformed_data[
            "{node}_prediction".format(node=node)
        ] = transformed_data.apply(
            lambda row: cpd[tuple([row[parent] for parent in parents])].idxmax()
            if parents
            else cpd[""].idxmax(),
            axis=1,
        )
        return transformed_data[[node + "_prediction"]]

    def _predict_from_incomplete_data(
        self, data: pd.DataFrame, node: str
    ) -> pd.DataFrame:
        """
        Predicts state of node when some parents of node do not exist within data.
        This method uses the pgmpy predict function, which predicts the most likely state for every node
        that is not contained within data.
        With incomplete data, pgmpy goes beyond parents in the network to determine the most likely predictions.
        This method is slow.

        Args:
            data: data to make prediction.
            node: the node to predict.

        Returns:
            A dataframe of predictions, containing a single column name {node}_prediction.
        """

        transformed_data = deepcopy(data)  # type: pd.DataFrame
        self._state_to_index(transformed_data)

        # transformed_data.is_copy()

        # pgmpy will predict all missing data, so drop column we want to predict
        transformed_data = transformed_data.drop(columns=[node])

        predictions = self._model.predict(transformed_data)[[node]]

        return predictions.rename(columns={node: node + "_prediction"})

    def predict_probability(self, data: pd.DataFrame, node: str) -> pd.DataFrame:
        """
        Predict the probability of each possible state of a node, based on some input data.

        Args:
            data: data to make prediction.
            node: the node to predict probabilities.

        Returns:
            A dataframe of predicted probabilities, contained one column per possible state, named {node}_{state}.
        """

        if all(parent in data.columns for parent in self._model.get_parents(node)):
            return self._predict_probability_from_complete_data(data, node)

        return self._predict_probability_from_incomplete_data(data, node)

    def _predict_probability_from_complete_data(
        self, data: pd.DataFrame, node: str
    ) -> pd.DataFrame:
        """
        Predict the probability of each possible state of a node, based on some input data.
        This method inspects the CPD of node directly, since all parent states are known.
        This avoids traversing the full network to compute marginals.
        This method is fast.

        Args:
            data: data to make prediction.
            node: the node to predict probabilities.

        Returns:
            A dataframe of predicted probabilities, contained one column per possible state, named {node}_{state}.
        """
        transformed_data = data.copy(deep=True)  # type: pd.DataFrame

        parents = sorted(self._model.get_parents(node))
        cpd = self.cpds[node]

        def lookup_probability(row, s):
            """Retrieve probability from CPD"""
            if parents:
                return cpd[tuple([row[parent] for parent in parents])].loc[s]
            return cpd.at[s, ""]

        for state in self.node_states[node]:
            transformed_data[
                "{n}_{s}".format(n=node, s=state)
            ] = transformed_data.apply(
                lambda row, st=state: lookup_probability(row, st), axis=1
            )

        return transformed_data[
            ["{n}_{s}".format(n=node, s=state) for state in self.node_states[node]]
        ]

    def _predict_probability_from_incomplete_data(
        self, data: pd.DataFrame, node: str
    ) -> pd.DataFrame:
        """
        Predict the probability of each possible state of a node, based on some input data.
        This method uses the pgmpy predict_probability function, which predicts the probability
        of every state for every node that is not contained within data.
        With incomplete data, pgmpy goes beyond parents in the network to determine the most likely predictions.
        This method is slow.

        Args:
            data: data to make prediction.
            node: the node to predict probabilities.

        Returns:
            A dataframe of predicted probabilities, contained one column per possible state, named {node}_{state}.
        """
        transformed_data = data.copy(deep=True)  # type: pd.DataFrame
        self._state_to_index(transformed_data)

        # pgmpy will predict all missing data, so drop column we want to predict
        transformed_data = transformed_data.drop(columns=[node])

        probability = self._model.predict_probability(
            transformed_data
        )  # type: pd.DataFrame

        # keep only probabilities for the node we are interested in
        cols = []
        pattern = re.compile("^{node}_[0-9]+$".format(node=node))
        # disabled open pylint issue (https://github.com/PyCQA/pylint/issues/2962)
        for col in probability.columns:
            if pattern.match(col):
                cols.append(col)
        probability = probability[cols]
        probability.columns = cols

        return probability
