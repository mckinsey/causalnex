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
This module contains the implementation of ``EMSingleLatentVariable``.

``EMSingleLatentVariable`` is a class that implements expectation-maximisation (EM) algorithm
for a single latent variable
"""
import logging
import os
from time import time
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from causalnex.structure import StructureModel
from causalnex.utils.data_utils import chunk_data, count_unique_rows
from causalnex.utils.pgmpy_utils import cpd_multiplication

INITIAL_PARAMS = ["random"]


class EMSingleLatentVariable:  # pylint: disable=too-many-arguments, too-many-instance-attributes
    """
    This class uses Expectation-Maximization to learn parameters of a single latent variable in a bayesian network.
    We do so by also allowing the user to CONSTRAINT the optimisation
    These are elements that help the algorithm find a local optimal point closer to the point we think

    The setting is:
    Input:
        - a StructureModel representing the whole network or any sub-graph containing the Markov Blanket of the LV
        - data as a dataframe. The LV must be in the dataframe, with missing values represented by `np.nan`s
        - constraints:
            - Box - A hard constraint; forbids the solution to be outside of certain boundaries
            - Priors - establishes Dirichlet priors to every parameter
    run:
        - using the method `run` or manually alternating over E and M steps)
    Result:
        - CPTs involving the latent variable, learnt by EM, found in the attribute `cpds`
        - CPTs not involving the LV not learned (They must be learned separately by MLE.
        This is faster and the result is the same)

    Example:
    >>> em = EMSingleLatentVariable(sm=sm, data=data, lv_name=lv_name, node_states=node_states)
    >>> em.run() # run EM until convergence
    >>> # or run E and M steps separately
    >>> for i in range(10): # Run EM 10 times
    >>>     em.e_step()
    >>>     em.m_step()
    """

    def __init__(
        self,
        sm: StructureModel,
        data: pd.DataFrame,
        lv_name: str,
        node_states: Dict[str, list],
        initial_params: Union[str, Dict[str, pd.DataFrame]] = "random",
        seed: int = 22,
        box_constraints: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = None,
        priors: Dict[str, pd.DataFrame] = None,
        non_missing_data_factor: int = 1,
        n_jobs: int = 1,
    ):
        """
        Args:
            sm: structure. Only requirement is: must contain all edges in the Markov Blanket of the latent variable.
                Note: all variable names must be non empty strings
            data: dataframe, must contain all variables in the Markov Blanket of the latent variable. Include one column
                with the latent variable name, filled with np.nan for missing info about LV.
                If some data is present about the LV, create complete columns.
            lv_name: name of latent variable
            node_states: dictionary mapping variable name and list of states
            initial_params: way to initialise parameters. Can be:
                - "random": random values (default)
                - if a dictionary of dataframes is provided, this will be used as the initialisation
            seed: seed for the random generator (used if iitialise parameters randomly)
            box_constraints: minimum and maximum values for each model parameter. Specified with a dictionary mapping:
                - Node
                - two dataframes, in order: Min(P(Node|Par(Node))) and Max(P(Node|Par(Node)))
            priors: priors, provided as a mapping Node -> dataframe with Dirichilet priors for P(Node|Par(Node))
            non_missing_data_factor:
                This is a weight added to the non-missing data samples. The effect is as if the amount of data provided
                was bigger. Empirically, it helps to set the factor to 10 if the non missing data is ~1% of the dataset
            n_jobs:
                If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful
                for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
                Thus for n_jobs = -2, all CPUs but one are used.
        """
        np.random.seed(seed)

        self.sm = sm
        self.lv_name = lv_name
        self.node_states = node_states
        self.initial_params = initial_params
        self.seed = seed
        self.non_missing_data_factor = non_missing_data_factor
        self.n_jobs = max(1, n_jobs if n_jobs >= 0 else (os.cpu_count() + 1 + n_jobs))

        self.box_constraints = self._check_box_constraints(box_constraints)
        self.priors = self._check_priors(priors)

        # These are the nodes for which we compute CPDs. We do not care about CPDs of parents of the latent variable or
        # parents of children of the LV. This is because these CPDs do not depend on the LV, and are not affected by
        # the fact that we do not observe it
        # **IMPORTANT**: The first name in the list MUST BE lv_name (for the multiplication to work correctly)
        self.valid_nodes = [self.lv_name] + list(self.sm.successors(self.lv_name))

        # Initialise CPDs
        self.cpds = self._initialise_network_cpds()
        self._old_cpds = None
        self._mb_partitions = None
        self._sufficient_stats = {}
        self._index_columns_lookup = {}

        # Compute aggregated data based on Markov blanket
        self._mb_data, self._mb_partitions = self._get_markov_blanket_data(data)

        # Build index columns lookup for each valid node
        self._lv_states = self.node_states[self.lv_name]
        self._mb_product = cpd_multiplication(
            [self.cpds[node] for node in self.valid_nodes]
        )
        for node in self.valid_nodes:
            self._mb_partitions[node]["_lookup_"] = self._mb_partitions[node].apply(
                lambda record: self._build_lookup(
                    node, record  # pylint: disable=cell-var-from-loop
                ),
                axis=1,
            )

    @property
    def _logger(self):
        """Obtains logger for this specific class"""
        return logging.getLogger(self.__class__.__name__)

    def run(self, n_runs: int, stopping_delta: float = 0.0, verbose: int = 0):
        """
        Runs E and M steps until convergence (`stopping_delta`) or max iterations is reached (n_runs)

        Args:
            n_runs: max number of EM alternations
            stopping_delta: if max difference in current - last iteration CPDS < stopping_delta => convergence reached
            verbose: amount of printing
        """
        if verbose:
            self._logger.info(
                "* Iteration 0: likelihood = %.4f",
                self.compute_total_likelihood(),
            )

        for i in range(n_runs):
            t_start = time()
            self.e_step()  # Expectation step
            e_duration = time() - t_start

            t_start = time()
            self.m_step()  # Maximisation step
            m_duration = time() - t_start

            self.apply_box_constraints()  # Apply box constraints
            delta = self._stopping_criteria()  # Compute change in parameters

            if verbose:
                self._logger.info(
                    "* Iteration %d: "
                    "likelihood = %.4f | "
                    "max(|theta - theta_old|)) = %.4f | "
                    "duration = %.4fs (E-step), %.4fs (M-step)",
                    (i + 1),
                    self.compute_total_likelihood(),
                    delta,
                    e_duration,
                    m_duration,
                )

            if delta < stopping_delta:
                break

    def e_step(self) -> Dict[str, pd.DataFrame]:
        """
        Performs the Expectation step.
        This boils down to computing the expected sufficient statistics M[X, U]
        for every "valid" node X, where U = Par(X)

        Returns:
            The expected sufficient statistics of each node X
        """
        # This is a product of elements in the Markov Blanket of the latent variable.
        # NOTE: Convert product dataframe to dictionary to speed up the E-step
        self._mb_product = cpd_multiplication(
            [self.cpds[node] for node in self.valid_nodes],
            normalize=True,
        ).to_dict(orient="dict")

        # Get M[X, U] for X being each valid node and U being its parents (Daphne Koller's notation)
        for node in self.valid_nodes:
            node_mb_data = self._mb_partitions[node]

            # Initialize ESS with zeros (or prior values) and then increase from data
            sufficient_stats_df = self._initialize_sufficient_stats(node)
            sufficient_stats = sufficient_stats_df.to_dict(orient="dict")

            # Update ESS based on all data records (observations)
            if self.n_jobs == 1:
                results = self._update_sufficient_stats(node_mb_data["_lookup_"])

                for updates in results:
                    for idx, cols, val in updates:
                        sufficient_stats[cols][idx] += val
            else:
                results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(self._update_sufficient_stats)(chunk_df["_lookup_"])
                    for chunk_df in chunk_data(node_mb_data, self.n_jobs * 2)
                )
                for chunk_results in results:
                    for updates in chunk_results:
                        for idx, cols, val in updates:
                            sufficient_stats[cols][idx] += val

            # Register sufficient statistics as Pandas dataframe
            self._sufficient_stats[node] = pd.DataFrame(
                sufficient_stats,
                index=sufficient_stats_df.index,
                columns=sufficient_stats_df.columns,
            )

        return self._sufficient_stats

    def m_step(self) -> Dict[str, pd.DataFrame]:
        """
        Maximization step. It boils down to normalising the likelihood table previously created

        $$ \\theta_{[X | U]} = M[X, U] / M[U] = M[X, U] / \\sum_X M[X, U] $$

        Returns:
            New updated CPDs
        """
        self._old_cpds = self.cpds  # Store old CPDs
        self.cpds = {
            node: self._normalise(self._sufficient_stats[node])
            for node in self.valid_nodes
        }
        return self.cpds

    def compute_total_likelihood(self) -> float:
        """
        This computes the LOG likelihood of the whole dataset (or MAP, if priors given)
        for the current parameter steps

        Returns:
            Total likelihood over dataset
        """
        cpd_prods = cpd_multiplication(
            [self.cpds[n] for n in self.valid_nodes],
            normalize=False,
        )
        proba_of_row = cpd_prods.sum(axis=0)

        def compute_likelihood_stub(record: Dict) -> float:
            t = tuple(record[el] for el in proba_of_row.index.names)

            if np.isnan(record[self.lv_name]):
                likelihood = proba_of_row.loc[t]
            else:
                likelihood = cpd_prods.loc[record[self.lv_name], t]

            return np.log(likelihood) * record["count"]

        return self._mb_data.apply(compute_likelihood_stub, axis=1).sum()

    def apply_box_constraints(self):
        """
        if CPDs fall outside the box constraints created, bring them back to inside the constraints.
        """
        if self.box_constraints is None:
            return

        for node in self.valid_nodes:
            min_vals, max_vals = self.box_constraints[node]
            cpd = self.cpds[node]

            # where replaces if the condition is false
            cpd.where(cpd < max_vals, max_vals, inplace=True)
            cpd.where(cpd > min_vals, min_vals, inplace=True)
            self.cpds[node] = self._normalise(cpd)

    @staticmethod
    def get_default_priors(
        sm: StructureModel,
        node_states: Dict[str, list],
        lv_name: str,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        The default dirichlet priors (zero values)

        Args:
            sm: model structure
            node_states: node states
            lv_name: name of latent variable

        Returns:
            Dictionary with pd dataframes initialized with zeros
        """
        valid_node_set = set([lv_name] + list(sm.successors(lv_name)))
        return {
            node: EMSingleLatentVariable._initialise_node_cpd(node, node_states, sm)
            for node in sm.nodes
            if node in valid_node_set
        }

    @staticmethod
    def get_default_box(
        sm: StructureModel,
        node_states: Dict[str, list],
        lv_name: str,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Get boxes with min = 0 and max = 1 for all parameters.

        Args:
            sm: model structure
            node_states: node states
            lv_name: name of latent variable

        Returns:
            Dictionary with a tuple of two elements: the first being the lower value constraint and the second
        the maximum value constraint
        """
        valid_node_set = set([lv_name] + list(sm.successors(lv_name)))
        boxes = {}

        for node in sm.nodes:
            if node in valid_node_set:
                cpd = EMSingleLatentVariable._initialise_node_cpd(node, node_states, sm)
                min_vals, max_vals = cpd.copy(), cpd.copy()
                min_vals.loc[:] = 0
                max_vals.loc[:] = 1
                boxes[node] = (min_vals, max_vals)

        return boxes

    def _build_lookup(
        self,
        node: str,
        record: Dict,
    ) -> Tuple[List[Tuple], Tuple]:
        """
        Build lookup table based on an individual data record/instance

        Args:
            node: Node name
            record: A data record/instance

        Returns:
            List of CPD index-columns-count triplets, and tuple of Markov blanket columns
        """
        count = record["count"]
        node_value = record[node]
        node_cpd = self.cpds[node]
        lv_states = self.node_states[self.lv_name]
        idx_cols_counts = []

        if not np.isnan(record[self.lv_name]):
            mb_cols = None

            if node_cpd.shape[1] == 1:
                # if the probability is unconditional (i.e. P(Z)), the column names are [""]
                idx_cols_counts.append((node_value, "", count))
            else:
                cols = tuple(record[j] for j in node_cpd.columns.names)
                idx_cols_counts.append((node_value, cols, count))
        else:
            mb_cols = tuple(record[j] for j in self._mb_product.columns.names)

            if node_cpd.shape[1] == 1:
                # if the probability is unconditional (i.e., P(Z)), the column names are [""]
                for lv_value in lv_states:
                    index = lv_value if node == self.lv_name else node_value
                    idx_cols_counts.append((index, "", count))
            else:
                if node == self.lv_name:
                    cols = tuple(record[j] for j in node_cpd.columns.names)

                    for lv_value in lv_states:
                        idx_cols_counts.append((lv_value, cols, count))
                else:
                    for lv_value in lv_states:
                        cols = tuple(
                            lv_value if j == self.lv_name else record[j]
                            for j in node_cpd.columns.names
                        )
                        idx_cols_counts.append((node_value, cols, count))

        return idx_cols_counts, mb_cols

    def _update_sufficient_stats(
        self,
        lookup: pd.Series,
    ) -> List[List[Tuple]]:
        """
        Update expected sufficient statistics based on a given dataframe

        Args:
            lookup: Lookup table for index, columns and count

        Returns:
            List of list of update tuples
        """
        updates = []

        for idx_cols_counts, mb_cols in lookup.values:
            if mb_cols is None:
                # Update the ESS: increase it by the number of times this row appears on the dataset
                updates.append(idx_cols_counts)
            else:
                # Update the ESS: increase it by:
                #     (number of times this row appears on the dataset) * (Probability of Z assuming that value)
                # Because lv is not observed, we consider all possible values it can assume and,
                # instead of adding 1 for the likelihood M[X=x_i, U=u_i], we add p(Z=z_1|observations)
                prob_lv_given_mb = self._mb_product[mb_cols]
                updates.append(
                    [
                        (idx, cols, count * prob_lv_given_mb[lv_value])
                        for (idx, cols, count), lv_value in zip(
                            idx_cols_counts, self._lv_states
                        )
                    ]
                )

        return updates

    def _initialize_sufficient_stats(self, node: str) -> pd.DataFrame:
        """
        Likelihood of node and parents, initialized with zeros (or prior values) and then increased from data.
        The likelihood is not a conditional expression (i.e. X|U), but a joint expression (i.e. X,U).
        However, we use the same structure as the CPT to store that likelihood. That structure is:
        a pandas table with the index being X values and the the columns being U values as a MultiIndex

        Args:
            node: Node key

        Returns:
            Dataframe containing the likelihood of the node's parents
        """
        if self.priors is None:
            sufficient_stats_df = self.cpds[node].copy()
            sufficient_stats_df.loc[:] = 0
        else:
            sufficient_stats_df = self.priors[node].copy()

        return sufficient_stats_df

    def _initialise_network_cpds(self) -> Dict[str, pd.DataFrame]:
        """
        Initialise all the CPDs according to the choice made in the constructor.
        It can be:
        - filling CPDs with random values,
        - filling CPDs with specific values given by user,
        - filling CPDs with uniform probabilities (Tends to have bad effects on convergence)

        Returns:
            Dictionary of CPD dataframes

        Raises:
            ValueError: if `initial_params` is neither a dictionary nor part of supported type strings
        """
        if isinstance(self.initial_params, str) and self.initial_params in set(
            INITIAL_PARAMS
        ):
            valid_node_set = set(self.valid_nodes)
            cpds = {}

            for node in self.sm.nodes:
                if node in valid_node_set:
                    cpd = self._initialise_node_cpd(node, self.node_states, self.sm)
                    cpd.loc[:] = 1

                    if self.initial_params == "random":
                        cpd.loc[:] = np.random.random(cpd.shape)

                    cpd = cpd / cpd.sum(axis=0)
                    cpds[node] = cpd  # Update dictionary
        elif isinstance(self.initial_params, dict):
            self._check_initial_params_dict()
            cpds = self.initial_params
        else:
            raise ValueError(
                f"`initial_params` must be a dictionary or one of {INITIAL_PARAMS}"
            )

        return cpds

    @staticmethod
    def _initialise_node_cpd(
        node: str,
        node_states: Dict[str, List],
        sm: StructureModel,
    ) -> pd.DataFrame:
        """
        Initialise the CPD of a specified node

        Args:
            node: Node name
            node_states: States of the node
            sm: Structure model

        Returns:
            CPD dataframe associated with the node

        Raises:
            ValueError: if node is not found in the network
        """
        parents = list(sorted(sm.predecessors(node)))
        columns = [""]

        if len(parents) > 0:
            columns = pd.MultiIndex.from_product(
                [sorted(node_states[p]) for p in parents],
                names=parents,
            )

        indices = pd.Index(data=sorted(node_states[node]), name=node)
        values = np.zeros(shape=(len(indices), len(columns)))
        return pd.DataFrame(index=indices, columns=columns, data=values)

    def _get_markov_blanket_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Keeps only features the belong to the latent variable's Markov blanket +
        groups and counts identical rows +
        multiplies non missing data counts by a factor

        Args:
            df: Raw data

        Returns:
            Aggregated data, as well as data partition for each node
        """
        # Add a column for latent variable if not already present
        if self.lv_name not in df.columns:
            df[self.lv_name] = np.nan

        # Get the counts of each unique record
        valid_cols = set()

        for node in self.valid_nodes:
            valid_cols.add(node)
            valid_cols.update(self.sm.predecessors(node))

        mb_data = count_unique_rows(df[list(valid_cols)])
        indices = ~mb_data.isna().any(axis=1)
        mb_data.loc[indices, "count"] *= self.non_missing_data_factor

        # Partition data based on the Markov blanket of each node
        mb_product = cpd_multiplication([self.cpds[node] for node in self.valid_nodes])
        mb_partitions = {}

        for node in self.valid_nodes:
            valid_cols = list(
                set(
                    [node, "count"]
                    + list(self.sm.predecessors(node))
                    + mb_product.columns.names
                )
            )
            mb_partitions[node] = count_unique_rows(mb_data[valid_cols])

        return mb_data, mb_partitions

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalises dataframe

        Args:
            df: Raw dataframe

        Returns:
            Normalised dataframe
        """
        norm_df = df / df.sum(axis=0)
        norm_df.fillna(1.0 / df.shape[0], inplace=True)
        return norm_df

    def _stopping_criteria(self) -> float:
        """
        Maximum change, in absolute values, between parameters of last EM iteration and params of current EM iteration

        Returns:
            Maximum absolute difference between new and old CPDs
        """
        return max(
            (
                (self._old_cpds[node] - self.cpds[node]).abs().values.max()
                for node in self.valid_nodes
            ),
            default=-1,  # return -1 if valid nodes list is empty
        )

    def _check_initial_params_dict(self):
        """
        Checks initial parameter dictionary

        Raises:
            ValueError: when the initial parameter dictionary keys are different from valid nodes, or
                when the CPD provided in the initial parameter dictionary has incorrect format
        """
        if sorted(self.valid_nodes) != sorted(self.initial_params.keys()):
            raise ValueError(
                "If `initial_params` is a dictionary, it has to map `valid nodes` to "
                "corresponding CPTs. A valid node is : L.V. or Successors(L.V.)"
            )
        for node in self.valid_nodes:
            df = self.initial_params[node]
            check = (
                isinstance(df, pd.DataFrame)
                and (df.index.name == node)
                and (list(df.index) == self.node_states[node])
            )
            check = check and (
                (np.all(df.columns == "") and (not self.sm.predecessors(node)))
                or (df.columns.names == list(self.sm.predecessors(node)))
            )
            if not check:  # pragma: no cover
                raise ValueError(
                    "CPTs provided in `initial_params` do not correspond to the expected format"
                )

    @staticmethod
    def _check_priors(priors: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Checks if the priors are passed in the right format and if they are valid

        Args:
            priors: Prior distribution to check

        Returns:
            Verified priors

        Raises:
            ValueError: when prior distributions are invalid
        """
        if priors is None or isinstance(priors, dict):
            return priors

        raise ValueError(f"Invalid priors {priors}")

    @staticmethod
    def _check_box_constraints(
        box_constraints: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Checks if the box constraints are passed in the right format and if they are valid

        Args:
            box_constraints: Box constraints to check

        Returns:
            Verified box constraints

        Raises:
            ValueError: when box constraints are invalid
        """
        if box_constraints is None or isinstance(box_constraints, dict):
            return box_constraints

        raise ValueError(f"Invalid box constraints {box_constraints}")
