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

from collections import Counter
from typing import Any, AnyStr, Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import pytest

from causalnex.estimator.em import EMSingleLatentVariable
from causalnex.network import BayesianNetwork
from causalnex.structure.structuremodel import StructureModel
from causalnex.utils.data_utils import states_to_df


def naive_bayes_plus_parents(
    categories: int = 3,
    samples: int = 500,
    parents: int = 3,
    children: int = 3,
    p_z: float = 0.9,
    p_c: float = 0.9,
    percentage_not_missing: float = 0,
    seed: int = 22,
) -> Tuple[pd.DataFrame, StructureModel, Dict, np.array]:
    """
    p0 ... pn
     \\  |  /
        z
     /  |  \\
    c0 ... cm

    z = mode of parents with probability p_z, otherwise mode of parents + 1 mod n_categories
    c0 = z with prob. p_c, otherwise it is z + 1 mod n_categories
    if no p are give, sample z from the categories uniformly

    Args:
        categories: number of categories
        samples: number of samples
        parents: number of parents, n as shown above
        children: number of children, m as above
        p_z: probability that z = mode(parents)
        p_c: probability that children equals parent
        percentage_not_missing: percentage of the LV that is provided. The default is 0, i.e. the LV is not observed
        seed: seed for random generator

    Returns:
        data: sampled pandas dataframe, missing data on z
        sm: structure model
        node_states: dictionary of list of states for each node
        true_lv_values: true values of latent variable
    """

    def mode(lst: Iterable) -> Any:
        return Counter(lst).most_common()[0][0] if len(lst) > 0 else np.nan

    np.random.seed(seed)
    par_samples = np.random.choice(categories, size=[samples, parents])

    if parents == 0:
        true_lv_values = np.random.choice(categories, size=[samples, 1])
    else:
        true_lv_values = np.array(
            [
                [(mode(el) + np.random.choice(2, p=[p_z, 1 - p_z])) % categories]
                for el in par_samples
            ]
        )

    child_samples = np.random.random(size=[samples, children])
    aux = true_lv_values.repeat(children, axis=1)
    child_samples = np.where(child_samples < p_c, aux, (aux + 1) % categories)

    df = pd.concat(
        [
            pd.DataFrame(par_samples, columns=[f"p_{i}" for i in range(parents)]),
            pd.DataFrame(child_samples, columns=[f"c_{i}" for i in range(children)]),
            pd.DataFrame(true_lv_values, columns=["z"]),
        ],
        axis=1,
    )
    df.loc[int(samples * percentage_not_missing) :, "z"] = np.nan

    sm = StructureModel()
    sm.add_edges_from([(f"p_{i}", "z") for i in range(parents)])
    sm.add_edges_from([("z", f"c_{i}") for i in range(children)])

    node_states = {"z": list(range(categories))}

    for i in range(parents):
        node_states[f"p_{i}"] = list(range(categories))
    for i in range(children):
        node_states[f"c_{i}"] = list(range(categories))

    return df, sm, node_states, true_lv_values


def compare_result_with_ideal(
    em_cpds: Dict[str, pd.DataFrame],
    sm: StructureModel,
    data: pd.DataFrame,
    true_values_lv: np.array,
    node_states: Dict[AnyStr, Union[List, Set]],
) -> Tuple[float, float]:
    """
    Compare learned CPDs with ideal CPDs

    Args:
        em_cpds: Learned CPDs for different nodes
        sm: Structure model
        data: Input dataset
        true_values_lv: Ideal values of the latent variable
        node_states: Possible tates of different nodes

    Returns:
        Maximum absolute difference and root mean square of differences
    """
    data["z"] = true_values_lv.reshape(-1)
    bn = BayesianNetwork(sm)
    bn.fit_node_states(states_to_df(node_states))
    bn.fit_cpds(data)

    max_delta = -1
    avg_delta = 0

    for node in em_cpds:
        deltas = (em_cpds[node] - bn.cpds[node]).abs().values
        max_delta = max(max_delta, deltas.max())
        avg_delta += np.mean(deltas ** 2)

    avg_delta = np.sqrt(avg_delta / len(em_cpds))
    return max_delta, avg_delta


def get_correct_cpds(
    df: pd.DataFrame,
    sm: StructureModel,
    node_states: Dict,
    true_lv_values: np.array,
) -> pd.DataFrame:
    """
    Get the cpds obtained if complete data was provided (no latent variable)

    Args:
        df: Input dataset
        sm: Structure model
        node_states: Dictionary of node states
        true_lv_values: True values of latent variable

    Returns:
        Ground-truth CPDs
    """
    data = df.copy()
    data["z"] = true_lv_values
    bn = BayesianNetwork(sm)
    bn.fit_node_states(states_to_df(node_states))
    bn.fit_cpds(data)
    return bn.cpds


class TestEMJobs:
    @pytest.mark.parametrize("n_jobs", [1, 3, -2])
    def test_em_no_missing_data(self, n_jobs):
        """If all data for the latent variable is provided, the result is the same as runing bn.fit_cpds"""
        df, sm, node_states, true_lv_values = naive_bayes_plus_parents(
            percentage_not_missing=1
        )
        em = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
            n_jobs=n_jobs,
        )
        em.run(n_runs=50, stopping_delta=0.001, verbose=2)

        max_error, _ = compare_result_with_ideal(
            em.cpds, sm, df, true_lv_values, node_states
        )
        assert max_error == 0

    @pytest.mark.parametrize("n_jobs", [1, 3, -2])
    def test_em_missing_data(self, n_jobs):
        """Test EM algorithm given some "missing" data """
        df, sm, node_states, true_lv_values = naive_bayes_plus_parents(
            p_c=0.6,
            p_z=0.6,
            parents=1,
            percentage_not_missing=0.25,
            samples=5000,
            categories=2,
        )
        em = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
            n_jobs=n_jobs,
        )
        em.run(n_runs=50, stopping_delta=0.001, verbose=2)

        max_error, rmse_error = compare_result_with_ideal(
            em.cpds, sm, df, true_lv_values, node_states
        )
        assert max_error < 0.02
        assert rmse_error < 1e-2

    @pytest.mark.parametrize("n_jobs", [1, 3, -2])
    def test_em_no_parents(self, n_jobs):
        """Test EM algorithm on pure naive Bayes structure without parents"""
        df, sm, node_states, true_lv_values = naive_bayes_plus_parents(
            p_c=0.6, parents=0, percentage_not_missing=0.02
        )
        em = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
            n_jobs=n_jobs,
        )
        em.run(n_runs=50, stopping_delta=0.001, verbose=2)

        max_error, rmse_error = compare_result_with_ideal(
            em.cpds, sm, df, true_lv_values, node_states
        )
        assert max_error < 0.02
        assert rmse_error < 1e-2

    @pytest.mark.parametrize("n_jobs", [1, 3, -2])
    def test_em_likelihood_always_go_up(self, n_jobs):
        """Test convergence properties of EM algorithm"""
        df, sm, node_states, _ = naive_bayes_plus_parents(
            parents=2,
            percentage_not_missing=0.1,
            samples=500,
            categories=2,
        )
        em = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
            n_jobs=n_jobs,
        )
        likelihood_old = -np.inf

        for _ in range(50):
            likelihood = em.compute_total_likelihood()
            assert likelihood > likelihood_old
            likelihood_old = likelihood

            em.e_step()
            em.m_step()
            em.apply_box_constraints()
            delta = em._stopping_criteria()  # pylint: disable=protected-access

            if delta < 0.01:
                break


class TestInitialParameters:
    def test_initial_params_provided(self):
        """
        If we provide initial parameters close to the real ones,
        EM should converge to values close to the real ones
        EM will still converge to a solution different to whether we observed all missing values
        """
        df, sm, node_states, true_lv_values = naive_bayes_plus_parents(
            p_c=0.6,
            parents=1,
            children=2,
            percentage_not_missing=0,
            samples=5000,
        )
        correct_cpds = get_correct_cpds(df, sm, node_states, true_lv_values)
        correct_cpds.pop("p_0", None)

        em = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
            initial_params=correct_cpds,
        )
        em.run(n_runs=20, stopping_delta=0.001, verbose=2)

        max_error, rmse_error = compare_result_with_ideal(
            em.cpds, sm, df, true_lv_values, node_states
        )
        assert max_error < 0.01
        assert rmse_error < 4e-3

    @pytest.mark.parametrize("initial_params", [None, [], set(), tuple(), "xyz", 123])
    def test_invalid_initial_params(self, initial_params):
        """An error should be raised if the latent variable is not part of the edges to add"""
        with pytest.raises(
            ValueError,
            match=r"`initial_params` must be a dictionary or one of .*",
        ):
            df, sm, node_states, _ = naive_bayes_plus_parents()
            EMSingleLatentVariable(
                data=df,
                sm=sm,
                node_states=node_states,
                lv_name="z",
                initial_params=initial_params,
            )

    @pytest.mark.parametrize(
        "initial_params", [{}, {"xyd": "abc"}, {"a": 123}, {123: "xyz"}]
    )
    def test_invalid_initial_params_dict(self, initial_params):
        """An error should be raised if the latent variable is not part of the edges to add"""
        with pytest.raises(
            ValueError,
            match=r"If `initial_params` is a dictionary, it has to map `valid nodes` to corresponding CPTs. .*",
        ):
            df, sm, node_states, _ = naive_bayes_plus_parents()
            EMSingleLatentVariable(
                data=df,
                sm=sm,
                node_states=node_states,
                lv_name="z",
                initial_params=initial_params,
            )


class TestPriors:
    def test_get_default_priors(self):
        """Test EM algorithm on naive Bayes structure with additional parents"""
        _, sm, node_states, _ = naive_bayes_plus_parents(
            p_c=0.6, parents=3, percentage_not_missing=0.02
        )
        default_priors = EMSingleLatentVariable.get_default_priors(sm, node_states, "z")
        assert default_priors.keys() == {"c_0", "c_1", "c_2", "z"}

        for k in default_priors:
            assert np.all(default_priors[k] == 0)

    def test_em_with_priors(self):
        """Test some specific priors chosen"""
        df, sm, node_states, true_lv_values = naive_bayes_plus_parents(
            p_c=0.6,
            parents=1,
            children=2,
            percentage_not_missing=0,
        )
        # Setting priors
        priors = EMSingleLatentVariable.get_default_priors(sm, node_states, "z")
        # prior values
        priors["c_0"].loc[:] = [[0.61, 0.0, 0.34], [0.39, 0.6, 0.0], [0.0, 0.4, 0.66]]
        priors["c_1"].loc[:] = [[0.61, 0.0, 0.4], [0.39, 0.6, 0.0], [0.0, 0.4, 0.6]]
        priors["z"].loc[:] = [[0.91, 0, 0.08], [0.09, 0.89, 0], [0, 0.11, 0.92]]

        # prior strengths
        priors["c_1"] = priors["c_1"] * 70
        priors["c_0"] = priors["c_0"] * 70
        priors["z"] = priors["z"] * 70

        em = EMSingleLatentVariable(
            data=df, sm=sm, node_states=node_states, lv_name="z", priors=priors
        )
        em.run(n_runs=20, stopping_delta=0.01)

        max_error, rmse_error = compare_result_with_ideal(
            em.cpds, sm, df, true_lv_values, node_states
        )
        assert max_error < 0.02
        assert rmse_error < 1e-2

    def test_em_with_close_priors(self):
        """If the priors are close to real parameters, the result is very accurate"""
        df, sm, node_states, true_lv_values = naive_bayes_plus_parents(
            p_c=0.6,
            parents=1,
            children=2,
            percentage_not_missing=0,
        )
        correct_cpds = get_correct_cpds(df, sm, node_states, true_lv_values)
        priors = EMSingleLatentVariable.get_default_priors(sm, node_states, "z")
        cte = 200

        # Setting boxes
        for el in ["c_1", "c_0", "z"]:
            priors[el].loc[:] = correct_cpds[el] * cte

        em = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
            priors=priors,
        )
        em.run(n_runs=20, stopping_delta=0.01, verbose=2)

        max_error, rmse_error = compare_result_with_ideal(
            em.cpds, sm, df, true_lv_values, node_states
        )
        assert max_error < 0.01
        assert rmse_error < 4e-3

    def test_default_priors_do_not_affect_result(self):
        """Test EM with default priors that do not affect the end results"""
        df, sm, node_states, _ = naive_bayes_plus_parents(
            p_c=0.6,
            parents=1,
            children=2,
            percentage_not_missing=0,
        )
        priors = EMSingleLatentVariable.get_default_priors(sm, node_states, "z")
        em_box = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
            priors=priors,
        )
        em_box.run(n_runs=20, stopping_delta=0.06)
        em_no_box = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
        )
        em_no_box.run(n_runs=20, stopping_delta=0.06)

        for el in em_box.cpds:
            assert np.all((em_box.cpds[el] - em_no_box.cpds[el]).abs() < 1e-15)

    @pytest.mark.parametrize("priors", [[], set(), tuple(), "xyz", 123, np.array([])])
    def test_invalid_priors(self, priors):
        """Test EM with invalid priors type"""
        with pytest.raises(ValueError, match=r"Invalid priors *"):
            df, sm, node_states, _ = naive_bayes_plus_parents()
            EMSingleLatentVariable(
                data=df, sm=sm, node_states=node_states, lv_name="z", priors=priors
            )


class TestBoxConstraints:
    def test_default_boxes(self):
        """Test EM with default box constraints"""
        _, sm, node_states, _ = naive_bayes_plus_parents(
            p_c=0.6,
            parents=3,
            percentage_not_missing=0.02,
        )
        priors = EMSingleLatentVariable.get_default_box(sm, node_states, "z")
        assert priors.keys() == {"c_0", "c_1", "c_2", "z"}

        for _, prior in priors.items():
            assert np.all(prior[0] == 0)
            assert np.all(prior[1] == 1)

    def test_em_with_close_box_constraints(self):
        """
        Test EM with box constraints that are close to real parameters.
        The result should be very accurate
        """
        df, sm, node_states, true_lv_values = naive_bayes_plus_parents(
            p_c=0.6,
            parents=1,
            children=2,
            percentage_not_missing=0,
            samples=5000,
        )
        correct_cpds = get_correct_cpds(df, sm, node_states, true_lv_values)
        box = EMSingleLatentVariable.get_default_box(sm, node_states, "z")

        # Setting boxes
        for el in ["c_1", "c_0", "z"]:
            box[el][0].loc[:] = correct_cpds[el] - 0.0001  # min
            box[el][1].loc[:] = correct_cpds[el] + 0.0001  # max

        em = EMSingleLatentVariable(
            data=df, sm=sm, node_states=node_states, lv_name="z", box_constraints=box
        )
        em.run(n_runs=20, stopping_delta=0.01, verbose=2)

        max_error, rmse_error = compare_result_with_ideal(
            em.cpds, sm, df, true_lv_values, node_states
        )
        assert max_error < 0.0002
        assert rmse_error < 1e-4

    def test_default_boxes_do_not_affect_result(self):
        """Test EM with box constraints that do not affect the end results"""
        df, sm, node_states, _ = naive_bayes_plus_parents(
            p_c=0.6,
            parents=1,
            children=2,
            percentage_not_missing=0,
        )
        box = EMSingleLatentVariable.get_default_box(sm, node_states, "z")
        em_box = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
            box_constraints=box,
        )
        em_box.run(n_runs=20, stopping_delta=0.06)

        em_no_box = EMSingleLatentVariable(
            data=df,
            sm=sm,
            node_states=node_states,
            lv_name="z",
        )
        em_no_box.run(n_runs=20, stopping_delta=0.06)

        for el in em_box.cpds:
            assert np.all((em_box.cpds[el] - em_no_box.cpds[el]).abs() < 1e-15)

    @pytest.mark.parametrize(
        "box_constraints", [[], set(), tuple(), "xyz", 123, np.array([])]
    )
    def test_invalid_box_constraints(self, box_constraints):
        """Test EM with invalid box constraint type"""
        with pytest.raises(ValueError, match=r"Invalid box constraints *"):
            df, sm, node_states, _ = naive_bayes_plus_parents()
            EMSingleLatentVariable(
                data=df,
                sm=sm,
                node_states=node_states,
                lv_name="z",
                box_constraints=box_constraints,
            )
