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

from typing import Dict

import numpy as np
import pandas as pd
import pytest
from pgmpy.models import BayesianModel

from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas


@pytest.fixture
def train_model() -> StructureModel:
    """
    This Bayesian Model structure will be used in all tests, and all fixtures will adhere to this structure.

    Cause-only nodes: [d, e]
    Effect-only nodes: [a, c]
    Cause / Effect nodes: [b]

            d
         ↙  ↓  ↘
        a ← b → c
            ↑  ↗
            e
    """
    model = StructureModel()
    model.add_edges_from(
        [
            ("b", "a"),
            ("b", "c"),
            ("d", "a"),
            ("d", "c"),
            ("d", "b"),
            ("e", "c"),
            ("e", "b"),
        ]
    )
    return model


@pytest.fixture
def train_model_idx(train_model) -> BayesianModel:
    """
    This Bayesian model is identical to the train_model() fixture, with the exception that node names
    are integers from zero to 1, mapped by:

    {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    """
    model = BayesianModel()
    idx_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    model.add_edges_from([(idx_map[u], idx_map[v]) for u, v in train_model.edges])
    return model


@pytest.fixture
def train_data() -> pd.DataFrame:
    """
    Training data for testing Bayesian Networks. There are 98 samples, with 5 columns:

    - a: {"a", "b", "c", "d"}
    - b: {"x", "y", "z"}
    - c: 0.0 - 100.0
    - d: Boolean
    - e: Boolean

    This data was generated by constructing the Bayesian Model train_model(), and then sampling
    from this structure. Since e and d are both independent of all other nodes, these were sampled first for
    each row (form their respective pre-defined distributions). This then allows the sampling of all further
    variables based on their conditional dependencies.

    The approximate distributions used to sample from can be viewed by inspecting train_data_cpds().

    """

    data_arr = [
        ["a", "x", 73.78658346945414, False, False],
        ["d", "x", 12.765853213346603, False, False],
        ["c", "y", 22.43657132589221, False, False],
        ["a", "x", 4.267744937038964, False, False],
        ["b", "x", 62.87087344904927, False, False],
        ["c", "x", 31.55295196889971, False, False],
        ["a", "x", 37.403388911083965, False, False],
        ["b", "x", 63.171968604247155, False, False],
        ["d", "x", 11.140539452118263, False, False],
        ["d", "x", 0.1555338799942385, True, False],
        ["c", "x", 9.269926225399187, False, True],
        ["b", "z", 75.38846241765208, True, True],
        ["c", "z", 33.10212378889936, False, True],
        ["b", "z", 57.04657630213301, True, True],
        ["b", "x", 72.03855905511072, True, False],
        ["c", "x", 5.106018765399956, False, False],
        ["c", "z", 5.802617702038839, False, True],
        ["c", "x", 17.22538330530506, False, False],
        ["a", "y", 87.05395007052729, False, False],
        ["d", "y", 19.09989481093348, False, False],
        ["c", "x", 4.313272835124353, True, False],
        ["b", "x", 13.660704178900938, True, True],
        ["b", "x", 7.693287813764131, False, False],
        ["c", "y", 32.791770073523246, False, False],
        ["c", "y", 12.039098492465282, False, False],
        ["a", "x", 51.97718339128754, False, False],
        ["d", "x", 8.393970656769238, False, False],
        ["a", "x", 0.3610815726384886, False, False],
        ["a", "y", 35.31788713900731, True, False],
        ["b", "x", 35.84702992379284, False, True],
        ["c", "y", 32.872350426703356, True, False],
        ["a", "x", 21.218746335586868, False, True],
        ["b", "y", 71.5495653029006, True, False],
        ["c", "x", 15.393846082097575, False, False],
        ["d", "y", 4.514559208625406, False, False],
        ["d", "x", 0.704928173400301, False, False],
        ["c", "y", 34.10829794112354, True, False],
        ["d", "x", 6.84602512195673, False, False],
        ["b", "y", 25.43743439885204, False, False],
        ["d", "x", 7.544831467091971, False, False],
        ["d", "x", 13.923699372025073, False, False],
        ["b", "x", 21.493005760070915, False, False],
        ["a", "x", 41.353977640369436, False, False],
        ["c", "z", 10.015835005248583, True, True],
        ["c", "z", 29.40115954319444, False, True],
        ["c", "x", 17.305145945035388, False, False],
        ["b", "x", 57.3687951851441, False, False],
        ["a", "x", 59.31395756039643, False, False],
        ["d", "x", 19.557939187075984, False, False],
        ["d", "y", 15.739556224725082, False, False],
        ["c", "x", 6.850626809845993, True, False],
        ["c", "x", 7.774579861173826, False, False],
        ["c", "x", 20.807136344297092, True, False],
        ["b", "y", 29.406207780312343, False, False],
        ["a", "x", 34.38851648220974, False, False],
        ["d", "x", 1.0951104244381218, True, False],
        ["c", "x", 37.27483338042188, False, False],
        ["b", "x", 15.745994603442064, False, False],
        ["c", "x", 17.78180189764816, False, True],
        ["a", "x", 17.067548428231493, True, False],
        ["c", "x", 26.857320012899727, False, False],
        ["a", "x", 41.0038510689549, False, True],
        ["d", "x", 0.2299684913699096, False, True],
        ["a", "x", 57.35885570158893, True, False],
        ["d", "x", 12.40118443712448, False, False],
        ["c", "x", 22.624550487374112, False, False],
        ["a", "x", 93.08587619178269, False, False],
        ["b", "y", 18.33030505634329, False, False],
        ["a", "z", 64.29945681859853, False, True],
        ["b", "x", 73.66024742961967, False, False],
        ["b", "x", 16.717397443478287, False, True],
        ["c", "y", 4.642615342125205, False, True],
        ["c", "x", 9.431345661106931, False, False],
        ["c", "y", 31.76238774237109, False, False],
        ["c", "y", 3.6961806894707965, False, False],
        ["d", "y", 2.298895066631253, True, False],
        ["d", "y", 13.222298172220462, False, False],
        ["c", "x", 28.301638775451153, False, False],
        ["d", "x", 7.702270580869413, True, False],
        ["a", "y", 41.38492280508702, True, False],
        ["d", "x", 13.047815503255656, True, False],
        ["c", "x", 22.14641490202623, False, False],
        ["b", "z", 43.13007970158368, False, True],
        ["b", "x", 60.09518672623882, True, False],
        ["a", "x", 79.6370082234198, False, False],
        ["d", "x", 16.60880504367762, False, False],
        ["a", "z", 22.88783470451029, False, True],
        ["a", "x", 33.66416643964188, False, False],
        ["b", "y", 69.91787304290465, True, True],
        ["c", "x", 31.941092922567663, True, False],
        ["d", "x", 16.739638908154518, False, False],
        ["a", "z", 11.129589373273108, False, True],
        ["d", "y", 4.96943558614434, True, False],
        ["d", "y", 6.585354730457387, False, False],
        ["d", "x", 9.859942318446954, False, False],
        ["b", "z", 18.541485302271496, False, True],
        ["a", "x", 87.53473074574995, True, False],
        ["a", "z", 59.61068083691302, False, True],
    ]

    data = pd.DataFrame(data_arr, columns=["a", "b", "c", "d", "e"])
    return data


@pytest.fixture
def train_data_discrete(train_data) -> pd.DataFrame:
    """
    train_data in discretised form. This maps "c" into 5 buckets:
    - 0: x < 20
    - 1: 20 <= x < 40
    - 2: 40 <= x < 60
    - 3: 60 <= x < 80
    - 4: 80 <= x
    """
    df = train_data.copy(deep=True)  # type: pd.DataFrame
    df["c"] = df["c"].apply(
        lambda c: 0 if c < 20 else 1 if c < 40 else 2 if c < 60 else 3 if c < 80 else 4
    )
    return df


@pytest.fixture
def train_data_idx(train_data) -> pd.DataFrame:
    """
    train_data in integer index form. This maps each column into values from 0..n
    """

    df = train_data.copy(deep=True)  # type: pd.DataFrame

    df["a"] = df["a"].map({"a": 0, "b": 1, "c": 2, "d": 3})
    df["b"] = df["b"].map({"x": 0, "y": 1, "z": 2})
    df["c"] = df["c"].apply(
        lambda c: 0 if c < 20 else 1 if c < 40 else 2 if c < 60 else 3 if c < 80 else 4
    )
    df["d"] = df["d"].map({True: 1, False: 0})
    df["e"] = df["e"].map({True: 1, False: 0})
    return df


@pytest.fixture
def train_data_idx_cpds(train_data_idx) -> Dict[str, np.ndarray]:
    """Conditional probability distributions of train_data in the train_model"""

    return create_cpds(train_data_idx)


@pytest.fixture
def train_data_discrete_cpds(train_data_discrete) -> Dict[str, np.ndarray]:
    """Conditional probability distributions of train_data in the train_model"""

    return create_cpds(train_data_discrete)


@pytest.fixture
def train_data_discrete_cpds_k2(train_data_discrete) -> Dict[str, np.ndarray]:
    """Conditional probability distributions of train_data in the train_model"""

    return create_cpds(train_data_discrete, pc=1)


def create_cpds(data, pc=0):

    df = data.copy(deep=True)  # type: pd.DataFrame

    df_vals = {col: list(df[col].unique()) for col in df.columns}
    for _, vals in df_vals.items():
        vals.sort()

    cpd_a = np.array(
        [
            [
                (len(df[(df["a"] == a) & (df["b"] == b) & (df["d"] == d)]) + pc)
                / (len(df[(df["b"] == b) & (df["d"] == d)]) + (pc * len(df_vals["a"])))
                for b in df_vals["b"]
                for d in df_vals["d"]
            ]
            for a in df_vals["a"]
        ]
    )

    cpd_b = np.array(
        [
            [
                (len(df[(df["b"] == b) & (df["d"] == d) & (df["e"] == e)]) + pc)
                / (len(df[(df["d"] == d) & (df["e"] == e)]) + (pc * len(df_vals["b"])))
                for d in df_vals["d"]
                for e in df_vals["e"]
            ]
            for b in df_vals["b"]
        ]
    )

    cpd_c = np.array(
        [
            [
                (
                    (
                        len(
                            df[
                                (df["c"] == c)
                                & (df["b"] == b)
                                & (df["d"] == d)
                                & (df["e"] == e)
                            ]
                        )
                        + pc
                    )
                    / (
                        len(df[(df["b"] == b) & (df["d"] == d) & (df["e"] == e)])
                        + (pc * len(df_vals["c"]))
                    )
                )
                if not df[(df["b"] == b) & (df["d"] == d) & (df["e"] == e)].empty
                else (1 / len(df_vals["c"]))
                for b in df_vals["b"]
                for d in df_vals["d"]
                for e in df_vals["e"]
            ]
            for c in df_vals["c"]
        ]
    )

    cpd_d = np.array(
        [
            [(len(df[df["d"] == d]) + pc) / (len(df) + (pc * len(df_vals["d"])))]
            for d in df_vals["d"]
        ]
    )

    cpd_e = np.array(
        [
            [(len(df[df["e"] == e]) + pc) / (len(df) + (pc * len(df_vals["e"])))]
            for e in df_vals["e"]
        ]
    )

    return {"a": cpd_a, "b": cpd_b, "c": cpd_c, "d": cpd_d, "e": cpd_e}


@pytest.fixture
def train_data_idx_marginals(train_data_idx_cpds):

    return create_marginals(
        train_data_idx_cpds,
        {
            "a": list(range(4)),
            "b": list(range(3)),
            "c": list(range(5)),
            "d": list(range(2)),
            "e": list(range(2)),
        },
    )


@pytest.fixture
def train_data_discrete_marginals(train_data_discrete_cpds):

    return create_marginals(
        train_data_discrete_cpds,
        {
            "a": ["a", "b", "c", "d"],
            "b": ["x", "y", "z"],
            "c": [0, 1, 2, 3, 4],
            "d": [False, True],
            "e": [False, True],
        },
    )


def create_marginals(cpds, data_vals):
    cpd_d = cpds["d"]
    p_d = {i: cpd_d[i, 0] for i in range(len(cpd_d))}

    cpd_e = cpds["e"]
    p_e = {i: cpd_e[i, 0] for i in range(len(cpd_e))}

    cpd_b = cpds["b"]
    c_b = np.array(
        [
            [p_d[d] * p_e[e] for d in range(len(cpd_d)) for e in range(len(cpd_e))]
            for _ in range(len(cpd_b))
        ]
    )
    p_b = dict(enumerate((c_b * cpd_b).sum(axis=1)))

    cpd_a = cpds["a"]
    c_a = np.array(
        [
            [p_b[b] * p_d[d] for b in range(len(cpd_b)) for d in range(len(cpd_d))]
            for _ in range(len(cpd_a))
        ]
    )
    p_a = dict(enumerate((c_a * cpd_a).sum(axis=1)))

    cpd_c = cpds["c"]
    c_c = np.array(
        [
            [
                p_b[b] * p_d[d] * p_e[e]
                for b in range(len(cpd_b))
                for d in range(len(cpd_d))
                for e in range(len(cpd_e))
            ]
            for _ in range(len(cpd_c))
        ]
    )
    p_c = dict(enumerate((c_c * cpd_c).sum(axis=1)))

    marginals = {
        "a": {data_vals["a"][k]: v for k, v in p_a.items()},
        "b": {data_vals["b"][k]: v for k, v in p_b.items()},
        "c": {data_vals["c"][k]: v for k, v in p_c.items()},
        "d": {data_vals["d"][k]: v for k, v in p_d.items()},
        "e": {data_vals["e"][k]: v for k, v in p_e.items()},
    }

    return marginals


@pytest.fixture
def test_data_c() -> pd.DataFrame:
    """Test data created so that C should be perfectly predicted based on train_data_cpds.

    Given the two independent variables are set randomly (d, e), all other variables are set to be
    from the category with maximum likelihood in train_data_cpds"""

    data_arr = [
        ["a", "x", 1, False, False],
        ["b", "x", 2, False, True],
        ["c", "x", 3, True, False],
        ["d", "x", 4, True, True],
        ["d", "y", 1, False, False],
        ["c", "y", 2, False, True],
        ["b", "y", 23, True, False],
        ["a", "y", 64, True, True],
        ["c", "z", 1, False, False],
        ["a", "z", 2, False, True],
        ["d", "z", 3, True, False],
        ["b", "z", 0, True, True],
    ]

    data = pd.DataFrame(data_arr, columns=["a", "b", "c", "d", "e"])
    return data


@pytest.fixture
def test_data_c_discrete(test_data_c) -> pd.DataFrame:
    """Test data C that has been discretised (see train_data_discrete)"""
    df = test_data_c.copy(deep=True)  # type: pd.DataFrame
    df["c"] = df["c"].apply(
        lambda c: 0 if c < 20 else 1 if c < 40 else 2 if c < 60 else 3 if c < 80 else 4
    )
    return df


@pytest.fixture
def test_data_c_likelihood(train_data_discrete_cpds) -> pd.DataFrame:
    """Marginal likelihoods for train_data in train_model"""

    # Known bug in pylint with generated Dict: https://github.com/PyCQA/pylint/issues/1498
    data_arr = [
        [
            (train_data_discrete_cpds["c"])[  # pylint: disable=unsubscriptable-object
                y, x
            ]
            for y in range(
                len(
                    # pylint: disable=unsubscriptable-object
                    train_data_discrete_cpds["c"]
                )
            )
        ]
        for x in range(len(train_data_discrete_cpds["c"][0]))
    ]

    likelihood = pd.DataFrame(data_arr, columns=["c_0", "c_1", "c_2", "c_3", "c_4"])
    return likelihood


@pytest.fixture
def bn(train_data_idx, train_data_discrete) -> BayesianNetwork:
    return BayesianNetwork(
        from_pandas(train_data_idx, w_threshold=0.3)
    ).fit_node_states_and_cpds(train_data_discrete)


## For Dynotears
@pytest.fixture
def train_model_temporal_intra() -> np.ndarray:
    """
    The generated intra weights with absolute value between 0.5 and 2.0
    """

    w_mat = np.array(
        [
            [0, 0, 0, 1.62663154, 0],
            [0, 0, 0, 0, -0.78228208],
            [0, 0, 0, 0, 0],
            [0, 1.56358252, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    return w_mat


@pytest.fixture
def train_model_temporal_inter() -> np.ndarray:
    """
    The generated inter weights with absolute value between 0.3 and 0.5
    """

    a_mat = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, -0.46142154],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    return a_mat


@pytest.fixture
def train_data_num_temporal() -> np.ndarray:
    """
    Training data for testing Dynamic Bayesian Networks. Return a time series with 99 time points, with 5 columns

    This data was generated by first generating a DAG (p=1) with intra-slice weights between 0.5 and 2.0, inter-slice
    weights between 0.3 and 0.5, and then simulating a linear SEM time series with Gaussian noise (noise scale = 1)
    """

    data_arr = np.array(
        [
            [-0.02321828, -1.61165408, -0.07385147, -0.24422552, 1.69273056],
            [1.69280458, 0.28621477, -0.77642329, 0.87501005, 0.61999326],
            [-0.16043124, 0.34238485, -2.7924182, 0.46106484, 0.24569186],
            [1.1126993, 4.20246622, 0.58408731, 2.36717278, -3.79635012],
            [-0.36314016, -2.63301957, 0.51866606, -1.62658743, 0.56624632],
            [-0.21056409, 0.02971924, -0.07326009, 0.10911186, 1.01465964],
            [0.32396894, -2.78944483, -0.57305703, -1.8535305, 2.55984801],
            [1.02646917, 4.35363431, 0.52951275, 2.24509786, -3.81351256],
            [-0.03260197, 0.01618086, 0.80840988, -0.16221164, -2.59469977],
            [-0.49979717, -3.81992733, -0.15390537, -1.98016553, 2.51277675],
            [-0.28870368, -1.7222887, 0.49180716, -0.453967, 3.27867182],
            [0.16200306, 0.49101695, 0.41051899, 0.19822653, 0.0184036],
            [0.53190605, -0.79465447, 3.03875251, 0.81700284, -0.70102849],
            [2.42960056, 8.01107092, -0.5491014, 4.76866227, -6.35841236],
            [-0.66925268, -0.36580951, 0.80012342, -0.89363253, -3.82845897],
            [1.42630846, 3.9797194, 0.07146405, 2.74190622, -3.55042177],
            [-0.31767154, -0.186424, -1.42683257, -0.07531761, -2.18712112],
            [0.04070656, -3.37731403, 0.63918187, -0.92276491, 4.02031746],
            [-0.31175741, 1.38383547, 0.84365538, 0.33897871, -0.52244876],
            [-0.52863905, -0.06394188, 0.35195968, -0.36708446, -0.8470632],
            [0.26631584, 2.83655579, 0.03918621, 0.66667103, -2.62742168],
            [0.14095262, 1.74182487, -1.92117552, 0.39919682, -2.91983942],
            [-0.22422218, -0.79009298, 0.10241767, -0.7680089, 1.06158916],
            [-0.46721752, -1.59336826, -1.93519184, -0.951727, 4.05542546],
            [1.30122696, 3.50468032, -0.2662291, 1.93309163, -2.26114515],
            [0.14660659, 0.30667756, -0.01627902, 0.10526078, -1.67555955],
            [-1.1229281, 1.76263148, -0.84703692, 0.40445294, -0.45612106],
            [0.24699356, 3.87845315, 0.26248821, 2.64465882, -4.98883823],
            [-1.01347068, -0.20505091, 1.00006403, -0.50997584, 0.24985109],
            [1.10935621, 2.5461818, 1.10342559, 1.03535485, -2.02208731],
            [0.03007325, 0.18819891, -0.80852747, -0.09376697, -1.24895992],
            [-1.12225884, -2.74448946, 1.26671293, -1.7551675, 0.84316999],
            [0.21170026, 0.8666726, -1.04734377, 1.81814828, 0.72346733],
            [-0.70450358, -1.55609055, -0.33084965, -1.74287962, 0.90505761],
            [1.02394457, 2.73120155, -0.03621037, 1.69090049, -1.69028755],
            [-0.13974442, 0.75733711, -0.24797786, 1.28761776, -3.01108137],
            [-0.2409733, 0.76716693, -2.74105903, 0.40532884, 1.06970522],
            [1.10293452, 2.08271, -0.27825493, 0.27247264, -1.70200507],
            [-0.26337273, -3.72973909, 0.17732437, -1.6806267, 1.86769367],
            [0.37119597, -0.73964829, -0.0702115, 0.11171073, 2.1572643],
            [0.63218489, 1.24616162, 1.79892424, 0.22980726, 0.29215982],
            [0.5428499, 1.89199799, 0.00339809, 1.53200717, -2.37173027],
            [-1.09334582, -0.82777007, -0.06386689, -0.50282303, 0.19881761],
            [0.44363416, 1.23614784, 0.19477709, -0.17092936, -1.56935483],
            [-0.58650474, -5.39091455, 0.46799586, -2.11442586, 3.62770361],
            [-0.36294583, -1.60751013, -0.41906402, -1.28829473, 4.58512431],
            [0.94776573, 2.90229038, -1.21914862, 2.21016331, -1.87864851],
            [-0.65933156, -0.4902382, 1.35866222, -0.0730998, 0.27163915],
            [-0.24712434, 3.38430332, -0.05595978, 1.3150185, -2.98010557],
            [-0.66608534, 2.13253515, -0.05084911, 0.62612475, -3.86170227],
            [-0.05770518, -1.27847749, 0.08567979, -2.50006281, -0.28223134],
            [0.10529196, -1.29774197, 0.2685198, 0.43001899, 2.30821964],
            [0.10302488, 1.45524469, -1.1702323, 0.6414378, 0.16732655],
            [0.11903256, 0.29430439, -1.71192971, 0.626839, -1.54512268],
            [-0.23821375, -0.55630394, 2.61837797, -0.47849221, 0.55930197],
            [0.20976274, 0.34783071, -0.42876291, 0.03390824, -1.59706198],
            [-0.65683079, -0.01900069, 0.27255125, -0.13186946, -0.50820485],
            [-0.80469276, -0.72837247, 0.36098953, -1.08021284, 0.84923978],
            [-0.66895374, 0.5964851, 0.67887803, -0.72580682, -0.30069165],
            [-0.12062117, -2.50973175, -0.07038693, -0.74729032, 0.77151419],
            [-0.42822116, 1.74342368, -0.07820894, 1.74263432, 0.43948777],
            [-0.82934079, -4.08337982, -1.34633028, -2.60952279, 6.81386196],
            [-1.12942086, -2.08208169, -1.38461788, -1.72447668, 4.13046221],
            [0.31470921, -3.63304746, -0.18841025, -0.96737182, 3.01261479],
            [0.85845468, 0.99028605, -0.30543819, 0.12456786, 0.66376562],
            [-0.38173283, 0.87666618, 0.59887464, 0.30624424, -0.12721811],
            [-1.5957473, -5.08066796, -0.37793896, -3.04852141, 2.59095326],
            [0.72869316, -1.46664699, -2.44893671, -0.14214266, 3.54432529],
            [1.0496291, -0.63721967, -0.99082022, -0.44143269, 0.63806218],
            [-0.25416386, -2.90734815, -0.71686804, -1.01348253, 3.19536627],
            [-0.3837594, 0.56178084, 0.04279428, -0.77097096, -0.2159793],
            [1.45467485, 1.32980284, -0.76190851, 1.76603647, -3.24490454],
            [-1.32231229, -2.0512392, -0.93245106, -1.31465228, 1.9682632],
            [1.38783083, 3.08884663, 0.08495293, 1.66257903, -0.11187307],
            [-0.87031164, -1.77959468, 1.4123345, -1.34280853, -1.13597349],
            [-1.02751501, -0.99694372, 0.95790259, -0.95160582, 1.45818764],
            [-0.3756856, -1.26008403, 0.40677209, -1.67779896, 1.55954617],
            [0.84772007, 2.91822088, 0.97996129, 1.49348603, -0.6107629],
            [-2.15450891, -4.90844382, 0.35588695, -3.39723275, 3.54174512],
            [-1.53718986, -4.17267638, 0.29020525, -2.76603146, 6.90510596],
            [0.35858626, -0.51498994, 0.61172917, -0.53678041, 2.15743],
            [-0.38294477, -0.66702856, -0.12482242, -0.84600473, 2.44856913],
            [-0.38553232, 0.92513702, 0.20003269, 1.0647153, -0.39992838],
            [-1.52542638, -1.57964503, -0.20538669, -1.01650943, 1.62311498],
            [1.05397622, 2.73006408, 1.17057079, 1.75844071, -1.35250129],
            [-1.51255063, -2.22853485, -0.086907, -0.53200201, 1.66200483],
            [-0.60265811, 0.98793281, 0.38941998, 0.19590575, 0.42309177],
            [0.82523607, 0.79800418, -0.16821019, 0.17837779, -1.16223326],
            [0.7706973, 2.28134737, -0.63057693, 2.19026111, -2.03067281],
            [-1.3388298, -2.20521516, -0.14339517, -2.10403481, 0.29761894],
            [0.59836837, -0.41983473, -0.63690782, 0.22945261, -0.05038398],
            [0.10708815, 1.40161878, 0.01907963, 0.03987306, -1.13027033],
            [0.4224614, 3.19136434, -0.48974602, 2.02692679, -2.59646715],
            [1.39079366, 1.11830633, 1.79376238, 2.22685897, -2.77804626],
            [1.12394621, 6.44697571, -0.28085969, 4.88933844, -5.47909877],
            [-1.03156969, -3.93819242, -0.30093919, -2.97956448, 1.56332015],
            [0.32178524, -1.96323289, -0.29999344, -0.56693131, 4.0389811],
            [0.90089148, 0.14834971, -0.39156163, 0.51059754, 1.33296277],
            [-1.1745885, -3.5349767, 0.16542735, -2.70793895, 3.56022575],
            [0.97395602, 3.09310162, -0.41087326, 2.18450333, -0.30632787],
        ]
    )
    return data_arr
