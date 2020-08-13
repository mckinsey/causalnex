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
Module of methods to sample variables of a single data type.
"""
import networkx as nx
import numpy as np
import pandas as pd

from causalnex.structure.data_generators import sem_generator


def generate_continuous_data(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "gaussian",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate samples from SEM with specified type of noise.
    The order of the columns on the returned array is the one provided by `sm.nodes`

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'gaussian'/'normal' (alias), 'student-t',
            'exponential', 'gumbel'.
        noise_scale: The standard deviation of the noise.
        intercept: Whether to use an intercept for each feature.
        seed: Random state
    Returns:
        x_mat: [n_samples,d_nodes] sample matrix
    Raises:
        ValueError: if distribution isn't gaussian/normal/student-t/exponential/gumbel
    """
    df = sem_generator(
        graph=sm,
        default_type="continuous",
        n_samples=n_samples,
        distributions={"continuous": distribution},
        noise_std=noise_scale,
        intercept=intercept,
        seed=seed,
    )
    return df[list(sm.nodes())].values


def generate_binary_data(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "logit",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate samples from SEM with specified type of noise.
    The order of the columns on the returned array is the one provided by `sm.nodes`

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'probit'/'normal' (alias),
            'logit' (default).
        noise_scale: The standard deviation of the noise. The binary and
            categorical features are created using a latent variable approach.
            The noise standard deviation determines how much weight the "mean"
            estimate has on the feature value.
        intercept: Whether to use an intercept for the latent variable of each feature.
        seed: Random state
    Returns:
        x_mat: [n_samples,d_nodes] sample matrix
    Raises:
        ValueError: if distribution isn't 'probit', 'normal', 'logit'
    """
    df = sem_generator(
        graph=sm,
        default_type="binary",
        n_samples=n_samples,
        distributions={"binary": distribution},
        noise_std=noise_scale,
        intercept=intercept,
        seed=seed,
    )
    return df[list(sm.nodes())].values


def generate_continuous_dataframe(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "gaussian",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.
    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'gaussian'/'normal' (alias), 'student-t',
            'exponential', 'gumbel'.
        noise_scale: The standard deviation of the noise.
        intercept: Whether to use an intercept for each feature.
        seed: Random state
    Returns:
        Dataframe with the node names as column names
    Raises:
        ValueError: if distribution is not 'gaussian', 'normal', 'student-t',
            'exponential', 'gumbel'
    """
    return sem_generator(
        graph=sm,
        default_type="continuous",
        n_samples=n_samples,
        distributions={"continuous": distribution},
        noise_std=noise_scale,
        intercept=intercept,
        seed=seed,
    )


def generate_binary_dataframe(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "logit",
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'probit'/'normal' (alias),
            'logit' (default).
        noise_scale: The standard deviation of the noise. The binary and
            categorical features are created using a latent variable approach.
            The noise standard deviation determines how much weight the "mean"
            estimate has on the feature value.
        intercept: Whether to use an intercept for the latent variable of each feature.
        seed: Random state
    Returns:
        x_mat: [n_samples,d_nodes] sample matrix
    Raises:
        ValueError: if distribution is not 'probit', 'normal', 'logit'
    """
    return sem_generator(
        graph=sm,
        default_type="binary",
        n_samples=n_samples,
        distributions={"binary": distribution},
        noise_std=noise_scale,
        intercept=intercept,
        seed=seed,
    )


def generate_count_dataframe(
    sm: nx.DiGraph,
    n_samples: int,
    zero_inflation_factor: float = 0.1,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        zero_inflation_factor: The probability of zero inflation for count data.
        seed: Random state
    Returns:
        x_mat: [n_samples, d_nodes] sample matrix
    Raises:
        ValueError: if ``zero_inflation_factor`` is not a float in [0, 1].
    """
    return sem_generator(
        graph=sm,
        default_type="count",
        n_samples=n_samples,
        distributions={"count": zero_inflation_factor},
        noise_std=1,  # not used for poisson
        intercept=True,
        seed=seed,
    )


def generate_categorical_dataframe(
    sm: nx.DiGraph,
    n_samples: int,
    distribution: str = "logit",
    n_categories: int = 3,
    noise_scale: float = 1.0,
    intercept: bool = False,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generates a dataframe with samples from SEM with specified type of noise.

    Args:
        sm: A DAG in form of a networkx or StructureModel. Does not require weights.
        n_samples: The number of rows/observations to sample.
        distribution: The type of distribution to use for the noise
            of a variable. Options: 'probit'/'normal' (alias),
            "logit"/"gumbel" (alias). Logit is default.
        n_categories: Number of categories per variable/node.
        noise_scale: The standard deviation of the noise. The categorical features
            are created using a latent variable approach. The noise standard
            deviation determines how much weight the "mean" estimate has on
            the feature value.
        intercept: Whether to use an intercept for the latent variable of each feature.
        seed: Random state
    Returns:
        x_mat: [n_samples, d_nodes] sample matrix
    Raises:
        ValueError: if distribution is not 'probit', 'normal', 'logit', 'gumbel'
    """
    return sem_generator(
        graph=sm,
        default_type="categorical:{}".format(n_categories),
        n_samples=n_samples,
        distributions={"categorical": distribution},
        noise_std=noise_scale,
        intercept=intercept,
        seed=seed,
    )
