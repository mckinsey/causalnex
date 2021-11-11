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
This module contains the helper functions for data manipulation
"""
from typing import AnyStr, Dict, List, Set, Union

import numpy as np
import pandas as pd


def chunk_data(df: pd.DataFrame, n_chunks: int) -> List[pd.DataFrame]:
    """
    Chunk dataframe into multiple smaller dataframes

    Args:
        df: Input dataframe
        n_chunks: Number of chunks

    Yields:
        Dataframe chunk
    """
    k = int(np.ceil(df.shape[0] / n_chunks))

    for i in range(n_chunks):
        yield df.iloc[i * k : (i + 1) * k]


def states_to_df(node_states: Dict[AnyStr, Union[list, Set]]) -> pd.DataFrame:
    """
    Return a dataframe containing all node states for all nodes.
    Used for fitting node states of an BN

    Args:
        node_states: Dictionary of node states

    Returns:
        Dataframe representing all node states
    """
    nodes = node_states.keys()
    max_card = max([len(el) for el in node_states.values()])
    df = pd.DataFrame(np.zeros([max_card, len(nodes)]), columns=nodes)

    for node in nodes:
        values = list(node_states[node])
        df.loc[:, node] = (values * (1 + max_card // len(values)))[:max_card]

    return df


def count_unique_rows(data: pd.DataFrame, placeholder: float = -np.inf) -> pd.DataFrame:
    """
    Take a dataset with repeated rows and returns another dataset,
    with unique rows and the count of rows in the original dataset

    Args:
        data: Input dataframe
        placeholder: Missing value placeholder

    Returns:
        Count dataframe

    Raises:
        ValueError: If the data is empty

    Example:
        a b c
        -----
        0 1 nan
        0 1 0
        1 0 0
        1 0 0
        1 1 1
        1 1 1
        1 1 1

        Is converted to :

        a b c    count
        ---------------
        0 1 nan  1
        0 1 0    1
        1 0 0    2
        1 1 1    3
    """
    # find a placeholder for NaNs: groupby excludes NaNs by default.
    # So we replace them with some value and later put the NaNs back
    data.fillna(placeholder, inplace=True)
    cols = list(data.columns)

    if "count" not in data.columns:
        data[
            "count"
        ] = 1  # Add a dummy count column to ensure that agg_data has non-empty columns

    agg_data = data.groupby(cols).sum().reset_index()
    agg_data.replace(placeholder, np.nan, inplace=True)
    return agg_data
