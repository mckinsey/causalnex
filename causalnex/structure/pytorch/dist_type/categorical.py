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
``causalnex.pytorch.data_type.categorical`` defines the categorical distribution type.
"""
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch import nn

from causalnex.structure.pytorch.dist_type._base import DistTypeBase, ExpandColumnsMixin
from causalnex.structure.structuremodel import StructureModel


class DistTypeCategorical(ExpandColumnsMixin, DistTypeBase):
    """ Class defining categorical distribution type functionality """

    # index group of categorical columns
    idx_group = None
    # column expander for later preprocessing
    encoder = None

    def get_columns(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Gets the column(s) associated with the instantiated DistType.

        Args:
            X: Full dataset to be selected from.

        Returns:
            1d or 2d np.ndarray of columns.
        """
        return X[:, self.idx_group]

    def preprocess_X(self, X: np.ndarray, fit_transform: bool = True) -> np.ndarray:
        """
        Expands the feature dimension for each categorical column by:
        - One hot encode each of the categorical features
        - For each feature, get handle on groups of one-hot expanded columns
        - Store the handle groups
        - Return expanded array
        NOTE: the number of expanded columns is EQUAL to the number of classes
        for ease of use with the Pytorch loss functions.
        This is technically wasteful computationally (only need C-1 columns).

        Args:
            X: The original passed-in data.

            fit_transform: Whether the class first fits
            then transforms the data, or just transforms.
            Just transforming is used to preprocess new data after the
            initial NOTEARS fit.

        Returns:
            Preprocessed X
        """
        # deepcopy to prevent overwrite errors
        X = deepcopy(X)

        # fit the OneHotEncoder
        if fit_transform:
            self.encoder = OneHotEncoder(sparse=False, categories="auto", drop=None)
            self.encoder.fit(X[:, [self.idx]])

        # expand columns for this feature
        expanded_columns = self.encoder.transform(X[:, [self.idx]])
        # update the original column with the first expanded column
        X[:, self.idx] = expanded_columns[:, 0]
        # append the remainder cols to X
        X = self._expand_columns(X, expanded_columns[:, 1:])

        # update the idx_group with expanded columns
        if fit_transform:
            self.idx_group = []
            # preserve the first column location
            self.idx_group.append(self.idx)
            # the new cols are appended to the end of X contiguously
            n_new_cols = expanded_columns.shape[1] - 1
            idx_start = X.shape[1] - n_new_cols
            # preserve location of expanded columns
            self.idx_group += list(range(idx_start, X.shape[1]))

        return X

    def preprocess_tabu_edges(
        self, tabu_edges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Update tabu_edges taking into account expanded columns.

        Args:
            tabu_edges: The original tabu_edges.

        Returns:
            Preprocessed tabu_edges.
        """
        return self.update_tabu_edges(
            idx_group=self.idx_group, tabu_edges=tabu_edges, tabu_idx_group=True
        )

    def preprocess_tabu_nodes(self, tabu_nodes: List[int]) -> List[int]:
        """
        Update tabu_nodes taking into account expanded columns.

        Args:
            tabu_nodes: The original tabu_nodes.

        Returns:
            Preprocessed tabu_nodes.
        """
        return self.update_tabu_nodes(idx_group=self.idx_group, tabu_nodes=tabu_nodes)

    def modify_h(self, square_weight_mat: torch.Tensor) -> torch.Tensor:
        """
        Used to prevent spurious cycles between expanded columns and other features.
        For example, A_cat1 -> B -> A_cat2 would not be penalized by the h(W) constraint.

        This modification solves that by adding the expanded columns of the
        squared adjacency matrix onto the original column. This effectively superimposes
        All expanded column connections onto a single connection

        Args:
            square_weight_mat: The squared adjacency matrix used in h(W).

        Returns:
            The modified W matrix.
        """
        orig_idx = self.idx_group[0]
        expand_idx = self.idx_group[1:]

        # Add on the edges from expanded nodes.
        square_weight_mat[orig_idx, :] = square_weight_mat[orig_idx, :] + torch.sum(
            square_weight_mat[expand_idx, :], dim=0
        )
        # Add on the edges to expanded nodes.
        square_weight_mat[:, orig_idx] = square_weight_mat[:, orig_idx] + torch.sum(
            square_weight_mat[:, expand_idx], dim=1
        )

        return square_weight_mat

    def collapse_adj(self, adj: np.ndarray) -> np.ndarray:
        """
        Collapse categories into one column through summation.
        Conceptually the same as modify_h.

        Args:
            adj: The adjacency matrix.

        Returns:
            Updated adjacency matrix.
        """
        orig_idx = self.idx_group[0]
        expand_idx = self.idx_group[1:]

        # Add on the edges from expanded nodes.
        adj[orig_idx, :] = adj[orig_idx, :] + np.sum(adj[expand_idx, :], axis=0)
        # Add on the edges to expanded nodes.
        adj[:, orig_idx] = adj[:, orig_idx] + np.sum(adj[:, expand_idx], axis=1)

        return adj

    @staticmethod
    def _to_index(X_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Recover the numerical columns by argmaxing a one-hot vector.

        Args:
            X_one_hot: The one-hot tensor to be collapsed.

        Returns:
            A 1d tensor representing the classes defined by the above one-hot
            tensor.
        """
        return torch.argmax(X_one_hot, dim=1)

    def add_to_node(self, sm: StructureModel) -> StructureModel:
        """
        Adds self to a node of a structure model corresponding to
        all indexes in self.idx_group.

        Args:
            sm: The input StructureModel

        Returns:
            Updated StructureModel
        """
        for idx in self.idx_group:
            sm.nodes[idx]["dist_type"] = self
        return sm

    def loss(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Uses the functional implementation of the CrossEntropyLoss class
        https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss.

        Returns the mean row wise cross entropy loss for a single group of categorical columns.

        NOTE: the pytorch implementation assumes a numeric target input.
        Therefore, collapse the one hot columns into a numeric target column.

        Args:
            X: The original data passed into NOTEARS (i.e. the reconstruction target).

            X_hat: The reconstructed data.

        Returns:
            Scalar pytorch tensor of the reconstruction loss between X and X_hat.
        """

        return nn.functional.cross_entropy(
            input=X_hat[:, self.idx_group],
            target=self._to_index(X[:, self.idx_group]),
            reduction="mean",
        )

    def inverse_link_function(self, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Softmax inverse link function for categorical data.

        Args:
            X_hat: Reconstructed data in the latent space.

        Returns:
            Modified X_hat.
            MUST be same shape as passed in data.
            Projects the self.idx column from the latent space to the dist_type space.
        """
        X_hat[:, self.idx_group] = torch.softmax(X_hat[:, self.idx_group], dim=1)
        return X_hat

    @staticmethod
    def make_node_name(colname: str, catidx: int) -> str:
        """
        Renaming scheme for expanded categorical columns.
        NOTE: column is not renamed if catidx is 0.
        This is bc original column name needs to stay constant.

        Args:
            colname: The base column used in the renaming.

            catidx: The index of the categorical expansion.

        Returns:
            Updated column name.
        """
        if catidx:
            return f"{colname}{catidx}"
        return colname

    def update_idx_col(self, idx_col: Dict[int, str]) -> Dict[int, str]:
        """
        Expand the named columns to include category names.

        Args:
            idx_col: The original index to column mapping.

        Returns:
            Updated index to column mapping.
        """
        new_idx_cols = {}
        colname = idx_col.pop(self.idx_group[0])
        for catidx, idx in enumerate(self.idx_group):
            new_idx_cols[idx] = self.make_node_name(colname, catidx)
        return {**idx_col, **new_idx_cols}
