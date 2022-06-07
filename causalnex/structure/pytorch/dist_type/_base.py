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
``causalnex.pytorch.dist_type._base`` defines the distribution type class interface and default behavior.
"""

import itertools
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch

from causalnex.structure.structuremodel import StructureModel


class DistTypeBase(metaclass=ABCMeta):
    """Base class defining the distribution default behavior and interface"""

    def __init__(self, idx: int):
        """
        Default constructor for the DistTypeBase class.
        Unless overridden, provides default behavior to all subclasses.

        Args:
            idx: Positional index in data passed to the NOTEARS algorithm
            which correspond to this datatype.
        """
        self.idx = idx

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
        return X[:, self.idx]

    # pylint: disable=unused-argument
    def preprocess_X(self, X: np.ndarray, fit_transform: bool = True) -> np.ndarray:
        """
        Overload this method to perform any required preprocessing of the data
        matrix. This can include data conversion, column expansion etc.
        Changes to the tabu parameters should also be done here.

        **WARN** This preprocessing CANNOT reorder the columns of X.

        Args:
            X: The original passed-in data.

            fit_transform: Whether the class first fits
            then transforms the data, or just transforms.
            Just transforming is used to preprocess new data after the
            initial NOTEARS fit.

        Returns:
            Preprocessed X
        """
        return X

    def preprocess_tabu_edges(
        self, tabu_edges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Overload this method to perform any required preprocessing of the tabu_edges.

        Args:
            tabu_edges: The original tabu_edges.

        Returns:
            Preprocessed tabu_edges.
        """
        return tabu_edges

    def preprocess_tabu_nodes(self, tabu_nodes: List[int]) -> List[int]:
        """
        Overload this method to perform any required preprocessing of the tabu_nodes.

        Args:
            tabu_nodes: The original tabu_nodes.

        Returns:
            Preprocessed tabu_nodes.
        """
        return tabu_nodes

    def update_idx_col(self, idx_col: Dict[int, str]) -> Dict[int, str]:
        """
        Overload this method to update the idx_col dict with expanded colnames.

        Args:
            idx_col: The original index to column mapping.

        Returns:
            Updated index to column mapping.
        """
        return idx_col

    def add_to_node(self, sm: StructureModel) -> StructureModel:
        """
        Adds self to a node of a structure model corresponding to self.idx.

        Args:
            sm: The input StructureModel

        Returns:
            Updated StructureModel
        """
        sm.nodes[self.idx]["dist_type"] = self
        return sm

    def modify_h(self, square_weight_mat: torch.Tensor) -> torch.Tensor:
        """
        Overload this method to apply updates to the W matrix in h(W).
        Typically used to prevent spurious cycles when using expended columns.

        Args:
            square_weight_mat: The weight matrix used in h(W).

        Returns:
            Updated weight matrix used in h(W).
        """
        return square_weight_mat

    def collapse_adj(self, adj: np.ndarray) -> np.ndarray:
        """
        Overload this method to apply updates to collapse the W matrix
        of a multi-parameter distribution
        Likely has the same impact as modify_h.

        Args:
            adj: The adjacency matrix.

        Returns:
            Updated adjacency matrix.
        """
        return adj

    @abstractmethod
    def loss(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: The original data passed into NOTEARS (i.e. the reconstruction target).

            X_hat: The reconstructed data.

        Returns:
            Scalar pytorch tensor of the reconstruction loss between X and X_hat.
        """
        raise NotImplementedError("Must implement the loss() method")

    @abstractmethod
    def inverse_link_function(self, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Convert the transformed data from the latent space to the original dtype
        using the inverse link function.

        Args:
            X_hat: Reconstructed data in the latent space.

        Returns:
            Modified X_hat.
            MUST be same shape as passed in data.
            Projects the self.idx column from the latent space to the dist_type space.
        """
        raise NotImplementedError("Must implement the inverse_link_function() method")


class ExpandColumnsMixin:
    """
    Mixin class providing convenience methods for column expansion.
    """

    @staticmethod
    def _expand_columns(X: np.ndarray, new_columns: np.ndarray) -> np.ndarray:
        """
        Expands the data matrix columns without reordering the indices.

        Args:
            X: Base dataset to expand.

            new_columns: The columns to expand the dataset by.

        Returns:
            Expanded dataset.
        """
        return np.hstack([X, new_columns])

    @staticmethod
    def update_tabu_edges(
        idx_group: List[int],
        tabu_edges: List[Tuple[int, int]],
        tabu_idx_group: bool,
    ) -> List[Tuple[int, int]]:
        """
        Tabu edges are:
            1. all user defined connections to original feature column
            2. all inter-feature connections (optional)

        Args:
            idx_group: The group of indices which correspond to a single
            expanded column.

            tabu_edges: The list of tabu_edges to be updated.

            tabu_idx_group: Whether inter-group edges should also be considered tabu.
            I.e if a result of a column expansion, often want to prevent edges being learned
            between parameters.

        Returns:
            Updated tabu_edges
        """

        if tabu_edges is None:
            tabu_edges = []

        # copy to prevent mutations
        tabu_edges = deepcopy(tabu_edges)

        # handle 1.
        new_tabu_edges = []
        # for each original tabu pair
        for (i, j) in tabu_edges:
            # idx_group[0] is the original column index
            if i == idx_group[0]:
                new_tabu_edges += [(idx, j) for idx in idx_group[1:]]
            elif j == idx_group[0]:
                new_tabu_edges += [(i, idx) for idx in idx_group[1:]]
        # all new edges added to tabu_edges
        tabu_edges += new_tabu_edges

        # handle 2.
        if tabu_idx_group:
            # add on all pairwise permutations of particular feature group
            # NOTE: permutations are needed for edge directionality
            tabu_edges += list(itertools.permutations(idx_group, 2))

        return tabu_edges

    @staticmethod
    def update_tabu_nodes(
        idx_group: List[int], tabu_nodes: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Tabu nodes are:
            1. all user defined connections to original feature column

        Args:
            idx_group: The group of indices which correspond to a single
            expanded column.

            tabu_nodes: The list of tabu_nodes to be updated.

        Returns:
            Updated tabu_nodes
        """
        if tabu_nodes is None:
            return tabu_nodes

        # copy to prevent mutations
        tabu_nodes = deepcopy(tabu_nodes)

        new_tabu_nodes = []
        for i in tabu_nodes:
            # NOTE: the first element in the idx_group is guaranteed as self.idx
            if i == idx_group[0]:
                new_tabu_nodes += idx_group[1:]
        # add on the new tabu nodes
        tabu_nodes += new_tabu_nodes
        return tabu_nodes
