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
``causalnex.pytorch.data_type.ordinal`` defines the ordinal distribution type.
"""

from copy import deepcopy

import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder

from causalnex.structure.pytorch.dist_type._base import DistTypeBase


class DistTypeOrdinal(DistTypeBase):
    """Class defining ordinal distribution type functionality"""

    # log cumulative odds of original distro
    log_cum_odds = None
    # ordinal encoder
    encoder = None

    def preprocess_X(self, X: np.ndarray, fit_transform: bool = True) -> np.ndarray:
        """
        Ordinal Transforms the data.
        Calculates the log cumulative odds of the original distribution.

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

        # deepcopy to prevent overwrite errors
        X = deepcopy(X)

        # fit the OrdinalEncoder
        if fit_transform:
            self.encoder = OrdinalEncoder()
            self.encoder.fit(X[:, [self.idx]])

        # Ordinal Encoder
        X[:, [self.idx]] = self.encoder.transform(X[:, [self.idx]])

        # only calc log_cum_odds on fit step
        if fit_transform:
            # calculate the cumulative probability across each category
            _, counts_elements = np.unique(X[:, self.idx], return_counts=True)
            class_probs = counts_elements / np.sum(counts_elements)
            cum_class_probs = np.cumsum(class_probs)
            # last value is guaranteed to be 1.0 so remove to prevent divide by zero
            cum_class_probs = cum_class_probs[:-1]

            # store the log cumulative odds across all categories
            self.log_cum_odds = np.log(cum_class_probs / (1 - cum_class_probs))

        return X

    def _get_probs(self, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Shifts and converts the original cumulative probabilities to probabilities.

        Args:
            X_hat: The reconstructed data.

        Returns:
            Tensor of ordered class probabilities.
        """
        # convert to torch tensor
        log_cum_odds = torch.as_tensor(self.log_cum_odds, device=X_hat.device)
        # shift the log probs by the NEGATIVE of the regression coefficient
        # the reason for negativity is explained p386 of Statistical Rethinking
        log_cum_odds = torch.unsqueeze(log_cum_odds, dim=0) - X_hat[:, [self.idx]]
        # recover the cumulative probabilities
        cum_prob = torch.sigmoid(log_cum_odds)

        # allocate class probability vector with extra space for top class
        probs = torch.zeros(
            (cum_prob.shape[0], cum_prob.shape[1] + 1), device=X_hat.device
        )
        # handle first and last class cumprobs to probs conversion
        probs[:, 0] = cum_prob[:, 0]
        probs[:, -1] = 1 - cum_prob[:, -1]
        # handle main cumprob conversion
        probs[:, 1:-1] = cum_prob[:, 1:] - cum_prob[:, 0:-1]

        return probs

    def loss(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Uses the cumulative link function to model the ordered categorical variables.
        See "Statistical Rethinking" p380 for further details.

        Args:
            X: The original data passed into NOTEARS (i.e. the reconstruction target).

            X_hat: The reconstructed data.

        Returns:
            Scalar pytorch tensor of the reconstruction loss between X and X_hat.
        """

        # get the predicted probabilities
        probs = self._get_probs(X_hat)
        # index probs based on the ordinal class
        gather_idx = (X[:, self.idx]).long().unsqueeze(dim=1)
        predicted_prob = probs.gather(dim=1, index=gather_idx).squeeze()
        # nll
        loss = -torch.log(predicted_prob)

        return torch.mean(loss)

    def inverse_link_function(self, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Inverse cumulative link returns the probability of each category.

        Args:
            X_hat: Reconstructed data in the latent space.

        Returns:
            Modified X_hat.
            MUST be same shape as passed in data.
            Projects the self.idx column from the latent space to the dist_type space.
        """
        return self._get_probs(X_hat)

    def get_columns(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Gets the column(s) associated with the instantiated DistType.

        NOTE: ordinal get_columns is special in that inverse_link_function does NOT return
        the entire dataset.

        Args:
            X: The return data from inverse_link_function.

        Returns:
            1d or 2d np.ndarray of columns.
        """
        return X
