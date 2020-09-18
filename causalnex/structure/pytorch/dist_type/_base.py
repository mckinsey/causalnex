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
``causalnex.pytorch.data_type._base`` defines the distribution type class interface and default behavior.
"""

from abc import ABCMeta, abstractmethod

import torch


class DistTypeBase(metaclass=ABCMeta):
    """ Base class defining the distribution default behavior and interface """

    def __init__(self, idx: int):
        """
        Default constructor for the DistTypeBase class.
        Unless overridden, provides default behavior to all subclasses.

        Args:
            idx: Positional index in data passed to the NOTEARS algorithm
            which correspond to this datatype.
        """
        self.idx = idx

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
