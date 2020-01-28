# Copyright 2019-2020 QuantumBlack Visual Analytics Limited
#
# The methods found in this file are adapted from a repository under Apache 2.0:
# eBay's Pythonic Bayesian Belief Network Framework.
# @online{
#     author = {Neville Newey,Anzar Afaq},
#     title = {bayesian-belief-networks},
#     organisation = {eBay},
#     codebase = {https://github.com/eBay/bayesian-belief-networks},
# }
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
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#


class InvalidGraphException(Exception):
    """
    Raised if the graph verification
    method fails.
    """

    pass


class InvalidSampleException(Exception):
    """Should be raised if a
    sample is invalid."""

    pass


class InvalidInferenceMethod(Exception):
    """Raise if the user tries to set
    the inference method to an unknown string."""

    pass


class InsufficientSamplesException(Exception):
    """Raised when the inference method
    is 'sample_db' and there are less
    pre-generated samples than the
    graphs n_samples attribute."""

    pass


class NoSamplesInDB(Warning):
    pass


class VariableNotInGraphError(Exception):
    """Exception raised when
    a graph is queried with
    a variable that is not part of
    the graph.
    """

    pass


class VariableValueNotInDomainError(Exception):
    """Raised when a BBN is queried with
    a value for a variable that is not within
    that variables domain."""

    pass


class IncorrectInferenceMethodError(Exception):
    """Raise when attempt is made to
    generate samples when the inference
    method is not 'sample_db'
    """

    pass
