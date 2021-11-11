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
"""Some Useful Helper Functions"""
import inspect

# TODO: Find a better location for get_args


def get_args(func):
    """
    Return the names of the arguments
    of a function as a list of strings.
    This is so that we can omit certain
    variables when we marginalize.
    Note that functions created by
    make_product_func do not return
    an argspec, so we add a argspec
    attribute at creation time.
    """
    if hasattr(func, "argspec"):
        return func.argspec
    # return inspect.getargspec(func).args
    return [p for p in inspect.signature(func).parameters]


def make_key(*args):
    """Handy for short truth table keys"""
    key = ""
    for a in args:
        if hasattr(a, "value"):
            raise ValueError("Unexpected type")
        else:
            key += str(a).lower()[0]
    return key


def get_original_factors(factors):
    """
    For a set of factors, we want to
    get a mapping of the variables to
    the factor which first introduces the
    variable to the set.
    To do this without enforcing a special
    naming convention such as 'f_' for factors,
    or a special ordering, such as the last
    argument is always the new variable,
    we will have to discover the 'original'
    factor that introduces the variable
    iteratively.
    """
    original_factors = {}
    while len(original_factors) < len(factors):
        for factor in factors:
            args = get_args(factor)
            unaccounted_args = [a for a in args if a not in original_factors]
            if len(unaccounted_args) == 1:
                original_factors[unaccounted_args[0]] = factor
    return original_factors
