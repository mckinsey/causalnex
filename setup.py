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

import re
from itertools import chain
from os import path

from setuptools import find_packages, setup

name = "causalnex"
here = path.abspath(path.dirname(__file__))

# get package version
with open(path.join(here, name, "__init__.py"), encoding="utf-8") as f:
    result = re.search(r'__version__ = ["\']([^"\']+)', f.read())
    if not result:
        raise ValueError("Can't find the version in causalnex/__init__.py")
    version = result.group(1)

# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = [x.strip() for x in f if x.strip()]

# get test dependencies and installs
with open("test_requirements.txt", "r", encoding="utf-8") as f:
    test_requires = [x.strip() for x in f if x.strip() and not x.startswith("-r")]

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    readme = f.read()

extras_require = {
    "plot": [
        "pygraphviz>=1.5, <2.0",
    ],
}

extras_require["all"] = sorted(chain.from_iterable(extras_require.values()))

setup(
    name=name,
    version=version,
    license="Apache Software License (Apache 2.0)",
    description="Toolkit for causal reasoning (Bayesian Networks / Inference)",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/quantumblacklabs/causalnex",
    python_requires=">=3.6, <3.9",
    author="QuantumBlack Labs",
    author_email="causalnex@quantumblack.com",
    packages=find_packages(exclude=["docs*", "tests*", "tools*"]),
    include_package_data=True,
    tests_require=test_requires,
    install_requires=requires,
    keywords="Causal Reasoning, Bayesian Network, Inference, Structure Learning, Do-Calculus",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    extras_require=extras_require,
)
