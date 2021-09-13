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
Contains utility functions to handle categorical features. While the statistical
dependencies are defined on the node level, categorical variables require an
expansion to "one-hot" encoding for numeric analysis.
"""
import re
from collections import OrderedDict
from itertools import chain
from typing import Dict, Hashable, Iterator, List, Optional, Set, Union

from networkx.classes.reportviews import NodeView


class VariableFeatureMapper:
    """
    When expanding the columns from variables to features, this class keeps
    track of the 1:m mapping between the collapsed and expanded columns.

    Args:
        schema: A dictionary mapping a variable (can be any hashable object) to
            a data type. Permissible data types are stored in the class
            attribute ``PERMISSIBLE_TYPES``.
    """

    PERMISSIBLE_TYPES = {"binary", "categorical", "continuous", "count"}
    EXPANDABLE_TYPE = "categorical"

    def __init__(self, schema: Dict[Hashable, str]):
        # 0. get all variables
        self.variable_type_dict = OrderedDict(
            [
                (x, [k for k, v in schema.items() if x in v])
                for x in self.PERMISSIBLE_TYPES
            ]
        )

        # 1. split categorical variable into features
        # dictionary: categorical variable to cardinality
        cat_card_dict = OrderedDict(
            [
                (cat_var, int(v.split(":")[1]))
                for cat_var, v in schema.items()
                if self.EXPANDABLE_TYPE in v
            ]
        )

        # dictionary: categorical feature to variable (C:1 mapping)
        self._cat_fte_var_dict = OrderedDict(
            [
                (f"{cat_var}_{i}", cat_var)
                for cat_var, card in cat_card_dict.items()
                for i in range(card)
            ]
        )
        cat_feature_list = list(self._cat_fte_var_dict.keys())

        # we put them together with the cont + binary in a feature list
        self.feature_list = (
            self.variable_type_dict["binary"]
            + self.variable_type_dict["continuous"]
            + self.variable_type_dict["count"]
            + cat_feature_list
        )

        # 2. we assign an index to each feature
        # dictionary: feature to index
        self._fte_index_dict = {fte: ix for ix, fte in enumerate(self.feature_list)}

        # 3. map a feature to all corresponding (expanded) columns
        # dictionary: variable to indices of all corresponding features
        self.var_indices_dict = {
            var: [self._fte_index_dict[var]]
            for var in self.variable_type_dict["continuous"]
            + self.variable_type_dict["binary"]
            + self.variable_type_dict["count"]
        }
        self.var_indices_dict.update(
            {
                k: [
                    self._fte_index_dict[fte]
                    for fte, var in self._cat_fte_var_dict.items()
                    if var == k
                ]
                for k in self.variable_type_dict["categorical"]
            }
        )

    @property
    def variable_list(self) -> List[Hashable]:
        """
        Returns a list of all variables/nodes.
        """
        return list(chain.from_iterable(self.variable_type_dict.values()))

    def get_var_of_type(self, data_type: str) -> List[Hashable]:
        """
        Returns all variables/nodes corresponding to the provided data type
        Args:
            data_type: Variable type.
        Returns:
            List of variables
        Raises:
            ValueError: if the variable type is not supported
        """
        if data_type not in self.PERMISSIBLE_TYPES:
            permissible_str = ", ".join(self.PERMISSIBLE_TYPES)
            raise ValueError(
                f"Unsupported variable type {data_type}, "
                f"supported data types are: {permissible_str}"
            )
        return self.variable_type_dict[data_type]

    def is_var_of_type(self, var: Hashable, data_type: str) -> bool:
        """
        Checks whether the variable/node is of the provided data type
        Args:
            var: Variable/node
            data_type: Supported data type

        Returns:
            Boolean flag
        Raises:
            ValueError: if the variable type is not supported
        """
        return var in self.get_var_of_type(data_type=data_type)

    def get_categorical_indices(self) -> List[List[int]]:
        """
        Returns a list of lists that includes all categorical feature indices
        for all categorical variables.
        """
        return [
            self.var_indices_dict[var] for var in self.variable_type_dict["categorical"]
        ]

    def get_indices(
        self,
        var: Union[Hashable, List[Hashable], Set[Hashable], Iterator],
        squeeze: bool = False,
    ) -> Union[int, List[int]]:
        """
        Returns the indices for a variable or list of variables.

        Args:
            var: A variable/node.
            squeeze: No effect if either a list,set,"dict_iterable" (e.g. from
                ``graph.predecessor(var)``) is provided or var is a
                categorical value.

        Returns:
            A list of indices. For binary and continuous variables this will be
            a list of length one. If squeeze is True, returns the index outside
            of a list.

        Raises:
            ValueError: if an unsupported variable object is provided.
        """
        if var in self.variable_list:
            if squeeze and var not in self.get_var_of_type("categorical"):
                return self.var_indices_dict[var][0]
            return self.var_indices_dict[var]
        if isinstance(var, (list, set)):
            return [ix for v in var for ix in self.get_indices(v)]
        if hasattr(var, "__next__"):
            # Deals with DiGraph.predecessors's "dict_keyiterator" and similar
            # Iterators
            return [ix for v in list(var) for ix in self.get_indices(v)]
        raise ValueError(
            "Provide a valid variable name, a set/list/Iterator of variable "
            "names. Other iterables are not supported."
        )

    def get_feature_index(self, feature: Hashable) -> int:
        """
        Gets the feature index.

        Returns:
             The index of a feature.

        Raises:
            ValueError: if a categorical variable instead of a categorical
                "one-hot" feature is provided.
        """
        if feature in self.variable_list and feature not in self.feature_list:
            raise ValueError(
                "Input is not a feature, use ``get_indices`` to get the indices "
                "associated for a variable/node."
            )
        return self._fte_index_dict[feature]

    def get_feature_names(
        self, var: Optional[Hashable] = None
    ) -> Union[Hashable, List[Hashable]]:
        """
        Get the feature name(s) corresponding to the variable. If none provided,
        returns all features.

        Returns:
             Returns all feature names corresponding to a variable/node.
                - For binary and continuous variables, this is the variable/node
                  itself.
                - For categorical variables, it returns a List of Hashables
        """
        if var is None:
            return self.feature_list
        if var not in self.variable_type_dict["categorical"]:
            return var
        return [k for k, v in self._cat_fte_var_dict.items() if v == var]

    @property
    def n_variables(self):
        """
        Returns:
            The number of variables
        """
        return len(self.variable_list)

    @property
    def n_features(self):
        """
        The number of features. If the schema only has binary and continuous
        variables, this is equal to ``n_variables``.
        """
        return len(self.feature_list)


def validate_schema(
    nodes: Union[List[Hashable], Set[Hashable], NodeView],
    default_type: str = "continuous",
    schema: Optional[Dict[Hashable, str]] = None,
) -> Dict:
    """
    Verifies category type and uses default data type for unspecified variables.

    Variables in the schema but not in the node list are ignored. The ``nodes``
    object is taken as the ground truth for variables to process.

    Args:
        nodes: All variables that should have a schema.
        schema: Dictionary mapping a variable to a data type.
        default_type: Allowed data types are 'binary', 'continuous',
            'categorical:X' where X stands for the cardinality of the category.
            Leading zeros are not allowed for the cardinality.
    Returns:
        Schema with missing type imputed by ``default_type``

    Raises:
        ValueError: for unknown data type
        ValueError: for missing cardinality for categorical variables
    """
    if not any(x in default_type for x in VariableFeatureMapper.PERMISSIBLE_TYPES):
        permissible_str = ", ".join(VariableFeatureMapper.PERMISSIBLE_TYPES)
        raise ValueError(
            f"Unknown default data type. Supported data types are {permissible_str}"
        )

    schema = {} if schema is None else schema
    # # add default data type to missing nodes
    schema = {k: schema.get(k, default_type) for k in nodes}

    # verify if the data type is supported
    if not all(
        any(t in x for t in VariableFeatureMapper.PERMISSIBLE_TYPES)
        for x in schema.values()
    ):
        unknown_vars = [
            k
            for k, v in schema.items()
            if v not in VariableFeatureMapper.PERMISSIBLE_TYPES
        ]
        permissible_str = ", ".join(VariableFeatureMapper.PERMISSIBLE_TYPES)
        raise ValueError(
            f"Unknown data type for variable(s) {unknown_vars}, "
            f"Supported data types are {permissible_str}"
        )

    missing_cardinality = {
        k
        for k, v in schema.items()
        if "categorical" in v and re.match(r"^categorical:[1-9]+[0-9]*", v) is None
    }

    if missing_cardinality:
        raise ValueError(
            f"Missing cardinality for categorical variable(s) {missing_cardinality} "
            "For example, specify the data type as `categorical:3` for a "
            "3-class categorical feature. Leading zeros are not allowed."
        )
    return schema
