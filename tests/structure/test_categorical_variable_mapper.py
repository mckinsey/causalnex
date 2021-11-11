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
from collections import OrderedDict

import pytest

from causalnex.structure import StructureModel
from causalnex.structure.categorical_variable_mapper import (
    VariableFeatureMapper,
    validate_schema,
)


@pytest.fixture
def schema():
    # OrderedDict for python 3.5 compatibility
    return OrderedDict(
        [
            ("B", "binary"),
            ("C", "categorical:5"),
            (True, "binary"),
            (0, "categorical:3"),
            ((0, 1), "continuous"),
        ]
    )


@pytest.fixture
def mapper(schema):
    return VariableFeatureMapper(schema)


class TestVariableFeatureMapper:
    def test_get_indices(self, mapper):
        x = mapper.get_indices("B")
        assert x[0] == 0

        # binary + continuous first (3 variables), C comes before 0 in the
        # schema, hence the first index is 4
        x = mapper.get_indices("C")
        assert x == [3, 4, 5, 6, 7]

    def test_get_feature_index(self, mapper):
        assert mapper.get_feature_index("B") == 0
        assert mapper.get_feature_index("C_0") == 3
        # The 0 variable will be converted to a string when we
        assert mapper.get_feature_index("0_0") == 8

    def test_is_var_of_type(self, mapper):
        assert mapper.is_var_of_type("B", "binary")
        assert mapper.is_var_of_type((0, 1), "continuous")
        assert mapper.is_var_of_type(0, "categorical")
        assert mapper.is_var_of_type("C", "categorical")

        assert not mapper.is_var_of_type("B", "continuous")

        with pytest.raises(ValueError, match="Unsupported variable type unknown"):
            assert not mapper.is_var_of_type("B", "unknown")

    def test_get_feature_index_valueerror(self, mapper):
        """
        Using a categorical variable instead of the one-hot encoded feature.
        """
        with pytest.raises(ValueError, match="Input is not a feature"):
            mapper.get_feature_index("C")

    def test_get_indices_squeeze(self, mapper):
        # squeeze for binary
        x = mapper.get_indices("B", squeeze=True)
        y = mapper.get_indices("B")
        assert x == y[0]

        # squeeze for continuous
        x = mapper.get_indices((0, 1), squeeze=True)
        y = mapper.get_indices((0, 1))
        assert x == y[0]

        # no effect for categorical variables
        x = mapper.get_indices("C", squeeze=True)
        y = mapper.get_indices("C")
        assert x == y

    def test_get_indices_empty_iterator(self, schema):
        graph = StructureModel()
        # add node without parents:
        graph.add_node(10)
        mapper = VariableFeatureMapper(schema)
        x = mapper.get_indices(graph.predecessors(10))
        assert len(x) == 0
        assert isinstance(x, list)

    @pytest.mark.parametrize("invalid_input", [lambda x: x, "A", ["A"]])
    def test_get_indices_valuerror(self, mapper, invalid_input):
        with pytest.raises(ValueError, match="Provide a valid variable name"):
            mapper.get_indices(invalid_input)

    def test_get_categorical_indices(self, mapper, schema):
        cat_indices = mapper.get_categorical_indices()
        cat_features = [x for x in schema.values() if "categorical" in x]
        assert len(cat_indices) == len(cat_features)

        cat_cardinality = [
            int(x.split(":")[1]) for x in schema.values() if "categorical" in x
        ]
        assert all(
            len(cat_indices[ix]) == card for ix, card in enumerate(cat_cardinality)
        )

    def test_get_feature_names(self, mapper, schema):
        for var_name, var_schema in schema.items():
            if var_schema in ("binary", "continuous"):
                assert mapper.get_feature_names(var_name) == var_name
            elif "categorical" in var_schema:
                assert all(
                    x.startswith(str(var_name))
                    for x in mapper.get_feature_names(var_name)
                )
                assert len(mapper.get_feature_names(var_name)) == int(
                    var_schema.split(":")[1]
                )
                # assert False
        assert all(x in mapper.get_feature_names() for x in mapper.feature_list)

    def test_n_variables_len(self, mapper, schema):
        assert mapper.n_variables == len(schema)

    def test_n_features_len(self, mapper, schema):
        assert mapper.n_features == sum(
            (int(v.split(":")[1]) if "categorical" in v else 1) for v in schema.values()
        )

    @pytest.mark.parametrize("data_type", ["continuous", "binary", "categorical:3"])
    def test_single_data_type(self, data_type):
        schema = {
            "A": data_type,
            "b": data_type,
            0: data_type,
            (0, 1): data_type,
            True: data_type,
        }

        mapper = VariableFeatureMapper(schema=schema)

        assert all(
            x in schema
            for x in mapper.get_var_of_type(
                data_type=data_type.split(":")[0]
                if "categorical" in data_type
                else data_type
            )
        )

    @pytest.mark.parametrize(
        "excluded_data_type", ["continuous", "binary", "categorical"]
    )
    def test_all_but_one_data_type(self, excluded_data_type):
        data_types = VariableFeatureMapper.PERMISSIBLE_TYPES
        # create schema with all data types but one, each data type has 3 entries:
        schema = {
            len(data_types) * iy + ix: dtype
            for iy in range(3)
            for ix, dtype in enumerate(data_types)
            if dtype != excluded_data_type
        }
        schema_with_cardinality = {
            k: (v + ":3" if "categorical" in v else v) for k, v in schema.items()
        }

        mapper = VariableFeatureMapper(schema=schema_with_cardinality)
        assert all(k in mapper.get_var_of_type(v) for k, v in schema.items())


class TestValidateSchema:
    def test_unknown_data_type(self, schema):
        schema = {"new": "unknown"}
        with pytest.raises(ValueError, match="Unknown data type"):
            validate_schema(nodes={"new"}, schema=schema)

    def test_missing_cardinality(self, schema):
        schema = {"new": "categorical:"}
        with pytest.raises(ValueError, match="Missing cardinality for categorical"):
            validate_schema(nodes={"new"}, schema=schema)

        schema = {"new": "categorical:01"}
        with pytest.raises(ValueError, match="Missing cardinality for categorical"):
            validate_schema(nodes={"new"}, schema=schema)

        schema = {"new": "categorical:100"}
        validate_schema(nodes={"new"}, schema=schema)

    def test_unknown_default_schema(self):
        with pytest.raises(ValueError, match="Unknown default data type"):
            validate_schema(nodes=["new"], schema={}, default_type="unknown")

    def test_imputation(self):
        default_schema = "continuous"
        new_schema = validate_schema(
            nodes=["new"], schema=None, default_type=default_schema
        )
        assert new_schema["new"] == default_schema

    def test_correct_schema(self, schema):
        new_schema = validate_schema(nodes=list(schema.keys()), schema=schema)
        assert new_schema == schema
