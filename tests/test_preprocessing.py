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

import math

import numpy as np
import pytest

from causalnex.discretiser import Discretiser


class TestUniform:
    def test_fit_creates_exactly_uniform_splits_when_possible(self):
        """splits should be exactly uniform"""

        arr = np.array(range(21))
        np.random.shuffle(arr)
        d = Discretiser(method="uniform", num_buckets=5)
        d.fit(arr)
        for n in range(2):
            assert (d.numeric_split_points[n + 1] - d.numeric_split_points[n]) == (
                (d.numeric_split_points[n + 2] - d.numeric_split_points[n + 1])
            )

    def test_fit_does_not_attempt_to_deal_with_identical_split_points(self):
        """if all data is identical, and num_buckets>1, then this is not possible.
        In this case the standard behaviour of numpy is followed, and many identical
        splits will be created. See transform for how these are applied"""

        arr = np.array([1 for _ in range(20)])
        d = Discretiser(method="uniform", num_buckets=4)
        d.fit(arr)
        assert np.array_equal(
            np.array([d.numeric_split_points[0] for _ in range(3)]),
            d.numeric_split_points,
        )

    def test_transform_larger_than_fit_range_goes_into_last_bucket(self):
        """If a value larger than the input is transformed, then it
        should go into the maximum bucket"""

        arr = np.array([n + 1 for n in range(10)])
        np.random.shuffle(arr)
        d = Discretiser(method="uniform", num_buckets=4)
        d.fit(arr)
        assert np.array_equal([3], d.transform(np.array([101])))

    def test_transform_smaller_than_fit_range_goes_into_first_bucket(self):
        """If a value smaller than the input is transformed, then it
        should go into the minimum bucket"""

        arr = np.array([n + 1 for n in range(10)])
        np.random.shuffle(arr)
        d = Discretiser(method="uniform", num_buckets=4)
        d.fit(arr)
        assert np.array_equal([0], d.transform(np.array([-101])))

    def test_fit_transform(self):
        """fit transform should give the same result as calling fit and
        transform separately"""

        arr = np.array([n + 1 for n in range(10)])
        np.random.shuffle(arr)

        d1 = Discretiser(method="uniform", num_buckets=4)
        d1.fit(arr)
        r1 = d1.transform(arr)

        d2 = Discretiser(method="uniform", num_buckets=4)
        r2 = d2.fit_transform(arr)

        assert np.array_equal(r1, r2)


class TestQuantile:
    def test_fit_uniform_data(self):
        """Fitting uniform data should produce uniform splits"""

        arr = np.array(range(100001))
        np.random.shuffle(arr)
        d = Discretiser(method="quantile", num_buckets=4)
        d.fit(arr)
        assert np.array_equal([25000, 50000, 75000], d.numeric_split_points)

    def test_fit_gauss_data(self):
        """Fitting gauss data should produce standard percentiles splits"""

        arr = np.random.normal(loc=0, scale=1, size=100001)
        np.random.shuffle(arr)
        d = Discretiser(method="quantile", num_buckets=4)
        d.fit(arr)
        assert math.isclose(-0.675, d.numeric_split_points[0], abs_tol=0.025)
        assert math.isclose(0, d.numeric_split_points[1], abs_tol=0.025)
        assert math.isclose(0.675, d.numeric_split_points[2], abs_tol=0.025)

    def test_transform_gauss(self):
        """Fitting gauss data should transform to predictable buckets"""

        arr = np.random.normal(loc=0, scale=1, size=1000000)
        np.random.shuffle(arr)
        d = Discretiser(method="quantile", num_buckets=4)
        d.fit(arr)
        unique, counts = np.unique(d.transform(arr), return_counts=True)
        # check all 4 buckets are used
        assert np.array_equal([0, 1, 2, 3], unique)
        assert np.array_equal([250000 for n in range(4)], counts)

    def test_fit_transform(self):
        """fit transform should give the same result as calling fit and
        transform separately"""

        arr = np.array([n + 1 for n in range(10)])
        np.random.shuffle(arr)

        d1 = Discretiser(method="quantile", num_buckets=4)
        d1.fit(arr)
        r1 = d1.transform(arr)

        d2 = Discretiser(method="quantile", num_buckets=4)
        r2 = d2.fit_transform(arr)

        assert np.array_equal(r1, r2)


class TestOutlier:
    def test_outlier_percentile_lower_boundary(self):
        """Discretiser should accept lower boundary down to zero"""

        Discretiser(method="outlier", outlier_percentile=0.0)
        Discretiser(method="outlier", outlier_percentile=-0.0)
        with pytest.raises(ValueError):
            Discretiser(method="outlier", outlier_percentile=-0.1)

    def test_outlier_percentile_upper_boundary(self):
        """Discretiser should accept upper boundary up to half"""

        Discretiser(method="outlier", outlier_percentile=0.49)
        with pytest.raises(ValueError):
            Discretiser(method="outlier", outlier_percentile=0.5)

    def test_outlier_lower_percentile(self):
        """the split point for lower outliers should be at provided percentile"""

        arr = np.array(range(100001))
        np.random.shuffle(arr)
        d = Discretiser(method="outlier", outlier_percentile=0.2)
        d.fit(arr)
        assert d.numeric_split_points[0] == 20000

    def test_outlier_upper_percentile(self):
        """the split point for upper outliers should be at range - provided percentile"""

        arr = np.array(range(100001))
        np.random.shuffle(arr)
        d = Discretiser(method="outlier", outlier_percentile=0.2)
        d.fit(arr)
        assert d.numeric_split_points[1] == 80000

    def test_transform_outlier(self):
        """transforming outliers should put the expected amount of data in each bucket"""

        arr = np.array(range(100001))
        np.random.shuffle(arr)
        d = Discretiser(method="outlier", outlier_percentile=0.2)
        d.fit(arr)
        unique, counts = np.unique(d.transform(arr), return_counts=True)
        # check all 3 buckets are used
        assert np.array_equal([0, 1, 2], unique)
        # check largest difference in outliers is 1
        print(counts)
        assert np.abs(counts[0] - counts[2]) <= 1

    def test_fit_transform(self):
        """fit transform should give the same result as calling fit and
        transform separately"""

        arr = np.array([n + 1 for n in range(10)])
        np.random.shuffle(arr)

        d1 = Discretiser(method="outlier", outlier_percentile=0.2)
        d1.fit(arr)
        r1 = d1.transform(arr)

        d2 = Discretiser(method="outlier", outlier_percentile=0.2)
        r2 = d2.fit_transform(arr)

        assert np.array_equal(r1, r2)


class TestFixed:
    def test_fit_raises_error(self):
        """since numeric split points are provided, fit will not do anything"""

        d = Discretiser(method="fixed", numeric_split_points=[1])
        with pytest.raises(RuntimeError):
            d.fit(np.array([1]))

    def test_fit_transform_raises_error(self):
        """since numeric split points are provided, fit will not do anything"""

        d = Discretiser(method="fixed", numeric_split_points=[1])
        with pytest.raises(RuntimeError):
            d.fit_transform(np.array([1]))

    def test_transform_splits_using_defined_split_points(self):
        """transforming should be done using the provided numeric split points"""

        d = Discretiser(method="fixed", numeric_split_points=[10, 20, 30])
        transformed = d.transform(np.array([9, 10, 11, 19, 20, 21, 29, 30, 31]))
        assert np.array_equal(transformed, [0, 1, 1, 1, 2, 2, 2, 3, 3])


class TestErrorHandling:
    def test_invalid_method(self):
        """a value error should be raised if an invalid method is given"""

        allowed_methods = ["uniform", "quantile", "outlier", "fixed", "percentiles"]
        selected_method = "INVALID"
        with pytest.raises(
            ValueError,
            match=f"{selected_method} is not a recognised method. "
            f"Use one of: {' '.join(allowed_methods)}",
        ):
            Discretiser(method=selected_method)

    def test_uniform_requires_num_buckets(self):
        """a value error should be raised if method=uniform and num_buckets is not provided"""

        selected_method = "uniform"
        with pytest.raises(
            ValueError,
            match=f"{selected_method} method expects num_buckets",
        ):
            Discretiser(method=selected_method)

    def test_quantile_requires_num_buckets(self):
        """a value error should be raised if method=quantile and num_buckets is not provided"""

        selected_method = "quantile"
        with pytest.raises(
            ValueError,
            match=f"{selected_method} method expects num_buckets",
        ):
            Discretiser(method=selected_method)

    def test_outlier_requires_outlier_percentile(self):
        """a value error should be raised if method=outlier and outlier_percentile is not provided"""

        selected_method = "outlier"
        with pytest.raises(
            ValueError,
            match=f"{selected_method} method expects outlier_percentile",
        ):
            Discretiser(method=selected_method)

    def test_outlier_geq_zero(self):
        """a value error should be raised if outlier is not >= 0"""

        Discretiser(method="outlier", outlier_percentile=0.0)
        Discretiser(method="outlier", outlier_percentile=-0.0)
        Discretiser(method="outlier", outlier_percentile=0.1)
        with pytest.raises(
            ValueError,
            match="outlier_percentile must be between 0 and 0.5",
        ):
            Discretiser(method="outlier", outlier_percentile=-0.0000001)

    def test_outlier_lt_half(self):
        """a value error should be raised if outlier is not < 0.5"""

        Discretiser(method="outlier", outlier_percentile=0.49)
        with pytest.raises(
            ValueError,
            match="outlier_percentile must be between 0 and 0.5",
        ):
            Discretiser(method="outlier", outlier_percentile=0.5)

    def test_fixed_split_points(self):
        """a value error should be raised if method=fixed and no numeric split points are provided"""

        selected_method = "fixed"
        with pytest.raises(
            ValueError,
            match=f"{selected_method} method expects numeric_split_points",
        ):
            Discretiser(method=selected_method)

    def test_fixed_split_points_monotonic(self):
        """a value error should be raised if numeric split points are not monotonically increasing"""

        Discretiser(method="fixed", numeric_split_points=[-1, -0, 0, 1])
        with pytest.raises(
            ValueError,
            match="numeric_split_points must be monotonically increasing",
        ):
            Discretiser(method="fixed", numeric_split_points=[1, -1])

    def test_percentile_requires_percentile_split_points(self):
        """a value error should be raised if method=percentiles and no percentile split points are provided"""

        selected_method = "percentiles"
        with pytest.raises(
            ValueError,
            match=f"{selected_method} method expects percentile_split_points",
        ):
            Discretiser(method=selected_method)

    def test_percentile_geq_zero(self):
        """a value error should be raised if not all percentiles split points >= 0"""

        Discretiser(method="percentiles", percentile_split_points=[-0.0, 0.0, 0.0001])
        with pytest.raises(
            ValueError,
            match="percentile_split_points must be between 0 and 1",
        ):
            Discretiser(
                method="percentiles", percentile_split_points=[-0.0000001, 0.0001]
            )

    def test_percentile_leq_1(self):
        """a value error should be raised if not all percentile split points <= 1"""

        Discretiser(method="percentiles", percentile_split_points=[0.0001, 1])
        with pytest.raises(
            ValueError,
            match="percentile_split_points must be between 0 and 1",
        ):
            Discretiser(
                method="percentiles", percentile_split_points=[0.0001, 1.0000001]
            )

    def test_percentile_split_points_monotonic(self):
        """a value error should be raised if percentile split points are not monotonically increasing"""

        Discretiser(method="percentiles", percentile_split_points=[0, -0, 0.1, 1])
        with pytest.raises(
            ValueError,
            match="percentile_split_points must be monotonically increasing",
        ):
            Discretiser(method="percentiles", percentile_split_points=[1, 0.1])


class TestPercentile:
    def test_fit_uniform_data(self):
        """Fitting uniform data should produce expected percentile splits of uniform distribution"""

        arr = np.array(range(100001))
        np.random.shuffle(arr)
        d = Discretiser(method="percentiles", percentile_split_points=[0.1, 0.4, 0.85])
        d.fit(arr)
        assert np.array_equal([10000, 40000, 85000], d.numeric_split_points)

    def test_fit_gauss_data(self):
        """Fitting gauss data should produce percentile splits of standard normal distribution"""

        arr = np.random.normal(loc=0, scale=1, size=100001)
        np.random.shuffle(arr)
        d = Discretiser(method="percentiles", percentile_split_points=[0.1, 0.4, 0.85])
        d.fit(arr)
        assert math.isclose(-1.2815, d.numeric_split_points[0], abs_tol=0.025)
        assert math.isclose(-0.253, d.numeric_split_points[1], abs_tol=0.025)
        assert math.isclose(1.036, d.numeric_split_points[2], abs_tol=0.025)

    def test_transform_uniform(self):
        """Fitting uniform data should transform to predictable buckets"""

        arr = np.array(range(100001))
        np.random.shuffle(arr)
        d = Discretiser(
            method="percentiles", percentile_split_points=[0.10, 0.40, 0.85]
        )
        d.fit(arr)
        unique, counts = np.unique(d.transform(arr), return_counts=True)
        # check all 4 buckets are used
        assert np.array_equal([0, 1, 2, 3], unique)
        assert np.array_equal([10000, 30000, 45000, 15001], counts)

    def test_fit_transform(self):
        """fit transform should give the same result as calling fit and
        transform separately"""

        arr = np.array([n + 1 for n in range(10)])
        np.random.shuffle(arr)

        d1 = Discretiser(
            method="percentiles", percentile_split_points=[0.10, 0.40, 0.85]
        )
        d1.fit(arr)
        r1 = d1.transform(arr)

        d2 = Discretiser(
            method="percentiles", percentile_split_points=[0.10, 0.40, 0.85]
        )
        r2 = d2.fit_transform(arr)

        assert np.array_equal(r1, r2)
