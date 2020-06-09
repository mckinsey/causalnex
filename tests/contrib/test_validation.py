import pytest

import numpy as np

from causalnex.contrib.utils.validation import (
  assert_all_finite
)

class TestValidation:
    def test_array_with_nan_raises_error(self):
        with pytest.raises(ValueError, match="Input contains NaN, infinity or a value too large for dtype*"):
          arr = np.ones((1,1))
          arr[0,0] = np.nan
          assert_all_finite(arr)

    def test_array_with_inf_raises_error(self):
        with pytest.raises(ValueError, match="Input contains NaN, infinity or a value too large for dtype*"):
          arr = np.ones((1,1))
          arr[0,0] = np.inf
          assert_all_finite(arr)