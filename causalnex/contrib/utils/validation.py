"""Utilities for input validation"""

# Authors: Jean-Baptiste Oger

import numpy as np

def assert_all_finite(
    X: np.ndarray,
    allow_nan: bool = False
    ):
    """Throw a ValueError if X contains NaN or Infinity.

    Based on Sklearn method to handle NaN & Infinity.
        @inproceedings{sklearn_api,
        author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
                    Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
                    Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
                    and Jaques Grobler and Robert Layton and Jake VanderPlas and
                    Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
        title     = {{API} design for machine learning software: experiences from the scikit-learn
                    project},
        booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
        year      = {2013},
        pages = {108--122},
        }

    Args:
        X: array
        allow_nan: bool

    Raises:
        ValueError: If X contains NaN or Infinity
    """
    is_float = X.dtype.kind in 'fc'
    if is_float and (np.isfinite(np.sum(X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(
                    msg_err.format
                    (type_err,
                    X.dtype)
            )
