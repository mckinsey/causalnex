"""
Data ingestion and discretization pipeline for the global relationship graph.

Continuous variables (GDP, commodity prices, FX indices) are binned into ordinal
integer states using ``causalnex.discretiser.Discretiser``.  Event and figure
nodes that are already coded as ordinal integers (0 / 1 / 2) are passed through.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from causalnex.discretiser import Discretiser


def discretize_dataframe(
    df_raw: pd.DataFrame,
    quantile_columns: Optional[List[str]] = None,
    fixed_columns: Optional[Dict[str, List[float]]] = None,
    passthrough_columns: Optional[List[str]] = None,
    num_buckets: int = 3,
) -> pd.DataFrame:
    """Discretize a mixed raw DataFrame into ordinal integer columns.

    Three strategies are supported and can be mixed within a single call:

    * **quantile** — data-driven percentile splits (good for skewed
      distributions like GDP or commodity prices).
    * **fixed** — domain-anchored numeric split points (e.g. DXY index
      thresholds for currency strength).
    * **passthrough** — columns that are already ordinal integers and
      require no transformation.

    Args:
        df_raw: raw numeric DataFrame; every column to be included in the
            graph must be present.
        quantile_columns: columns to discretize with
            ``Discretiser(method="quantile", num_buckets=num_buckets)``.
        fixed_columns: mapping of ``{column: [split_point, ...]}``, using
            ``Discretiser(method="fixed", numeric_split_points=...)``.
        passthrough_columns: columns already coded as integers — copied as-is.
        num_buckets: number of quantile buckets (default 3).

    Returns:
        A new DataFrame with all specified columns discretized to ``int``.
    """
    quantile_columns = quantile_columns or []
    fixed_columns = fixed_columns or {}
    passthrough_columns = passthrough_columns or []

    df_disc = pd.DataFrame(index=df_raw.index)

    # Quantile discretization
    for col in quantile_columns:
        disc = Discretiser(method="quantile", num_buckets=num_buckets)
        df_disc[col] = disc.fit_transform(df_raw[col].values.reshape(-1, 1)).flatten()

    # Fixed-threshold discretization
    for col, split_points in fixed_columns.items():
        disc = Discretiser(method="fixed", numeric_split_points=split_points)
        df_disc[col] = disc.transform(df_raw[col].values.reshape(-1, 1)).flatten()

    # Passthrough (already ordinal)
    for col in passthrough_columns:
        df_disc[col] = df_raw[col].astype(int)

    return df_disc.astype(int)


def generate_synthetic_data(
    node_names: List[str],
    n_samples: int = 200,
    n_states: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic dataset for quick prototyping and tests.

    Each column is sampled uniformly from ``range(n_states)``.  This is
    *not* meant to capture real-world distributions — use it only for
    smoke-testing the pipeline end-to-end.

    Args:
        node_names: column names (should match the node registry).
        n_samples: number of rows.
        n_states: number of discrete states per column.
        seed: random seed for reproducibility.

    Returns:
        A DataFrame of shape ``(n_samples, len(node_names))`` with integer
        values in ``[0, n_states)``.
    """
    rng = np.random.RandomState(seed)
    data = rng.randint(0, n_states, size=(n_samples, len(node_names)))
    return pd.DataFrame(data, columns=node_names)
