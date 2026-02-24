"""Tests for causalnex.global_graph.ripple_engine."""

import numpy as np
import pandas as pd
import pytest

from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.global_graph.ripple_engine import (
    _jsd,
    compute_ripple_effect,
    compound_ripple_effect,
)


def _make_simple_bn():
    """Build a minimal 3-node BN for testing: A -> B -> C."""
    sm = StructureModel()
    sm.add_edge("a", "b", origin="expert")
    sm.add_edge("b", "c", origin="expert")

    rng = np.random.RandomState(42)
    n = 200
    a = rng.choice([0, 1, 2], size=n, p=[0.2, 0.3, 0.5])
    b = np.where(a == 2, rng.choice([1, 2], size=n), rng.choice([0, 1], size=n))
    c = np.where(b >= 1, rng.choice([1, 2], size=n), 0)
    df = pd.DataFrame({"a": a, "b": b, "c": c})

    bn = BayesianNetwork(sm)
    bn.fit_node_states(df)
    bn.fit_cpds(df, method="BayesianEstimator", bayes_prior="K2")
    return bn


class TestJSD:
    def test_identical_distributions(self):
        p = np.array([0.3, 0.3, 0.4])
        assert np.isclose(_jsd(p, p), 0.0, atol=1e-10)

    def test_completely_different(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        jsd = _jsd(p, q)
        assert 0 < jsd <= 1.0

    def test_symmetric(self):
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.1, 0.4, 0.5])
        assert np.isclose(_jsd(p, q), _jsd(q, p))

    def test_bounded_zero_to_one(self):
        rng = np.random.RandomState(0)
        for _ in range(50):
            p = rng.dirichlet([1, 1, 1])
            q = rng.dirichlet([1, 1, 1])
            jsd = _jsd(p, q)
            assert 0 <= jsd <= 1.0 + 1e-10


class TestComputeRippleEffect:
    def test_returns_sorted_list(self):
        bn = _make_simple_bn()
        result = compute_ripple_effect(bn, "a", 2)
        assert isinstance(result, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)
        # Sorted descending by JSD
        jsds = [jsd for _, jsd in result]
        assert jsds == sorted(jsds, reverse=True)

    def test_intervened_node_excluded(self):
        bn = _make_simple_bn()
        result = compute_ripple_effect(bn, "a", 2)
        nodes = [n for n, _ in result]
        assert "a" not in nodes

    def test_top_k_limits_results(self):
        bn = _make_simple_bn()
        result = compute_ripple_effect(bn, "a", 2, top_k=1)
        assert len(result) == 1

    def test_direct_child_more_affected_than_grandchild(self):
        bn = _make_simple_bn()
        result = compute_ripple_effect(bn, "a", 2)
        impact = dict(result)
        # b is direct child, c is grandchild — b should be at least as affected
        assert impact["b"] >= impact["c"] - 0.01


class TestCompoundRippleEffect:
    def test_multiple_interventions(self):
        bn = _make_simple_bn()
        result = compound_ripple_effect(bn, {"a": 2, "b": 2})
        nodes = [n for n, _ in result]
        assert "a" not in nodes
        assert "b" not in nodes
        assert "c" in nodes

    def test_returns_sorted(self):
        bn = _make_simple_bn()
        result = compound_ripple_effect(bn, {"a": 0})
        jsds = [jsd for _, jsd in result]
        assert jsds == sorted(jsds, reverse=True)
