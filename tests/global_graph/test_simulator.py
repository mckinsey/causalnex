"""Tests for causalnex.global_graph.simulator."""

import numpy as np
import pandas as pd
import pytest

from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.global_graph.simulator import GlobalRippleSimulator
from causalnex.global_graph.node_registry import NODE_REGISTRY


def _make_test_bn():
    """Build a small BN matching a subset of the default expert skeleton."""
    sm = StructureModel()
    sm.add_edge("fig_mbs", "evt_opec_cut", origin="expert")
    sm.add_edge("evt_opec_cut", "res_oil_brent", origin="expert")
    sm.add_edge("res_oil_brent", "cur_usd_strength", origin="expert")
    sm.add_edge("res_oil_brent", "ctr_russia_gdp", origin="expert")

    rng = np.random.RandomState(42)
    n = 300
    nodes = ["fig_mbs", "evt_opec_cut", "res_oil_brent", "cur_usd_strength", "ctr_russia_gdp"]
    data = {}
    data["fig_mbs"] = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    data["evt_opec_cut"] = np.where(
        data["fig_mbs"] == 2, rng.choice([1, 2], size=n), rng.choice([0, 1], size=n)
    )
    data["res_oil_brent"] = np.where(
        data["evt_opec_cut"] >= 1, rng.choice([1, 2], size=n), rng.choice([0, 1], size=n)
    )
    data["cur_usd_strength"] = np.where(
        data["res_oil_brent"] >= 1, rng.choice([1, 2], size=n), rng.choice([0, 1], size=n)
    )
    data["ctr_russia_gdp"] = np.where(
        data["res_oil_brent"] >= 2, rng.choice([1, 2], size=n), rng.choice([0, 1], size=n)
    )

    df = pd.DataFrame(data)
    bn = BayesianNetwork(sm)
    bn.fit_node_states(df)
    bn.fit_cpds(df, method="BayesianEstimator", bayes_prior="K2")
    return bn, sm


class TestGlobalRippleSimulator:
    def test_simulate_scenario_returns_expected_keys(self):
        bn, sm = _make_test_bn()
        sim = GlobalRippleSimulator(bn, sm, NODE_REGISTRY)
        result = sim.simulate_scenario(
            "OPEC Major Cut", {"evt_opec_cut": 2}
        )
        assert "scenario" in result
        assert "interventions" in result
        assert "ranked_impact" in result
        assert "post_marginals" in result
        assert "narrative" in result
        assert "propagation_paths" in result

    def test_scenario_name_preserved(self):
        bn, sm = _make_test_bn()
        sim = GlobalRippleSimulator(bn, sm, NODE_REGISTRY)
        result = sim.simulate_scenario("test123", {"evt_opec_cut": 2})
        assert result["scenario"] == "test123"

    def test_narrative_is_string(self):
        bn, sm = _make_test_bn()
        sim = GlobalRippleSimulator(bn, sm, NODE_REGISTRY)
        result = sim.simulate_scenario("test", {"evt_opec_cut": 2})
        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 0

    def test_ranked_impact_excludes_intervened(self):
        bn, sm = _make_test_bn()
        sim = GlobalRippleSimulator(bn, sm, NODE_REGISTRY)
        result = sim.simulate_scenario("test", {"evt_opec_cut": 2})
        nodes = [n for n, _ in result["ranked_impact"]]
        assert "evt_opec_cut" not in nodes

    def test_propagation_paths_are_lists(self):
        bn, sm = _make_test_bn()
        sim = GlobalRippleSimulator(bn, sm, NODE_REGISTRY)
        result = sim.simulate_scenario("test", {"evt_opec_cut": 2})
        for node, paths in result["propagation_paths"].items():
            assert isinstance(paths, list)
            for path in paths:
                assert isinstance(path, list)
                assert path[0] == "evt_opec_cut"
                assert path[-1] == node
