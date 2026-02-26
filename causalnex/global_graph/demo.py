"""
Global graph demo — end-to-end pipeline that generates an HTML ripple-effect
visualization you can open in any browser.

Usage::

    python -m causalnex.global_graph.demo
    # or
    python causalnex/global_graph/demo.py

Output:
    global_ripple_demo.html   — interactive causal graph coloured by impact
    global_ripple_demo2.html  — second scenario (Taiwan + trade-war)

The script uses *synthetic* data so that it runs with zero external
dependencies.  Replace ``generate_synthetic_data(...)`` with your own
DataFrame of real observations to get meaningful CPDs and impact scores.
"""

import os
import sys

# Allow running as a standalone script from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _step(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def run_demo(output_dir: str = ".") -> None:
    """Run the full pipeline and write HTML files to *output_dir*."""

    # ------------------------------------------------------------------
    # 1. Load ontology from YAML config
    # ------------------------------------------------------------------
    _step("1/6  Loading ontology registry")
    from causalnex.global_graph import get_default_registry
    registry = get_default_registry()
    print(registry)

    # ------------------------------------------------------------------
    # 2. Build expert skeleton DAG
    # ------------------------------------------------------------------
    _step("2/6  Building expert skeleton DAG")
    from causalnex.global_graph import build_expert_skeleton, enforce_dag
    sm = build_expert_skeleton(registry=registry)
    sm = enforce_dag(sm)
    print(f"  Nodes : {len(sm.nodes)}")
    print(f"  Edges : {len(sm.edges)}")

    # ------------------------------------------------------------------
    # 3. Fit BayesianNetwork with synthetic data
    # ------------------------------------------------------------------
    _step("3/6  Fitting BayesianNetwork (synthetic data)")
    from causalnex.global_graph import build_bayesian_network
    from causalnex.global_graph.data_ingestion import generate_synthetic_data

    node_names = list(sm.nodes)
    df = generate_synthetic_data(node_names, n_samples=500, n_states=3, seed=42)
    bn = build_bayesian_network(sm, df)
    print("  BayesianNetwork fitted successfully.")
    print("  NOTE: using synthetic uniform data — replace with real")
    print("        historical data for meaningful impact scores.")

    # ------------------------------------------------------------------
    # 4. Scenario A — Ukraine war escalation + OPEC cut
    # ------------------------------------------------------------------
    _step("4/6  Scenario A: Ukraine Escalation + OPEC Cut")
    from causalnex.global_graph import GlobalRippleSimulator

    sim = GlobalRippleSimulator(bn, sm, registry=registry)

    scenario_a = sim.simulate_scenario(
        scenario_name="Ukraine Escalation + OPEC Cut",
        events={"evt_ukraine_war": 2, "evt_opec_cut": 2},
        top_k=15,
    )
    print(scenario_a["narrative"])

    # ------------------------------------------------------------------
    # 5. Scenario B — Taiwan tensions + US-China trade war
    # ------------------------------------------------------------------
    _step("5/6  Scenario B: Taiwan Crisis + Trade War")

    scenario_b = sim.simulate_scenario(
        scenario_name="Taiwan Crisis + US-China Trade War",
        events={"evt_taiwan_tensions": 2, "evt_us_china_trade_war": 2},
        top_k=15,
    )
    print(scenario_b["narrative"])

    # ------------------------------------------------------------------
    # 6. Generate HTML visualizations
    # ------------------------------------------------------------------
    _step("6/6  Generating HTML visualizations")
    from causalnex.global_graph import visualize_impact_heatmap

    path_a = os.path.join(output_dir, "global_ripple_scenario_a.html")
    path_b = os.path.join(output_dir, "global_ripple_scenario_b.html")

    visualize_impact_heatmap(
        sm=sm,
        impact_scores=scenario_a["ranked_impact"],
        intervened_nodes=list(scenario_a["interventions"].keys()),
        output_path=path_a,
        registry=registry,
    )
    print(f"  Saved: {os.path.abspath(path_a)}")

    visualize_impact_heatmap(
        sm=sm,
        impact_scores=scenario_b["ranked_impact"],
        intervened_nodes=list(scenario_b["interventions"].keys()),
        output_path=path_b,
        registry=registry,
    )
    print(f"  Saved: {os.path.abspath(path_b)}")

    print("\nDone! To view:")
    print(f"  python -m http.server 8080   # then open in browser:")
    print(f"  http://localhost:8080/{os.path.basename(path_a)}")
    print(f"  http://localhost:8080/{os.path.basename(path_b)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate global-graph ripple-effect HTML visualizations."
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write HTML files into (default: current directory)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_demo(output_dir=args.output_dir)
