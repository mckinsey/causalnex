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
import multiprocessing
import os
import warnings
from collections import defaultdict
from time import time
from typing import List, Tuple

import dataframe_image
import networkx as nx
import numpy as np
import pandas as pd
from IPython.core.display import display
from IPython.display import Image
from sklearn import metrics
from sklearn.model_selection import KFold

from causalnex.evaluation import roc_auc
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.plots import plot_structure
from causalnex.structure import StructureModel

warnings.simplefilter("ignore")


def load_structure_model_and_remove_variable(
    filename: str, var_to_remove: str, default_weight: float = 0.1
):
    """
    Utility function to load a structure model and drop a certain node/variable

    Args:
        filename: DOT file containing the structure model
        var_to_remove: Variable to remove
        default_weight: Default edge weight
    """
    # Load model
    raw_sm = nx.drawing.nx_pydot.read_dot(filename)

    # Convert edge weights from string to float
    sm = nx.DiGraph()
    sm.add_weighted_edges_from(
        [
            (u, v, float(w) if w is not None else default_weight)
            for u, v, w in raw_sm.edges(data="weight")
        ]
    )
    # These are the edges in the latent variable 'G1'.
    # We remove G1's edges and connect each parent to all children
    edges_to_add = [edge for edge in sm.edges() if var_to_remove in edge]
    parents = [u for u, v in sm.edges() if v == var_to_remove]
    children = [v for u, v in sm.edges() if u == var_to_remove]
    edges_to_remove = [(p, c) for p in parents for c in children]

    # Remove latent variable and connect all parents to all children
    sm.remove_node(var_to_remove)
    sm.add_edges_from(edges_to_remove, weight=default_weight)

    return sm, edges_to_add, edges_to_remove


def plot_pretty_structure(
    g: StructureModel,
    edges_to_highlight: Tuple[str, str],
    default_weight: float = 0.2,
    weighted: bool = False,
):
    """
    Utility function to plot our networks in a pretty format

    Args:
        g: Structure model (directed acyclic graph)
        edges_to_highlight: List of edges to highlight in the plots
        default_weight: Default edge weight
        weighted: Whether the graph is weighted

    Returns:
        a styled pygraphgiz graph that can be rendered as an image
    """
    graph_attributes = {
        "splines": "spline",  # I use splies so that we have no overlap
        "ordering": "out",
        "ratio": "fill",  # This is necessary to control the size of the image
        "size": "16,9!",  # Set the size of the final image. (this is a typical presentation size)
        "fontcolor": "#FFFFFFD9",
        "fontname": "Helvetica",
        "fontsize": 24,
        "labeljust": "c",
        "labelloc": "c",
        "pad": "1,1",
        "nodesep": 0.8,
        "ranksep": ".5 equally",
    }
    # Making all nodes hexagonal with black coloring
    node_attributes = {
        node: {
            "shape": "hexagon",
            "width": 2.2,
            "height": 2,
            "fillcolor": "#000000",
            "penwidth": "10",
            "color": "#4a90e2d9",
            "fontsize": 24,
            "labelloc": "c",
            "labeljust": "c",
        }
        for node in g.nodes
    }
    # Customising edges
    if weighted:
        edge_weights = [
            (u, v, w if w else default_weight) for u, v, w in g.edges(data="weight")
        ]
    else:
        edge_weights = [(u, v, default_weight) for u, v in g.edges()]

    edge_attributes = {
        (u, v): {
            "penwidth": w * 20 + 2,  # Setting edge thickness
            "weight": int(w),  # Higher "weight"s mean shorter edges
            "arrowsize": 2 - 2.0 * w,  # Avoid too large arrows
            "arrowtail": "dot",
            "color": "#DF5F00" if ((u, v) in set(edges_to_highlight)) else "#888888",
        }
        for u, v, w in edge_weights
    }
    return plot_structure(
        g,
        prog="dot",
        graph_attributes=graph_attributes,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )


def display_colored_df(df: pd.DataFrame):
    """
    Utility function to display dataframe as a heatmap

    Args:
        df: Input dataframe
    """

    def display_df_as_img(df):
        temp_file_name = "./temp.png"
        dataframe_image.export(df, temp_file_name)

        with open(temp_file_name, "rb") as file:
            display(Image(file.read()))

        os.remove(temp_file_name)

    if not isinstance(df.columns, pd.MultiIndex) or len(df.columns.names) != 1:
        display_df_as_img(df.style.background_gradient(axis=None))
    else:
        df = df.copy()
        df.columns = df.columns.levels[0]
        display_df_as_img(df.style.background_gradient(axis=None))


def adjacency_correlations(
    data: pd.DataFrame,
    bn: BayesianNetwork,
    method: str,
) -> pd.DataFrame:
    """
    Utility function to compute pairwise correlation of connected nodes in a graph

    Args:
        data: Input dataframe
        bn: Bayesian network
        method: Correlation scoring method - {‘pearson’, ‘kendall’, ‘spearman’} or callable

    Returns:
        Correlation dataframe
    """
    correl_df = data[bn.nodes].corr(method=method)
    edge_set = set(bn.structure.edges())

    for row in correl_df.index:
        for col in correl_df.columns:
            if (row, col) not in edge_set:
                correl_df.loc[row, col] = np.nan

    return correl_df


def predict_using_all_nodes(
    bn: BayesianNetwork,
    data: pd.DataFrame,
    target_var: str,
    markov_blanket: bool = False,
    lv_name: str = "LV",
) -> pd.DataFrame:
    """
    Compute marginals using all nodes

    Args:
        bn: Bayesian network
        data: Input dataframe
        target_var: Target variable name
        markov_blanket: Whether to compute marginals based only on Markov blanket of the target variable
        lv_name: Latent variable name

    Returns:
        Marginal dataframe
    """
    # Extract columns of interest
    if markov_blanket:
        blanket = bn.structure.get_markov_blanket([target_var, lv_name])
        cols_to_keep = blanket.nodes
    else:
        cols_to_keep = bn.nodes

    # Further drop target variable and latent variable (if applicable)
    cols_to_keep = [col for col in cols_to_keep if col not in {target_var, lv_name}]

    # Perform inference
    ie = InferenceEngine(bn)
    observations = data[cols_to_keep].to_dict(orient="records")
    marginals = [prob[target_var] for prob in ie.query(observations)]
    return pd.DataFrame(marginals)


def _build_ground_truth(
    bn: BayesianNetwork,
    data: pd.DataFrame,
    node: str,
) -> pd.DataFrame:
    """
    Utility function to build ground truth from data

    Args:
        bn: Bayesian network
        data: Input dataframe
        node: Node name

    Returns:
        Ground truth dataframe
    """
    ground_truth = pd.get_dummies(data[node])

    # it's possible that not all states are present in the test set, so we need to add them to ground truth
    for dummy in bn.node_states[node]:
        if dummy not in ground_truth.columns:
            ground_truth[dummy] = [0 for _ in range(len(ground_truth))]

    # update ground truth column names to be correct, since we may have added missing columns
    return ground_truth[sorted(ground_truth.columns)]


def get_avg_auc(
    df: pd.DataFrame,
    bn: BayesianNetwork,
    n_splits: int = 5,
    seed: int = 2021,
) -> float:
    """
    Estimate the average auc of all nodes in a Bayesian Network given a structure and a dataset using
    k-fold cross-validation. This function uses the bn.predict method in causalnex and cannot be used
    with latent variable models

    Args:
        df: a dataset in the pandas format
        bn: a bayesian network EM object
        n_splits: Number of folds in k-fold cv
        seed: random seed used in k-fold cv

    Returns:
        Average AUC
    """
    bn.fit_node_states(df)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    total_auc = 0

    for fold, (train_idx, test_idx) in enumerate(cv.split(df)):
        t0 = time()
        cur_auc = 0
        train_df = df.loc[train_idx, :]
        test_df = df.loc[test_idx, :]
        bn.fit_cpds(train_df, method="BayesianEstimator", bayes_prior="K2")

        for var in bn.nodes:
            _, auc = roc_auc(bn, test_df, var)
            cur_auc += auc

        print(f"Processing fold {fold} takes {time() - t0} seconds")
        total_auc += cur_auc / len(bn.nodes)

    return total_auc / n_splits


def get_auc_data(
    df: pd.DataFrame,
    bn: BayesianNetwork,
    n_splits: int = 5,
    seed: int = 2021,
) -> pd.Series:
    """
    Utility function to compute AUC based only on data observations

    Args:
        df: Input dataframe
        bn: Bayesian network
        n_splits: Number of cross-validation folds
        seed: Random seed number

    Returns:
        Average AUC
    """
    bn.fit_node_states(df)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    nodes_auc = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(cv.split(df)):
        t0 = time()
        train_df = df.loc[train_idx, :]
        test_df = df.loc[test_idx, :]
        bn.fit_cpds(train_df, method="BayesianEstimator", bayes_prior="K2")

        for var in bn.nodes:
            _, auc = roc_auc(bn, test_df, var)
            nodes_auc[var].append(auc)

        print(f"Processing fold {fold} takes {time() - t0} seconds")

    nodes_auc = pd.DataFrame(nodes_auc)
    col = nodes_auc.mean(axis=0).idxmin()
    val = nodes_auc.mean(axis=0).min()
    print(f"Variable with lowest AUC is {col} with the value of {val}")
    return nodes_auc.mean().sort_values()


def roc_auc_lv(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Utility function to compute area under the curve (AUC)

    Args:
        ground_truth: Ground truth dataframe
        predictions: Prediction dataframe

    Returns:
        ROC curve and AUC
    """
    predictions = predictions[sorted(predictions.columns)]

    fpr, tpr, _ = metrics.roc_curve(
        ground_truth.values.ravel(), predictions.values.ravel()
    )
    roc = list(zip(fpr, tpr))
    auc = metrics.auc(fpr, tpr)
    return roc, auc


def get_avg_auc_lvs(
    df: pd.DataFrame,
    bn: BayesianNetwork,
    lv_states: List,
    n_splits: int = 5,
    seed: int = 2021,
    markov_blanket: bool = False,
    n_cpus: int = multiprocessing.cpu_count() - 1,
) -> float:
    """
    Utility function to compute AUC using only the parent nodes

    Args:
        df: Input dataframe
        bn: Bayesian network
        lv_states: the states the LV can assume
        n_splits: Number of cross-validation folds
        seed: Random seed number
        markov_blanket: Whether we predict only using the Markov blanket
        n_cpus: Number of CPU cores to use

    Returns:
        Average AUC
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    total_auc = 0

    for fold, (train_idx, test_idx) in enumerate(cv.split(df)):
        t0 = time()
        train_df = df.loc[train_idx, :]
        test_df = df.loc[test_idx, :]
        bn.fit_latent_cpds("LV", lv_states, train_df, n_runs=30)
        chunks = [
            [bn, test_df, target, markov_blanket]
            for target in bn.nodes
            if target != "LV"
        ]
        with multiprocessing.Pool(n_cpus) as p:
            result = p.starmap(_compute_auc_lv_stub, chunks)

        total_auc += sum(result) / (len(bn.nodes) - 1)
        print(
            f"Processing fold {fold} using {n_cpus} cores takes {time() - t0} seconds"
        )

    return total_auc / n_splits


def get_avg_auc_all_info(
    df: pd.DataFrame,
    bn: BayesianNetwork,
    n_splits: int = 5,
    seed: int = 2021,
    n_cpus: int = multiprocessing.cpu_count() - 1,
) -> float:
    """
    Utility function to compute AUC using all nodes beyond the parent nodes

    Args:
        df: Input dataframe
        bn: Bayesian network
        n_splits: Number of cross-validation folds
        seed: Random seed number
        n_cpus: Number of CPU cores to use

    Returns:
        Average AUC
    """
    bn.fit_node_states(df)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    total_auc = 0

    for fold, (train_idx, test_idx) in enumerate(cv.split(df)):
        t0 = time()
        train_df = df.loc[train_idx, :]
        test_df = df.loc[test_idx, :]
        bn.fit_cpds(train_df, method="BayesianEstimator", bayes_prior="K2")
        chunks = [[bn, test_df, target] for target in bn.nodes]

        with multiprocessing.Pool(n_cpus) as p:
            result = p.starmap(_compute_auc_stub, chunks)

        total_auc += sum(result) / len(bn.nodes)
        print(
            f"Processing fold {fold} using {n_cpus} cores takes {time() - t0} seconds"
        )

    return total_auc / n_splits


def _compute_auc_stub(
    bn: BayesianNetwork,
    data: pd.DataFrame,
    var: str,
) -> float:
    """
    Helper function to compute AUC using all available information

    Args:
        bn: Bayesian network
        data: Input dataframe
        var: Target variable name

    Returns:
        Area under the curve (AUC)
    """
    y_true = _build_ground_truth(bn, data, var)
    y_pred = predict_using_all_nodes(bn, data, var, lv_name=None)
    _, auc = roc_auc_lv(y_true, y_pred)
    return auc


def _compute_auc_lv_stub(
    bn: BayesianNetwork,
    data: pd.DataFrame,
    var: str,
    markov_blanket: bool,
) -> float:
    """
    Helper function to compute AUC excluding the latent variable

    Args:
        bn: Bayesian network
        data: Input dataframe
        var: Target variable name

    Returns:
        Area under the curve (AUC)
    """
    y_true = _build_ground_truth(bn, data, var)
    y_pred = predict_using_all_nodes(
        bn, data, var, markov_blanket=markov_blanket, lv_name="LV"
    )
    _, auc = roc_auc_lv(y_true, y_pred)
    return auc
