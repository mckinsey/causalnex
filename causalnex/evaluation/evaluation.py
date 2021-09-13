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
"""Evaluation metrics for causal models."""

from typing import Dict, List, Tuple

import pandas as pd
from sklearn import metrics

from causalnex.network import BayesianNetwork


def _build_ground_truth(
    bn: BayesianNetwork, data: pd.DataFrame, node: str
) -> pd.DataFrame:

    ground_truth = pd.get_dummies(data[node])

    # it's possible that not all states are present in the test set, so we need to add them to ground truth
    for dummy in bn.node_states[node]:
        if dummy not in ground_truth.columns:
            ground_truth[dummy] = [0 for _ in range(len(ground_truth))]

    # update ground truth column names to be correct, since we may have added missing columns
    return ground_truth[sorted(ground_truth.columns)]


def roc_auc(
    bn: BayesianNetwork, data: pd.DataFrame, node: str
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Build a report of the micro-average Receiver-Operating Characteristics (ROC), and the Area Under the ROC curve
    Micro-average computes roc_auc over all predictions for all states of node.

    Args:
        bn (BayesianNetwork): model to compute roc_auc.
        data (pd.DataFrame): test data that will be used to calculate ROC.
        node (str): name of the variable to generate the report for.

    Returns:
        roc - auc tuple
         - roc (List[Tuple[float, float]]): list of [(fpr, tpr)] observations.
         - auc float: auc for the node predictions.

    Example:
    ::
        >>> from causalnex.structure import StructureModel
        >>> from causalnex.network import BayesianNetwork
        >>>
        >>> sm = StructureModel()
        >>> sm.add_edges_from([
        >>>                    ('rush_hour', 'traffic'),
        >>>                    ('weather', 'traffic')
        >>>                    ])
        >>> bn = BayesianNetwork(sm)
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        >>>                      'rush_hour': [True, False, False, False, True, False, True],
        >>>                      'weather': ['Terrible', 'Good', 'Bad', 'Good', 'Bad', 'Bad', 'Good'],
        >>>                      'traffic': ['heavy', 'light', 'heavy', 'light', 'heavy', 'heavy', 'heavy']
        >>>                      }
        >>> bn = bn.fit_node_states_and_cpds(data)
        >>> test_data = pd.DataFrame({
        >>>                         'rush_hour': [False, False, True, True],
        >>>                         'weather': ['Good', 'Bad', 'Good', 'Bad'],
        >>>                         'traffic': ['light', 'heavy', 'heavy', 'light']
        >>>                         })
        >>> from causalnex.evaluation import roc_auc
        >>> roc, auc = roc_auc(bn, test_data, "traffic")
        >>> print(auc)
        0.75
    """

    ground_truth = _build_ground_truth(bn, data, node)
    predictions = bn.predict_probability(data, node)

    # update column names to match those of ground_truth
    predictions.rename(columns=lambda x: x.lstrip(node + "_"), inplace=True)
    predictions = predictions[sorted(predictions.columns)]

    fpr, tpr, _ = metrics.roc_curve(
        ground_truth.values.ravel(), predictions.values.ravel()
    )
    roc = list(zip(fpr, tpr))
    auc = metrics.auc(fpr, tpr)

    return roc, auc


def classification_report(bn: BayesianNetwork, data: pd.DataFrame, node: str) -> Dict:
    """
    Build a report showing the main classification metrics.

    Args:
        bn (BayesianNetwork): model to compute classification report using.
        data (pd.DataFrame): test data that will be used for predictions.
        node (str): name of the variable to generate report for.

    Returns:
        Text summary of the precision, recall, F1 score for each class.

        The reported averages include micro average (averaging the
        total true positives, false negatives and false positives), macro
        average (averaging the unweighted mean per label), weighted average
        (averaging the support-weighted mean per label) and sample average
        (only for multilabel classification).

        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".

    Example:
    ::
        >>> from causalnex.structure import StructureModel
        >>> from causalnex.network import BayesianNetwork
        >>>
        >>> sm = StructureModel()
        >>> sm.add_edges_from([
        >>>                    ('rush_hour', 'traffic'),
        >>>                    ('weather', 'traffic')
        >>>                    ])
        >>> bn = BayesianNetwork(sm)
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        >>>                      'rush_hour': [True, False, False, False, True, False, True],
        >>>                      'weather': ['Terrible', 'Good', 'Bad', 'Good', 'Bad', 'Bad', 'Good'],
        >>>                      'traffic': ['heavy', 'light', 'heavy', 'light', 'heavy', 'heavy', 'heavy']
        >>>                      }
        >>> bn = bn.fit_node_states_and_cpds(data)
        >>> test_data = pd.DataFrame({
        >>>                         'rush_hour': [False, False, True, True],
        >>>                         'weather': ['Good', 'Bad', 'Good', 'Bad'],
        >>>                         'traffic': ['light', 'heavy', 'heavy', 'light']
        >>>                         })
        >>> from causalnex.evaluation import classification_report
        >>> classification_report(bn, test_data, "traffic")
        {'precision': {
            'macro avg': 0.8333333333333333, 'micro avg': 0.75,
            'traffic_heavy': 0.6666666666666666,
            'traffic_light': 1.0,
            'weighted avg': 0.8333333333333333
          },
         'recall': {
            'macro avg': 0.75,
            'micro avg': 0.75,
            'traffic_heavy': 1.0,
            'traffic_light': 0.5,
            'weighted avg': 0.75
          },
         'f1-score': {
            'macro avg': 0.7333333333333334,
            'micro avg': 0.75,
            'traffic_heavy': 0.8,
            'traffic_light': 0.6666666666666666,
            'weighted avg': 0.7333333333333334
          },
         'support': {
            'macro avg': 4,
            'micro avg': 4,
            'traffic_heavy': 2,
            'traffic_light': 2,
            'weighted avg': 4
          }}
    """

    predictions = bn.predict(data, node)

    labels = sorted(list(bn.node_states[node]))
    target_names = [f"{node}_{v}" for v in sorted(bn.node_states[node])]
    report = metrics.classification_report(
        y_true=data[node],
        y_pred=predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
    )

    return report
