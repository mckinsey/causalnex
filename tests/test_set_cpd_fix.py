import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from causalnex.structure.notears import from_pandas
from causalnex.network import BayesianNetwork
from causalnex.discretiser import Discretiser
from sklearn.model_selection import train_test_split

data = pd.read_csv("student-por.csv", delimiter=";")[
    ["famrel", "higher", "freetime", "G1", "paid"]
]
struct_data = data.copy()
non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
le = LabelEncoder()
for col in non_numeric_columns:
    struct_data[col] = le.fit_transform(struct_data[col])
sm = from_pandas(struct_data, w_threshold=0.8)

bn = BayesianNetwork(sm)

discretised_data = data.copy()
data_vals = {col: data[col].unique() for col in data.columns}
discretised_data["G1"] = Discretiser(
    method="fixed", numeric_split_points=[10]
).transform(discretised_data["G1"].values)
G1_map = {0: "Fail", 1: "Pass"}
discretised_data["G1"] = discretised_data["G1"].map(G1_map)

train, test = train_test_split(
    discretised_data, train_size=0.9, test_size=0.1, random_state=7
)
bn = bn.fit_node_states(discretised_data)
bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
bn.set_cpd(
    "G1", bn.cpds["G1"]
) 
bn.set_cpd(
    "paid", bn.cpds["paid"]
)