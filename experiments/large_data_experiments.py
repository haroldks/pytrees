import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from pydl85 import DL85Classifier
from pytrees.lgdt import LGDTClassifier

MIN_SUP = 1
DEPTHS = range(2, 5)

data_folder = "../data/parallel_datasets"
to = 30

models = ["lgdt_error_sparse", "lgdt_ig", "c4.5", "dl8.5"]

# Append to a csv file and save
def append_to_csv(data, path):
    # print(data)
    df = pd.DataFrame(data)
    if os.path.isfile(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


results = list()
for depth in DEPTHS:
    for file in os.listdir(data_folder):
        name = file.split(".")[0]
        if name in [
            "small",
            "small_",
            "rsparse_dataset",
            "tic-tac-toe__",
            "tic-tac-toe_",
            "appendicitis-un-reduced_converted",
        ]:
            continue

        path = os.path.join(data_folder, file)

        data = np.genfromtxt(path, delimiter=" ")
        X, y = data[:, 1:], data[:, 0]
        infos = {"name": name, "depth": depth}
        for model in models:
            print(f"Running {model} on {name} with depth {depth}")
            if model == "lgdt_error_sparse":
                instance = LGDTClassifier(
                    min_sup=MIN_SUP,
                    max_depth=depth,
                    data_structure="reversible_sparse_bitset",
                    fit_method="murtree",
                )
            elif model == "lgdt_ig":
                instance = LGDTClassifier(
                    min_sup=MIN_SUP,
                    max_depth=depth,
                    data_structure="reversible_sparse_bitset",
                    fit_method="info_gain",
                )
            elif model == "c4.5":
                print("c4.5")
                instance = DecisionTreeClassifier(
                    criterion="log_loss",
                    splitter="best",
                    max_depth=depth,
                    min_samples_split=MIN_SUP,
                )
            elif model == "dl8.5":
                instance = DL85Classifier(
                    min_sup=MIN_SUP,
                    max_depth=depth,
                    time_limit=to,
                )
            instance.fit(X, y)

            new = {
                "name": name,
                f"{model}_train_acc": np.round(instance.score(X, y), 5),
            }
            infos = {**infos, **new}
        append_to_csv([infos], f"large_expe_runtimes_and_errors_v2_{to}s.csv")
