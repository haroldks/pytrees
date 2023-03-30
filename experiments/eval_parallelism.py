import os
import numpy as np
import pandas as pd
from pytrees.lgdt import LGDTClassifier


# import sys
# import graphviz
# data = np.genfromtxt("data/raw/datasetsDL/german-credit.txt")
# X, y = data[:, 1:], data[:, 0]
# d = 5
# clf = LGDTClassifier(
#     min_sup=5,
#     max_depth=d
# )
# clf.fit(X, y)
# obj = graphviz.Source(clf.export_to_graphviz_dot())
# obj.view(f"tree_{d}")
#
#
# sys.exit(0)
#


data_dir = "data/raw/datasetsDL/"
n_folds = range(5)
min_sup = 5
depths = range(5, 8)
results = []
n_threads = range(0, 9)
for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)
    data = np.genfromtxt(file_path, delimiter=" ")
    file_name = file.split(".")[0]
    X, y = data[:, 1:], data[:, 0]
    for depth in depths:

        for thread in n_threads:
            print(f"Running {file_name} with depth {depth} and {thread} threads")
            runtimes = []
            for fold in n_folds:

                if thread == 0:
                    clf = LGDTClassifier(
                        min_sup=min_sup,
                        max_depth=depth,
                        parallel=False,
                        num_threads=0,
                        data_structure="reversible_sparse_bitset",
                        fit_method="murtree",
                    )
                else:
                    clf = LGDTClassifier(
                        min_sup=min_sup,
                        max_depth=depth,
                        parallel=True,
                        num_threads=thread,
                        data_structure="reversible_sparse_bitset",
                        fit_method="murtree",
                    )
                try:
                    clf.fit(X, y)
                    runtimes.append(clf.statistics["duration_milliseconds"])
                except:
                    continue
            try:
                results.append(
                    {
                        "filename": file_name,
                        "depth": depth,
                        "n_threads": thread,
                        "runtime": np.mean(runtimes),
                    }
                )
            except:
                continue
    # break

df = pd.DataFrame(results)
df.to_csv("parallelism_comparison.csv", index=False)

# df = pd.read_csv("parallelism_comparison.csv")

import matplotlib.pyplot as plt

dataname = df["filename"].unique()
coords = [(0, 0), (0, 1), (0, 2)]
for name in dataname:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i, depth in enumerate(depths):
        a_x, a_y = coords[i]
        sub_df = df[(df.filename == name) & (df.depth == depth)]

        sub_df.plot(
            x="n_threads", y="runtime", ax=axes[a_y], title=f"{name} - depth {depth}"
        )
        axes[a_y].set_xlabel("Number of threads")
        axes[a_y].set_ylabel("Runtime (ms)")
        axes[a_y].grid(True)
        axes[a_y].legend()
    plt.savefig(
        f"para_plots/parallelism_comparison_{name}.pdf", dpi=300, bbox_inches="tight"
    )
    fig.clf()
    # break


#
#
#
#
#
#
# dataset = np.genfromtxt("data/raw/datasetsDL/ionosphere.txt", delimiter=" ")
# X, y = dataset[:, 1:], dataset[:, 0]
#
# normal_classifier = LGDTClassifier(
#     min_sup=1,
#     max_depth=5,
#     parallel=False,
#     num_threads=0,
#     data_structure="reversible_sparse_bitset",
#     fit_method="murtree",
# )
#
# parallel_classifier = LGDTClassifier(
#     min_sup=1,
#     max_depth=5,
#     parallel=True,
#     num_threads=4,
#     data_structure="reversible_sparse_bitset",
#     fit_method="murtree",)
#
#
#
# normal_classifier.fit(X, y)
# print(normal_classifier.statistics)
# parallel_classifier.fit(X, y)
# print(parallel_classifier.statistics)
