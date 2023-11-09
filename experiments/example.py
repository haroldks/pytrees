import pandas as pd

from pytrees.optimal import DL85Classifier
import numpy as np
from sklearn.model_selection import train_test_split

# from pytrees.optimal.lds_dl85 import LDSDL85Classifier
from pytrees import (
    Specialization,
    LowerBound,
    Branching,
    CacheInit,
    DiscrepancyStrategy,
)

import sys

# from pytrees.optimal import DL85Classifier
# dataset = np.genfromtxt("data/raw/datasetsDL/splice-1.txt", delimiter=" ")
# X, y = dataset[:, 1:], dataset[:, 0]
# clf = DL85Classifier(min_sup=1, max_depth=2)
# clf.fit(X, y)
# print(f"Train accuracy : {clf.accuracy_}")
# print(f"Errror : {clf.tree_error_}")
# print(clf.tree_)
# sys.exit(0)

#
# from sklearn.datasets import load_iris
#
# X, y = load_iris(return_X_y=True)
#
# from binarizer import LessGreatBinarizer
#
# binarizer = LessGreatBinarizer()
#
# df = pd.DataFrame(X)
# new_df = binarizer.fit(df, continuous=True, n_bins=12)
#
# X_bin = new_df.values
# print(X_bin.shape)

# from pydl85 import DL85Classifier
from pytrees.lgdt import LGDTClassifier
from graphviz import Source


def miss_class_error(vector, X=2):
    return np.sum(vector) - np.max(vector), np.argmax(vector)


# print(miss_class_error([32, 34, 45]))

clf = DL85Classifier(
    min_sup=1,
    max_depth=4,
    custom_function=miss_class_error,
    specialization=Specialization.None_,
)
dataset = np.genfromtxt("../test_data/anneal.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]
clf.fit(X, y.astype(float))
print(f"Train accuracy : {clf.tree_error_}")
print(f"Statistics : {clf.statistics}")
obj = Source(clf.export_to_graphviz_dot())
obj.view(filename="iris_lgdt")

# from pytrees.optimal import DL85Classifier
# clf_2 = DL85Classifier(min_sup=1, max_depth=4, specialization=Specialization.None_, lower_bound=LowerBound.None_)
# clf_2.fit(X_bin, y.astype(float))
# print(f"Train accuracy : {clf_2.tree_}")
#
# print(clf_2.tree_)
#
# obj_2 = Source(clf_2.export_to_graphviz_dot())
# obj_2.view(filename="iris_g_2")
#
# vals = np.append(y.reshape(-1, 1), X_bin, axis=1)
# np.savetxt("iris_multi_class.txt", vals, delimiter=" ", fmt="%d")
#

#
# dataset = np.genfromtxt("data/raw/datasetsDL/splice-1.txt", delimiter=" ")
# X, y = dataset[:, 1:], dataset[:, 0]
# #
# #
# # limit = 5
# #
# from graphviz import Source
# # from pydl85 import DL85Classifier
# #
# # clf = DL85Classifier(
# #     min_sup=1,
# #     max_depth=4,
# #     time_limit=limit,
# #     print_output=True,
# # )
# #
# #
# # clf.fit(X, y)
# # print(f"Train accuracy : {clf.error_}")
# # print(f"Runtime : {clf.runtime_}")
# # print(clf.export_graphviz())
# #
# # obj = Source(clf.export_graphviz())
# # obj.view(filename="dl85.pdf")
#
#
# from pytrees.optimal import DL85Classifier
#
#
# # clf = DL85Classifier(
# #     min_sup=1,
# #     max_depth=3,
# #     specialization=Specialization.None_,
# #     lower_bound=LowerBound.None_,
# #     branching=Branching.None_,
# #     cache_init=CacheInit.Dynamic,
# # )
#
# clf = LGDTClassifier(
#     min_sup=1,
#     max_depth=4,
# )
#
#
# clf.fit(X, y)
#
# print()
# print()
# print(f"Train accuracy : {clf.accuracy_}")
# print(f"Statistics : {clf.statistics}")
# # print(clf.tree_)
# print(clf.export_to_graphviz_dot())
# a = Source(clf.export_to_graphviz_dot())
# a.view(filename="lds.pdf")
# clf.max_depth = 7
# clf.fit(X, y)
#
# print()
# print()
# print(f"Train accuracy : {clf.accuracy_}")
# print(f"Statistics : {clf.statistics}")
# # print(clf.tree_)
# print(clf.export_to_graphviz_dot())
# a = Source(clf.export_to_graphviz_dot())
# a.view(filename="ldsx.pdf")
# #
# #
# # # print(clf.get_dot_body())
