from pytrees.lgdt import LGDTClassifier
import numpy as np
from sklearn.model_selection import train_test_split

dataset = np.genfromtxt("data/raw/datasetsDL/anneal.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# classifier = LGDTClassifier(
#     min_sup=5,
#     max_depth=7,
#     fit_method="murtree",
#     data_structure="reversible_sparse_bitset",
# )
#
# classifier.fit(X_train, y_train)
#
# print(f"Train accuracy : {classifier.accuracy_}")
# print(f"Test accuracy : {classifier.score(X_test, y_test)}")

#
from pydl85 import DL85Classifier

clf = DL85Classifier(
    min_sup=1,
    max_depth=3,
    depth_two_special_algo=False,
    similar_lb=False,
    dynamic_branch=True,
    similar_for_branching=False,
    print_output=True,
)
clf.fit(X, y)
print(f"Train accuracy : {clf.error_}")
print(f"Runtime : {clf.runtime_}")


# from pytrees.optimal import DL85Classifier
# clf = DL85Classifier(min_sup=1, max_depth=4, specialization="none", lower_bound="none")
# clf.fit(X, y)
# print(f"Train accuracy : {clf.accuracy_}")
# print(f"Runtime : {clf.statistics}")
