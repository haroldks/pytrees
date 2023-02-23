from pytrees.lgdt import LGDTClassifier
import numpy as np
from sklearn.model_selection import train_test_split

from pytrees.optimal.lds_dl85 import LDSDL85Classifier
from pytrees.predictor import (
    Specialization,
    LowerBound,
    Branching,
    CacheInit,
    DiscrepancyStrategy,
)

dataset = np.genfromtxt("data/raw/datasetsDL/splice-1.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]


limit = 5

from graphviz import Source
from pydl85 import DL85Classifier

clf = DL85Classifier(
    min_sup=1,
    max_depth=4,
    time_limit=limit,
    print_output=True,
)


clf.fit(X, y)
print(f"Train accuracy : {clf.error_}")
print(f"Runtime : {clf.runtime_}")
print(clf.export_graphviz())

obj = Source(clf.export_graphviz())
obj.view(filename="dl85.pdf")


from pytrees.optimal import DL85Classifier


# clf = DL85Classifier(
#     min_sup=1,
#     max_depth=3,
#     specialization=Specialization.None_,
#     lower_bound=LowerBound.None_,
#     branching=Branching.None_,
#     cache_init=CacheInit.Dynamic,
# )

clf = LDSDL85Classifier(
    min_sup=1,
    max_depth=4,
    max_time=limit,
    discrepancy_strategy=DiscrepancyStrategy.Double,
    specialization=Specialization.None_,
    lower_bound=LowerBound.None_,
    branching=Branching.None_,
    cache_init=CacheInit.Dynamic,
)

clf.fit(X, y)
print()
print()
print(f"Train accuracy : {clf.accuracy_}")
print(f"Statistics : {clf.statistics}")
# print(clf.tree_)
print(clf.export_to_graphviz_dot())
a = Source(clf.export_to_graphviz_dot())
a.view(filename="lds.pdf")


# print(clf.get_dot_body())
