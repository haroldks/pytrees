from pytrees.lgdt import LGDTClassifier
import numpy as np

clf = LGDTClassifier(
    min_sup=1,
    max_depth=2,
)
dataset = np.genfromtxt("../test_data/anneal.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]
clf.fit(X, y.astype(float))
print(f"Train accuracy : {clf.tree_error_}")
print(f"Statistics : {clf.statistics}")
