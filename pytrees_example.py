import numpy as np
from pytrees.optimal import DL85Classifier
from pytrees.lgdt import LGDTClassifier

dataset = np.genfromtxt("test_data/anneal.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

clf = DL85Classifier(min_sup=1, max_depth=4)
# clf.fit(X, y)

print(clf.accuracy_)
print(clf.statistics)

clf = LGDTClassifier(min_sup=1, max_depth=3)
clf.fit(X, y)
print(clf.accuracy_)
print(clf.statistics)
