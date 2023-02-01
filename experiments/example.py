from pytrees.lgdt import LGDTClassifier
import numpy as np
from sklearn.model_selection import train_test_split

dataset = np.genfromtxt("data/raw/datasetsDL/anneal.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = LGDTClassifier(
    min_sup=5,
    max_depth=7,
    fit_method="murtree",
    data_structure="reversible_sparse_bitset",
)

classifier.fit(X_train, y_train)

print(f"Train accuracy : {classifier.accuracy_}")
print(f"Test accuracy : {classifier.score(X_test, y_test)}")
