import numpy as np
from pylgdt import LGDTPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

dataset = np.genfromtxt("datasets/letter.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LGDTPredictor(
    min_sup=1,
    max_depth=10,
    data_structure="sparse_bitset",
    fit_method="murtree",
    log=True,
)

model.fit(X_train, y_train)

print("Train error : ", model.error_)
print("Train accuracy : ", round(model.accuracy_, 5))
y_pred = model.predict(X_test)
m = confusion_matrix(y_test, y_pred)
print("Test accuracy : ", round(accuracy_score(y_test, y_pred), 5))
print("Test confusion matrix : \n", m)
print()
print("Tree size : ", model.size_)
print("Tree depth : ", model.depth_)
