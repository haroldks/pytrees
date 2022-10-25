import os

import numpy as np
import pandas as pd
from pylgdt import LGDTPredictor, IDKPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.exceptions import NotFittedError
import logging

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

DATA_FOLDERS = [
    "datasetsNina_reduced",
    "datasetsDL",
    "datasetsNina",
    "datasetsNL",
    "datasetsHu",
]
results = []

for folder in DATA_FOLDERS:

    for file in os.listdir(folder):
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
        print(f"Currently on {name}\n")

        path = os.path.join(folder, file)
        dataset = np.genfromtxt(path, delimiter=" ")
        X, y = dataset[:, 1:], dataset[:, 0]
        idk_infogain = IDKPredictor(
            min_sup=5,
            data_structure="sparse_bitset",
            fit_method="infogain",
            log=True,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        idk_murtree = IDKPredictor(
            min_sup=5,
            data_structure="sparse_bitset",
            fit_method="murtree",
            log=True,
        )
        try:
            idk_murtree.fit(X_train, y_train)
            idk_infogain.fit(X_train, y_train)

            y_pred_mur = idk_murtree.predict(X_test)
            y_pred_ig = idk_infogain.predict(X_test)

            infos = {
                "name": name,
                "train_error_ig": idk_infogain.error_,
                "train_error_mur": idk_murtree.error_,
                "train_acc_ig": round(idk_infogain.accuracy_, 3),
                "train_acc_mur": round(idk_murtree.accuracy_, 3),
                "test_acc_ig": round(accuracy_score(y_test, y_pred_ig), 5),
                "test_acc_mur": round(accuracy_score(y_test, y_pred_mur), 5),
            }
            results.append(infos)
        except NotFittedError as e:
            print(f"Failed to fit for {name}")
        except Exception as e:
            print("Investigation need")

df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

df["count_mur"] = df["test_acc_mur"] >= df["test_acc_ig"]
df["count_ig"] = df["test_acc_ig"] >= df["test_acc_mur"]
print(f'Winning MUR on {df["count_mur"].sum()}/{len(df)}')
print(f'Winning IG on {df["count_ig"].sum()}/{len(df)}')
#
#
# dataset = np.genfromtxt("datasets/pendigits.txt", delimiter=" ")
# X, y = dataset[:, 1:], dataset[:, 0]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# lgdt_model = IDKPredictor(
#     min_sup=1, data_structure="sparse_bitset", fit_method="murtree", log=True,
# )
#
# idk_model = IDKPredictor(
#     min_sup=1, data_structure="sparse_bitset", fit_method="infogain", log=True,
# )
#
# lgdt_model.fit(X_train, y_train)
# idk_model.fit(X_train, y_train)
#
# print("LGDT Train error : ", lgdt_model.error_)
# print("IDK Train error : ", idk_model.error_)
# print("LGDT Train accuracy : ", round(lgdt_model.accuracy_, 5))
# print("IDK Train accuracy : ", round(idk_model.accuracy_, 5))
# y_pred = lgdt_model.predict(X_test)
# y_pred_idk = idk_model.predict(X_test)
# m = confusion_matrix(y_test, y_pred)
# print("LGDT Test accuracy : ", round(accuracy_score(y_test, y_pred), 5))
# print("IDK Test accuracy : ", round(accuracy_score(y_test, y_pred_idk), 5))
# print("Test confusion matrix : \n", m)
# print()
# # print("Tree size : ", model.size_)
# # print("Tree depth : ", model.depth_)
