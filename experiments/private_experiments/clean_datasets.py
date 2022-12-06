import os
import shutil
import numpy as np


def clean_dataset(path, keep_unique_rows=False):
    dataset = np.genfromtxt(path, delimiter=" ")
    X, y = dataset[:, 1:], dataset[:, 0]

    transactions = list()
    labels = list()
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        x_cls = X[idx]
        y_cls = y[idx]
        if keep_unique_rows:
            x_cls, indexes = np.unique(x_cls, axis=0, return_index=True)
            y_cls = y_cls[indexes]

        transactions.append(x_cls)
        labels.append(y_cls)

    labels = sorted(labels, key=lambda v: len(v))
    transactions = sorted(transactions, key=lambda v: len(v))

    trs = [trs.copy() for trs in transactions]
    lbs = [trs.copy() for trs in labels]

    for transaction in transactions[0]:
        _in_other_class = np.where(np.all(transaction == trs[1], axis=1))[0]
        if len(_in_other_class) == 0:
            continue
        else:
            _inside_class = np.where(np.all(transaction == trs[0], axis=1))[0]
            if len(_in_other_class) > len(_inside_class):

                trs[0] = np.delete(trs[0], _inside_class, axis=0)
                lbs[0] = np.delete(lbs[0], _inside_class, axis=0)
            elif len(_in_other_class) < len(_inside_class):
                trs[1] = np.delete(trs[1], _in_other_class, axis=0)
                lbs[1] = np.delete(lbs[1], _in_other_class, axis=0)
            else:
                if len(transactions[0]) < len(transactions[1]):
                    trs[0] = np.delete(trs[0], _inside_class, axis=0)
                    lbs[0] = np.delete(lbs[0], _inside_class, axis=0)
                else:
                    trs[1] = np.delete(trs[1], _in_other_class, axis=0)
                    lbs[1] = np.delete(lbs[1], _in_other_class, axis=0)

    out_x = np.append(trs[0], trs[1], axis=0)
    out_y = np.append(lbs[0], lbs[1], axis=0)
    return out_x, out_y


if __name__ == "__main__":

    DATA_FOLDERS = [
        "../datasetsNina_reduced",
        "../datasetsDL",
        "../datasetsNina",
        "../datasetsNL",
        "../datasets",
        "../datasetsHu",
    ]

    CLEANED_BASE_DIR = "cleaned"

    if os.path.isdir(CLEANED_BASE_DIR):
        shutil.rmtree((CLEANED_BASE_DIR))

    for folder in DATA_FOLDERS:
        name = folder.split("/")[-1]
        os.makedirs(f"{CLEANED_BASE_DIR}/unique/{name}")
        os.makedirs(f"{CLEANED_BASE_DIR}/non-unique/{name}")

    for folder in DATA_FOLDERS:
        folder_name = folder.split("/")[-1]
        print(f"Inside {folder_name}")
        print(folder)
        files = os.listdir(folder)
        for i, file in enumerate(files):
            print(f"Current percentage {i+1} / {len(files)}", end="\r")
            path = os.path.join(folder, file)
            X, y = clean_dataset(path, keep_unique_rows=False)
            data = np.append(y.reshape(-1, 1), X, axis=1)
            np.savetxt(
                f"{CLEANED_BASE_DIR}/non-unique/{folder_name}/{file}",
                data,
                delimiter=" ",
                fmt="%i",
            )

            X, y = clean_dataset(path, keep_unique_rows=True)
            data = np.append(y.reshape(-1, 1), X, axis=1)
            np.savetxt(
                f"{CLEANED_BASE_DIR}/unique/{folder_name}/{file}",
                data,
                delimiter=" ",
                fmt="%i",
            )
        print()
