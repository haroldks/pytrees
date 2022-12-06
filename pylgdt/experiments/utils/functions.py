import os
import random
import time
import sys
import uuid
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit

MAIN_COLS = ["name", "depth", "noise_level"]


def run_on_single_test_set(
    datapath,
    models,
    min_sup=5,
    depths=range(2, 3),
    val_size=0.2,
    noise_levels=None,
    n_folds=5,
):

    if noise_levels is None:
        noise_levels = [0]

    dataset = np.genfromtxt(datapath, delimiter=" ")
    X, y = dataset[:, 1:], dataset[:, 0]
    results = list()

    print("|", datapath)
    for fold in range(n_folds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=val_size, random_state=0
        )
        for level in noise_levels:
            noisy_y_train = add_noise_numpy(y_train.copy(), percentage=level)

            for depth in depths:
                out = dict()
                dataset_name = {
                    "name": datapath.split("/")[-1][0:-4],
                    "noise_level": np.round(level, decimals=3),
                    "depth": depth,
                }
                for model in models:
                    instance = model["instance"]
                    if model["name"] in ["cart"]:
                        instance.max_depth = depth
                        instance.min_samples_split = min_sup
                    else:
                        instance.max_depth = depth
                        instance.min_sup = min_sup
                    try:
                        start = time.time()
                        instance.fit(X_train, noisy_y_train)
                        duration = time.time() - start
                        if model["name"] in ["cart"]:
                            train_acc = {
                                f'{model["name"]}_train_acc': np.round(
                                    instance.score(X_train, noisy_y_train), decimals=3
                                ),
                                f'{model["name"]}_runtime': duration,
                            }
                        else:
                            train_acc = {
                                f'{model["name"]}_train_acc': np.round(
                                    instance.accuracy_, decimals=3
                                ),
                                f'{model["name"]}_runtime': duration,
                            }
                        out = {**out, **train_acc}
                        y_pred = instance.predict(X_test)
                        test_acc = {
                            f'{model["name"]}_test_acc': np.round(
                                accuracy_score(y_test, y_pred), decimals=3
                            )
                        }
                        out = {**out, **test_acc}
                    except Exception as e:
                        print(
                            f'File : {datapath} failed for depth = {depth} and model {model["name"]}'
                        )
                        print(f"Error : {e}")
                out = {**dataset_name, **out}
                results.append(out)
    return pd.DataFrame(results)


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)

    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def run_multithread_on_single_test_set(
    datapath,
    models,
    min_sup=5,
    depths=range(2, 3),
    val_size=0.2,
    noise_levels=None,
    n_folds=5,
    n_threads=8,
):
    if noise_levels is None:
        noise_levels = [0]

    dataset = np.genfromtxt(datapath, delimiter=" ")
    X, y = dataset[:, 1:], dataset[:, 0]

    @globalize
    def exec_noise(level, data, f):
        X_train_, X_test_, y_train_, y_test_ = data
        ans = list()
        noisy_y_train = add_noise_numpy(y_train_.copy(), percentage=level)
        for depth in depths:
            out = dict()
            dataset_name = {
                "name": datapath.split("/")[-1][0:-4],
                "noise_level": np.round(level, decimals=3),
                "depth": depth,
            }
            for model in models:
                # print("|\t|", f'{depth}, {model["name"]} {level}, fold {f}')
                instance = model["instance"]
                if model["name"] in ["cart"]:
                    instance.max_depth = depth
                    instance.min_samples_split = min_sup
                elif "bagged" in model["name"]:
                    instance.base_estimator.max_depth = depth
                    instance.base_estimator.min_sup = min_sup
                else:
                    instance.max_depth = depth
                    instance.min_sup = min_sup
                try:
                    start = time.time()
                    instance.fit(X_train_, noisy_y_train)
                    duration = time.time() - start
                    if model["name"] in ["cart"] or "bagged" in model["name"]:
                        train_acc = {
                            f'{model["name"]}_train_acc': np.round(
                                instance.score(X_train_, noisy_y_train), decimals=4
                            ),
                            # f'{model["name"]}_runtime': duration,
                        }
                    else:
                        train_acc = {
                            f'{model["name"]}_train_acc': np.round(
                                instance.accuracy_, decimals=4
                            ),
                            # f'{model["name"]}_runtime': duration,
                        }
                    out = {**out, **train_acc}
                    y_pred = instance.predict(X_test_)
                    test_acc = {
                        f'{model["name"]}_test_acc': np.round(
                            accuracy_score(y_test_, y_pred), decimals=4
                        )
                    }
                    out = {**out, **test_acc}
                except Exception as e:
                    print(
                        f'File : {datapath} failed for depth = {depth} and model {model["name"]}'
                    )
                    print(f"Error : {e}")
            out = {**dataset_name, **out}
            ans.append(out)
        return ans

    results = list()
    for fold in range(n_folds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=val_size, random_state=0
        )
        parameters = [
            (level, (X_train, X_test, y_train, y_test), fold) for level in noise_levels
        ]

        with Pool(n_threads) as p:
            results.append(p.starmap(exec_noise, parameters))
    everything = list()
    for fold_results in results:
        for sub in fold_results:
            everything += sub
    return pd.DataFrame(everything)


def add_noise(path, percentage=0.1):
    dataset = np.genfromtxt(path, delimiter=" ")
    X, y = dataset[:, 1:], dataset[:, 0]
    size = y.shape[0]
    if percentage == 0:
        return X, y
    to_flip = random.sample(range(size), k=int(percentage * size))
    y[to_flip] = (y[to_flip] + 1) % 2

    return X, y


def add_noise_numpy(y_train, percentage=0.1):
    size = y_train.shape[0]
    if percentage == 0:
        return y_train
    to_flip = random.sample(range(size), k=int(percentage * size))
    y_train[to_flip] = (y_train[to_flip] + 1) % 2
    return y_train


def create_paths(data_folders):
    paths = []
    for folder in data_folders:
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
            else:
                paths.append(os.path.join(folder, file))
    return paths


def get_stats(path):
    if path.endswith(".py") or path.endswith(".csv"):
        return None
    dataset = np.genfromtxt(path, delimiter=" ")
    X, y = dataset[:, 1:], dataset[:, 0]
    return {
        # "name": os.path.splitext(path)[0],
        "features": X.shape[1],
        "transactions": X.shape[0],
    }


def get_directory_datasets_stats(directory_path, save=None):
    data = list()
    for file in os.listdir(directory_path):
        path = os.path.join(directory_path, file)
        infos = get_stats(path)
        if infos is not None:
            data.append(infos)
    df = pd.DataFrame(data)
    if save is not None:
        df.to_csv(save, index=False)

    return df


def models_plots(
    data, dataset, subset="train", depth=3, methods=None, use_limit=False, save=None
):
    def rename_method_col(row):
        s = row["method"].split("_")
        s = s[0 : len(s) - 2]
        row["method"] = s[0] if len(s) < 2 else "_".join(s)
        return row

    subset_cols = [col for col in data.columns if subset in col]
    sub_df = data[(data.name == dataset) & (data.depth == depth)][
        MAIN_COLS + subset_cols
    ]
    others = [a for a in sub_df.columns if subset not in a]
    sub_df = sub_df.melt(id_vars=others, var_name="method", value_name="error")
    sub_df = sub_df.apply(lambda x: rename_method_col(x), axis=1)
    if methods is not None:
        sub_df = sub_df[sub_df.method.isin(methods)]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    plot = sns.lineplot(
        x="noise_level",
        y="error",
        hue="method",
        style="method",
        markers=True,
        alpha=0.9,
        data=sub_df,
    )
    plt.legend(title="Methods", fontsize=20)
    plt.xlabel("Noise Level", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=18)

    if use_limit:
        plt.ylim(0)

    if save is not None:
        plot.get_figure().savefig(save, dpi=600)
    plt.close()
