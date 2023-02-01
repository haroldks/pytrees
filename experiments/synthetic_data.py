import os
import shutil
import numpy as np
from pytrees.experiments.synthetic import generate_decision_trees
from pytrees.experiments.synthetic import generate_dataset_from_decision_tree

from pytrees.experiments.utils.functions import (
    run_multithread_on_single_test_set,
    run_on_single_test_set,
    get_stats,
    models_plots,
)

from pytrees.experiments.utils.models import LGDT_IG, LGDT_ERROR, CART, DL85

USE_FULL_TREES = True

MIN_SUP = 5
DEPTHS = range(2, 6)
VAL_SIZE = 0.2
N_FOLDS = 5

SAVE_PLOTS = True
N_THREADS = 10

NOISE_LEVELS = np.arange(0.0, 0.55, 0.05)

RESULTS_DIR = "results_synthetic_data"
SUB_DIRS = ["csv", "data", "plots", "trees"]

TREE_NAME = f"{RESULTS_DIR}" + "/trees/synthetic_tree_{type}_d_{depth}.json"

DATASETS_DIR = f"{RESULTS_DIR}/data"
DATASET_NAME = f"{RESULTS_DIR}" + "/data/synthetic_dataset_{type}_d_{depth}.txt"

OUTPUT_FILE = f"{RESULTS_DIR}" + "/csv/synthetic_datasets_{type}.csv"
PLOT_NAME = f"{RESULTS_DIR}" + "/plots/{name}_{depth}_{subset}_plot.pdf"

MODELS = [
    LGDT_IG,
    LGDT_ERROR,
    CART,
    DL85,
]


def main():
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    for sub in SUB_DIRS:
        os.makedirs(f"{RESULTS_DIR}/{sub}")

    print("Generating Datasets and Trees")

    for depth in DEPTHS:
        generate_decision_trees(
            depth, to_prune=False, save=USE_FULL_TREES, name=TREE_NAME
        )
        generate_dataset_from_decision_tree(
            TREE_NAME.format(type="full", depth=depth),
            samples=depth * 1000,
            save=DATASET_NAME.format(type="full", depth=depth),
        )

    print("Running Tests")

    results = list()
    max_len = len(max(os.listdir(DATASETS_DIR), key=lambda x: len(x)))
    folder_size = len(os.listdir(DATASETS_DIR))
    for i, file in enumerate(os.listdir(DATASETS_DIR)):
        name = file.split(".")[0]
        print(
            f"Evolution in the folder : {name}",
            " " * (max_len - len(name)),
            f" {i + 1}/{folder_size}",
            end="\n",
            flush=True,
        )
        path = os.path.join(DATASETS_DIR, file)
        infos = get_stats(path)
        if N_THREADS > 0:
            out = run_multithread_on_single_test_set(
                path,
                MODELS.copy(),
                min_sup=MIN_SUP,
                depths=DEPTHS,
                val_size=VAL_SIZE,
                noise_levels=NOISE_LEVELS,
                n_folds=N_FOLDS,
                n_threads=N_THREADS,
            )
        else:
            out = run_on_single_test_set(
                path,
                MODELS.copy(),
                min_sup=MIN_SUP,
                depths=DEPTHS,
                val_size=VAL_SIZE,
                noise_levels=NOISE_LEVELS,
                n_folds=N_FOLDS,
            )
        out["features"] = infos["features"]
        out["transactions"] = infos["transactions"]

        cols = list(out.columns)
        size = len(cols)
        cols = [cols[0]] + cols[size - 2 : size] + cols[1 : size - 2]
        out = out[cols]
        results.append(out)

    df = results[0]
    for dfi in results[1:]:
        df = df.append(dfi, ignore_index=False)

    if N_FOLDS > 1:
        df = (
            df.groupby(["name", "features", "transactions", "noise_level", "depth"])
            .mean()
            .reset_index()
            .round(3)
        )
    df.to_csv(OUTPUT_FILE.format(type="acc"), index=False)
    acc_df = df

    if SAVE_PLOTS:
        for name in df.name.unique():
            for depth in df.depth.unique():
                for sub in ["train", "test"]:
                    models_plots(
                        acc_df,
                        name,
                        subset=sub,
                        depth=depth,
                        methods=[
                            "lgdt_error",
                            "lgdt_ig",
                            "cart",
                        ],
                        save=PLOT_NAME.format(name=name, depth=depth, subset=sub),
                    )


if __name__ == "__main__":
    main()
