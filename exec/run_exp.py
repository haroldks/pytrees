import os
import shutil

import numpy as np

from functions import (
    create_paths,
    run_multithread_on_single_test_set,
    run_on_single_test_set,
    get_stats,
    models_plots,
)
from models import LGDT_IG, LGDT_MUR, CART

MIN_SUP = 5
DEPTHS = range(4, 5)
VAL_SIZE = 0.2
N_FOLDS = 5

SAVE_PLOTS = True
N_THREADS = 8

NOISE_LEVELS = np.arange(0.0, 0.25, 0.05)

RESULTS_DIR = "results"
SUB_DIRS = ["csv", "data", "plots", "tex", "trees"]
OUTPUT_FILE = f"{RESULTS_DIR}" + "/csv/known_models_res_{type}.csv"
PLOT_NAME = f"{RESULTS_DIR}" + "/plots/{name}_{depth}_{subset}_plot.pdf"

BASE_FOLDER = "../data/cleaned/unique"


DATA_FOLDERS = [
    # f"{BASE_FOLDER}/datasetsNina_reduced",
    # f"{BASE_FOLDER}/datasetsDL",
    f"{BASE_FOLDER}/datasetsNina",
    # f"{BASE_FOLDER}/datasetsNL",
    # f"{BASE_FOLDER}/base_datasets",
    # f"{BASE_FOLDER}/datasetsHu",
]

MODELS = [LGDT_IG, LGDT_MUR, CART]


def main():
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    for sub in SUB_DIRS:
        os.makedirs(f"{RESULTS_DIR}/{sub}")

    paths = create_paths(DATA_FOLDERS)
    results = list()

    for path in paths:
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
                        methods=["lgdt_mur", "lgdt_ig", "cart"],
                        save=PLOT_NAME.format(name=name, depth=depth, subset=sub),
                    )


if __name__ == "__main__":
    main()
