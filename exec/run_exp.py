import os
import shutil

import numpy as np

from functions import (
    run_multithread_on_single_test_set,
    run_on_single_test_set,
    get_stats,
    models_plots,
)
from models import LGDT_IG, LGDT_MUR, CART

MIN_SUP = 5
DEPTHS = range(2, 3)
VAL_SIZE = 0.2
N_FOLDS = 1

SAVE_PLOTS = True
N_THREADS = 8

NOISE_LEVELS = np.arange(0.0, 0.05, 0.05)

RESULTS_DIR = "results"
SUB_DIRS = ["csv", "data", "plots", "tex", "trees"]
OUTPUT_FILE = f"{RESULTS_DIR}/" + "{data_type}/{folder}/csv/known_models_res_{type}.csv"
PLOT_NAME = (
    f"{RESULTS_DIR}/" + "{data_type}/{folder}/plots/{name}_{depth}_{subset}_plot.pdf"
)

DATA_TYPE = ["raw", "cleaned/non-unique", "cleaned/unique"]
BASE_FOLDER = "../data"

DATA_FOLDERS = [
    "datasetsNina_reduced",
    "datasetsDL",
    "datasetsNina",
    "datasetsNL",
    "base_datasets",
    "datasetsHu",
]

MODELS = [LGDT_IG, LGDT_MUR, CART]


def main():
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    for data_type in DATA_TYPE:
        for folder in DATA_FOLDERS:
            for sub in SUB_DIRS:
                os.makedirs(f"{RESULTS_DIR}/{data_type}/{folder}/{sub}")

    for data_type in DATA_TYPE:
        type_folder = os.path.join(BASE_FOLDER, data_type)
        for folder in DATA_FOLDERS:
            folder_path = os.path.join(type_folder, folder)
            results = list()
            for file in os.listdir(folder_path):
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
                path = os.path.join(folder_path, file)
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
                    df.groupby(
                        ["name", "features", "transactions", "noise_level", "depth"]
                    )
                    .mean()
                    .reset_index()
                    .round(3)
                )
            df.to_csv(
                OUTPUT_FILE.format(data_type=data_type, folder=folder, type="acc"),
                index=False,
            )
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
                                save=PLOT_NAME.format(
                                    name=name,
                                    depth=depth,
                                    subset=sub,
                                    folder=folder,
                                    data_type=data_type,
                                ),
                            )


if __name__ == "__main__":
    main()
