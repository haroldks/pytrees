import os
import shutil

import numpy as np

from pytrees.experiments.utils.functions import (
    run_multithread_on_single_test_set,
    run_on_single_test_set,
    get_stats,
)
from pytrees.experiments.utils.models import (
    C45,
    LGDT_BITSET,
    LGDT_SPARSE,
    LGDT_HZ,
    DL85,
)

MIN_SUP = 5
DEPTHS = range(2, 5)
VAL_SIZE = 0.2
N_FOLDS = 1

SAVE_PLOTS = True
N_THREADS = 0

NOISE_LEVELS = np.arange(0.0, 0.05, 0.05)

RESULTS_DIR = "results"
SUB_DIRS = ["csv", "data", "plots", "tex", "trees"]
OUTPUT_FILE = f"{RESULTS_DIR}/" + "{data_type}/{folder}/csv/known_models_res_{type}.csv"
PLOT_NAME = (
    f"{RESULTS_DIR}/" + "{data_type}/{folder}/plots/{name}_{depth}_{subset}_plot.pdf"
)

DATA_TYPE = ["raw"]
BASE_FOLDER = "data"

DATA_FOLDERS = [
    "datasetsDL",
]

MODELS = [LGDT_BITSET, LGDT_SPARSE, LGDT_HZ, C45, DL85]


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
            max_len = len(max(os.listdir(folder_path), key=lambda x: len(x)))
            folder_size = len(os.listdir(folder_path))
            print(f"Inside :\t{folder_path}")
            for i, file in enumerate(os.listdir(folder_path)):
                name = file.split(".")[0]
                if name in [  # These are test datasets and are too small
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
                print(
                    f"Evolution in the folder : {name}",
                    " " * (max_len - len(name)),
                    f" {i + 1}/{folder_size}",
                    end="\r",
                    flush=True,
                )
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
                    .round(5)
                )
            df.to_csv(
                OUTPUT_FILE.format(data_type=data_type, folder=folder, type="acc"),
                index=False,
            )


if __name__ == "__main__":
    main()
