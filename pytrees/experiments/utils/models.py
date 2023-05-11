from sklearn.ensemble import BaggingClassifier
from pydl85 import DL85Classifier, Cache_Type
from pytrees.lgdt import LGDTClassifier
from sklearn.tree import DecisionTreeClassifier

LGDT_ERROR = {
    "name": "lgdt_mur",
    "instance": LGDTClassifier(
        min_sup=0,
        max_depth=0,
        data_structure="reversible_sparse_bitset",
        fit_method="murtree",
    ),
}

LGDT_IG = {
    "name": "lgdt_ig",
    "instance": LGDTClassifier(
        min_sup=0,
        max_depth=0,
        data_structure="reversible_sparse_bitset",
        fit_method="info_gain",
    ),
}

CART = {
    "name": "cart",
    "instance": DecisionTreeClassifier(
        criterion="gini", splitter="best", max_depth=0, min_samples_split=0
    ),
}

BAGGED_LDGT_MUR = {
    "name": "bagged_lgdt_mur",
    "instance": BaggingClassifier(
        base_estimator=LGDTClassifier(
            min_sup=0,
            max_depth=0,
            data_structure="reversible_sparse_bitset",
            fit_method="murtree",
        ),
        n_estimators=10,
        max_samples=0.6321,
        bootstrap=True,
    ),
}
BAGGED_LDGT_IG = {
    "name": "bagged_lgdt_ig",
    "instance": BaggingClassifier(
        base_estimator=LGDTClassifier(
            min_sup=0,
            max_depth=0,
            data_structure="reversible_sparse_bitset",
            fit_method="info_gain",
        ),
        n_estimators=10,
        max_samples=0.6321,
        bootstrap=True,
    ),
}

DL85 = {
    "name": "dl8.5",
    "instance": DL85Classifier(
        min_sup=0, max_depth=0, time_limit=600, cache_type=Cache_Type.Cache_HashCover
    ),
}

LGDT_SPARSE = {
    "name": "lgdt_error_sparse",
    "instance": LGDTClassifier(
        min_sup=0,
        max_depth=0,
        data_structure="reversible_sparse_bitset",
        fit_method="murtree",
    ),
}

LGDT_BITSET = {
    "name": "lgdt_error_bitset",
    "instance": LGDTClassifier(
        min_sup=0, max_depth=0, data_structure="regular_bitset", fit_method="murtree"
    ),
}

LGDT_HZ = {
    "name": "lgdt_error_horizontal",
    "instance": LGDTClassifier(
        min_sup=0, max_depth=0, data_structure="horizontal_data", fit_method="murtree"
    ),
}

LGDT_RAW = {
    "name": "lgdt_error_raw",
    "instance": LGDTClassifier(
        min_sup=0, max_depth=0, data_structure="raw_binary_data", fit_method="murtree"
    ),
}
