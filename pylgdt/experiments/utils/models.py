from sklearn.ensemble import BaggingClassifier
from pydl85 import DL85Classifier
from pylgdt import LGDTPredictor
from sklearn.tree import DecisionTreeClassifier

LGDT_ERROR = {
    "name": "lgdt_mur",
    "instance": LGDTPredictor(
        min_sup=0, max_depth=0, data_structure="sparse_bitset", fit_method="murtree"
    ),
}

LGDT_IG = {
    "name": "lgdt_ig",
    "instance": LGDTPredictor(
        min_sup=0, max_depth=0, data_structure="sparse_bitset", fit_method="infogain"
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
        base_estimator=LGDTPredictor(
            min_sup=0, max_depth=0, data_structure="sparse_bitset", fit_method="murtree"
        ),
        n_estimators=10,
        max_samples=0.6321,
        bootstrap=True,
    ),
}
BAGGED_LDGT_IG = {
    "name": "bagged_lgdt_ig",
    "instance": BaggingClassifier(
        base_estimator=LGDTPredictor(
            min_sup=0,
            max_depth=0,
            data_structure="sparse_bitset",
            fit_method="infogain",
        ),
        n_estimators=10,
        max_samples=0.6321,
        bootstrap=True,
    ),
}

DL85 = {
    "name": "dl8.5",
    "instance": DL85Classifier(min_sup=0, max_depth=0, time_limit=600),
}

LGDT_SPARSE = {
    "name": "lgdt_error_sparse",
    "instance": LGDTPredictor(
        min_sup=0, max_depth=0, data_structure="sparse_bitset", fit_method="murtree"
    ),
}

LGDT_BITSET = {
    "name": "lgdt_error_bitset",
    "instance": LGDTPredictor(
        min_sup=0, max_depth=0, data_structure="bitset", fit_method="murtree"
    ),
}

LGDT_HZ = {
    "name": "lgdt_error_horizontal",
    "instance": LGDTPredictor(
        min_sup=0, max_depth=0, data_structure="horizontal", fit_method="murtree"
    ),
}
