from pylgdt import LGDTPredictor, IDKPredictor
from sklearn.tree import DecisionTreeClassifier

LGDT_MUR = {
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


#
# C45 = {
#     "name": "c4.5",
#     "instance": CartPredictor(min_sup=0, max_depth=0, metric=3, print_output=False),
# }
#
# CART_GINI = {
#     "name": "cart_gini",
#     "instance": CartPredictor(min_sup=0, max_depth=0, metric=0, print_output=False),
# }
#
# CART_ERROR = {
#     "name": "cart_error",
#     "instance": CartPredictor(min_sup=0, max_depth=0, metric=1, print_output=False),
# }
#
# SOURCE_TREE = {
#     "name": "source_tree",
#     "instance": TreePredictor("", load=False),
# }
#
#
# DL85 = {"name": "dl8.5", "instance": DL85Classifier(min_sup=0, max_depth=0, time_limit=600)}
