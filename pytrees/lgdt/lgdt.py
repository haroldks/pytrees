from sklearn.base import BaseEstimator, ClassifierMixin
from pytrees.predictor import Predictor
from pytrees_internal.lgdt import LGDTInternalClassifier


class LGDTClassifier(Predictor, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        data_structure="reversible_sparse_bitset",
        fit_method="murtree",
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.data_structure = data_structure
        self.fit_method = fit_method

        clf = LGDTInternalClassifier(min_sup, max_depth, data_structure, fit_method)
        self.set_classifier(clf)


class IDKClassifier(Predictor, BaseEstimator, ClassifierMixin):
    def __init__(
        self, min_sup=1, data_structure="reversible_sparse_bitset", fit_method="murtree"
    ):
        super().__init__()
        self.min_sup = min_sup
        self.data_structure = data_structure
        self.fit_method = fit_method

        clf = LGDTInternalClassifier(min_sup, 0, data_structure, fit_method)
        self.set_classifier(clf)
