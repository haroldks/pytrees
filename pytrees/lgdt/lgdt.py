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
        self.is_optimal_ = False
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.data_structure = data_structure
        self.fit_method = fit_method
        self.set_internal_class(LGDTInternalClassifier)


class IDKClassifier(Predictor, BaseEstimator, ClassifierMixin):
    def __init__(
        self, min_sup=1, data_structure="reversible_sparse_bitset", fit_method="murtree"
    ):
        super().__init__()
        self.is_optimal_ = False
        self.min_sup = min_sup
        self.max_depth = 0
        self.data_structure = data_structure
        self.fit_method = fit_method
        self.set_internal_class(LGDTInternalClassifier)
