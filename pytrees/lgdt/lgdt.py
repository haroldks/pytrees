from sklearn.base import BaseEstimator, ClassifierMixin

from pytrees.enum_params import DataStructure, FitMethod
from pytrees.predictor import Predictor
from pytrees_internal.lgdt import LGDTInternalClassifier, ParallelLGDTInternalClassifier


class LGDTClassifier(Predictor, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        parallel=False,
        num_threads=0,
        data_structure=DataStructure.ReversibleBitset,
        fit_method=FitMethod.MurTree,
    ):
        super().__init__()
        self.is_optimal_ = False
        self.parallel = parallel
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.data_structure = data_structure
        self.fit_method = fit_method
        self.is_parallel_ = parallel
        self.num_threads = num_threads
        if parallel:
            self.set_internal_class(ParallelLGDTInternalClassifier)
        else:
            self.set_internal_class(LGDTInternalClassifier)


class IDKClassifier(Predictor, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_sup=1,
        data_structure=DataStructure.ReversibleBitset,
        fit_method=FitMethod.MurTree,
    ):
        super().__init__()
        self.is_optimal_ = False
        self.min_sup = min_sup
        self.max_depth = 0
        self.data_structure = data_structure
        self.fit_method = fit_method
        self.set_internal_class(LGDTInternalClassifier)
