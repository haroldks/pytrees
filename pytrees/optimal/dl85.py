from sklearn.base import BaseEstimator, ClassifierMixin
from pytrees.predictor import (
    Predictor,
    Specialization,
    LowerBound,
    Branching,
    CacheInit,
    Heuristic,
)
from pytrees_internal.optimal import Dl85InternalClassifier


class DL85Classifier(Predictor, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        max_error=-1,
        max_time=-1,
        specialization=Specialization.MurTree,
        lower_bound=LowerBound.Similarity,
        one_time_sort=True,
        heuristic=Heuristic.None_,
        branching=Branching.Dynamic,
        cache_init=CacheInit.Dynamic,
        cache_init_size=0,
    ):
        super().__init__()
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.max_error = max_error
        self.max_time = max_time
        self.specialization = specialization
        self.lower_bound = lower_bound
        self.branching = branching
        self.cache_init = cache_init
        self.cache_init_size = cache_init_size
        self.one_time_sort = one_time_sort
        self.heuristic = heuristic

        clf = Dl85InternalClassifier(
            min_sup,
            max_depth,
            max_error,
            max_time,
            specialization,
            lower_bound,
            branching,
            one_time_sort,
            heuristic,
            cache_init,
            cache_init_size,
        )

        self.set_classifier(clf)
