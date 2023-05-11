from sklearn.base import BaseEstimator, ClassifierMixin
from pytrees.predictor import Predictor
from pytrees.enum_params import (
    Specialization,
    LowerBound,
    Branching,
    CacheInit,
    Heuristic,
    DiscrepancyStrategy,
)
from pytrees_internal.optimal import Dl85InternalClassifier


class LDSDL85Classifier(Predictor, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        discrepancy_budget=-1,
        discrepancy_strategy=DiscrepancyStrategy.Incremental,
        max_error=-1,
        max_time=-1,
        specialization=Specialization.None_,
        lower_bound=LowerBound.None_,
        one_time_sort=False,
        heuristic=Heuristic.InformationGain,
        branching=Branching.None_,
        cache_init=CacheInit.Dynamic,
        cache_init_size=0,
    ):
        super().__init__()
        self.is_optimal_ = True
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.discrepancy_budget = discrepancy_budget
        self.discrepancy_strategy = discrepancy_strategy
        self.max_error = max_error
        self.max_time = max_time
        self.specialization = specialization
        self.lower_bound = lower_bound
        self.branching = branching
        self.cache_init = cache_init
        self.cache_init_size = cache_init_size
        self.one_time_sort = one_time_sort
        self.heuristic = heuristic

        self.set_internal_class(Dl85InternalClassifier)
