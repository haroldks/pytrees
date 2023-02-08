from sklearn.base import BaseEstimator, ClassifierMixin
from pytrees.predictor import Predictor
from pytrees_internal.optimal import Dl85InternalClassifier


class DL85Classifier(Predictor, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        max_error=-1,
        max_time=-1,
        specialization="murtree",
        lower_bound="similarity",
        one_time_sort=True,
        heuristic="no_heuristic",
        branching="dynamic",
    ):
        super().__init__()
        self.min_sup = min_sup
        self.max_depth = max_depth
        self.max_error = max_error
        self.max_time = max_time
        self.specialization = specialization
        self.lower_bound = lower_bound
        self.branching = branching
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
        )

        self.set_classifier(clf)
