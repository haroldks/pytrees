import json
import sys

from sklearn.base import BaseEstimator
from .exceptions import TreeNotFoundError, SearchFailedError
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_X_y, assert_all_finite, check_array


class LGDTPredictor(BaseEstimator):
    def __init__(
        self,
        min_sup=1,
        max_depth=1,
        data_structure="sparse_bitset",
        fit_method="murtree",
        log=False,
    ):
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.data_structure = data_structure
        self.fit_method = fit_method
        self.log = log

        self.tree_ = None
        self.size_ = -1
        self.depth_ = -1
        self.error_ = -1
        self.accuracy_ = -1
        self.runtime_ = -1
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """Implements the standard fitting function for a DL8.5 classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        target_is_need = True if y is not None else False

        if target_is_need:  # target-needed tasks (eg: classification, regression, etc.)
            # Check that X and y have correct shape and raise ValueError if not
            X, y = check_X_y(X, y, dtype="float64")
            # if opt_func is None and opt_pred_func is None:
            #     print("No optimization criterion defined. Misclassification error is used by default.")
        else:  # target-less tasks (clustering, etc.)
            # Check that X has correct shape and raise ValueError if not
            assert_all_finite(X)
            X = check_array(X, dtype="float64")

        # print('aaa :', y.astype('float64'))
        import perf_lgdt

        solution = perf_lgdt.run(
            X,
            y.astype("float64"),  # For Bagging remove after
            self.min_sup,
            self.max_depth,
            self.data_structure,
            self.fit_method,
            self.log,
        )
        solution = json.loads(solution)
        error = self.get_model_train_error(solution)
        if error < sys.maxsize:
            self.tree_ = solution
            self.is_fitted_ = True
            self.error_ = error
            self.compute_max_depth()
            self.compute_size()
            self.compute_accuracy(len(X))
        else:
            self.tree_ = None

    @staticmethod
    def get_model_train_error(tree):
        error_ = tree["tree"][0]["value"]["error"]
        return error_

    def compute_max_depth(self):
        def recursion(subtree_index):
            node = self.tree_["tree"][subtree_index]
            if node["left"] == node["right"]:
                return 1
            else:
                d = max(recursion(node["left"]), recursion(node["right"])) + 1
                return d

        if self.is_fitted_:
            self.depth_ = recursion(0) - 1

    def compute_size(self):
        if self.is_fitted_:
            self.size_ = len(self.tree_["tree"])

    def compute_accuracy(self, train_size):
        if self.is_fitted_:
            self.accuracy_ = round(1 - self.error_ / train_size, 5)

    @staticmethod
    def is_leaf_node(node):
        return (node["left"] == 0) and (node["right"] == 0)

    def predict(self, X):
        """Implements the standard predict function for a DL8.5 classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """

        # Check is fit is called
        # check_is_fitted(self, attributes='tree_') # use of attributes is deprecated. alternative solution is below

        # if hasattr(self, 'sol_size') is False:  # fit method has not been called
        if self.is_fitted_ is False:  # fit method has not been called
            raise NotFittedError(
                "Call fit method first" % {"name": type(self).__name__}
            )

        if self.tree_ is None:
            raise TreeNotFoundError(
                "predict(): ",
                "Tree not found during training by DL8.5 - "
                "Check fitting message for more info.",
            )

        if hasattr(self, "tree_") is False:  # normally this case is not possible.
            raise SearchFailedError(
                "PredictionError: ",
                "DL8.5 training has failed. Please contact the developers "
                "if the problem is in the scope supported by the tool.",
            )

        # Input validation
        X = check_array(X)

        pred = []

        for i in range(X.shape[0]):
            pred.append(self.pred_value_on_dict(X[i, :]))

        return pred

    def pred_value_on_dict(self, instance, tree=None):
        node = tree if tree is not None else self.tree_["tree"][0]
        while not LGDTPredictor.is_leaf_node(node):
            if instance[node["value"]["test"]] == 1:
                node = self.tree_["tree"][node["right"]]
            else:
                node = self.tree_["tree"][node["left"]]
        return node["value"]["out"]
