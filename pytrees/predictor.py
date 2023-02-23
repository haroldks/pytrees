import json
import uuid
from enum import IntEnum

from sklearn.utils import check_array, check_X_y, assert_all_finite
from sklearn.exceptions import NotFittedError
from .exceptions import TreeNotFoundError, SearchFailedError


class Specialization(IntEnum):
    None_ = 0
    MurTree = 1


class LowerBound(IntEnum):
    None_ = 0
    Similarity = 1


class Branching(IntEnum):
    None_ = 0
    Dynamic = 1


class CacheInit(IntEnum):
    None_ = 0
    Dynamic = 1
    FromUser = 2


class Heuristic(IntEnum):
    None_ = 0
    InformationGain = 1
    InformationGainRatio = 2
    GiniIndex = 3


class DiscrepancyStrategy(IntEnum):
    None_ = (0,)
    Incremental = (1,)
    Double = (2,)


class Predictor:
    def __init__(self):
        self.tree_error_ = None
        self.accuracy_ = None
        self.tree_ = None
        self.size_ = None
        self.depth_ = None
        self.is_fitted_ = False
        self.__internal_classifier = None
        self.statistics = None

    def set_classifier(self, clf):
        self.__internal_classifier = clf

    def fit(self, X, y=None):

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

        self.__internal_classifier.train(X, y)

        tree = json.loads(self.__internal_classifier.tree)

        if len(tree["tree"]) == 1 and tree["tree"][0]["value"]["out"] not in [0, 1]:
            self.tree_ = None
        else:
            self.tree_ = tree
            self.is_fitted_ = True
            self.tree_error_ = tree["tree"][0]["value"]["error"]
            self.compute_max_depth()
            self.compute_size()
            self.compute_accuracy(len(X))
        self.statistics = json.loads(self.__internal_classifier.statistics)

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
            self.accuracy_ = round(1 - self.tree_error_ / train_size, 5)

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
        while not Predictor.is_leaf_node(node):
            if instance[node["value"]["test"]] == 1:
                node = self.tree_["tree"][node["right"]]
            else:
                node = self.tree_["tree"][node["left"]]
        return node["value"]["out"]

    def get_dot_body_rec(self, node, parent=None, left=False):
        gstring = ""
        id = str(uuid.uuid4())
        id = id.replace("-", "_")

        if node["right"] == node["left"]:
            gstring += (
                "leaf_"
                + id
                + ' [label="{{class|'
                + str(node["value"]["out"])
                + "}|{error|"
                + str(node["value"]["error"])
                + '}}"];\n'
            )
            gstring += (
                "node_"
                + parent
                + " -> leaf_"
                + id
                + " [label="
                + str(int(left))
                + "];\n"
            )
        else:
            gstring += (
                "node_"
                + id
                + ' [label="{{feat|'
                + str(node["value"]["test"])
                + '}}"];\n'
            )
            gstring += (
                "node_"
                + parent
                + " -> node_"
                + id
                + " [label="
                + str(int(left))
                + "];\n"
            )
            gstring += self.get_dot_body_rec(
                self.tree_["tree"][node["left"]], id, left=True
            )
            gstring += self.get_dot_body_rec(
                self.tree_["tree"][node["right"]], id, left=False
            )
        return gstring

    def export_to_graphviz_dot(self):
        gstring = "digraph Tree { \n" "graph [ranksep=0]; \n" "node [shape=record]; \n"
        id = str(uuid.uuid4())
        id = id.replace("-", "_")

        root = self.tree_["tree"][0]
        print(root)
        feat = root["value"]["test"]
        if feat is not None:
            gstring += (
                "node_"
                + id
                + ' [label="{{feat|'
                + str(feat)
                + "}|{error|"
                + str(self.tree_error_)
                + '}}"];\n'
            )
            gstring += self.get_dot_body_rec(
                self.tree_["tree"][root["left"]], parent=id
            )
            gstring += self.get_dot_body_rec(
                self.tree_["tree"][root["right"]], parent=id
            )
        gstring += "}"
        return gstring