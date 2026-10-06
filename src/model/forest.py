"""Evaluate the original scikit-learn 0.22.2 random forest without scikit-learn.

`model.pk` was pickled with scikit-learn 0.22.2 on Python 3.7. Modern
scikit-learn cannot load it (the tree node layout changed in 1.3), and 0.22.2
has no wheels for the Python the LMDS runner needs. The forest is just arrays,
though: this module reads the pickle with an allow-listed unpickler that maps
the scikit-learn classes onto inert stand-ins (nothing from scikit-learn is
imported or executed), pulls out each tree's node table and leaf values, and
evaluates them with numpy exactly as scikit-learn 0.22.2 did:

- inputs are cast to float32 (sklearn's tree DTYPE) before comparison;
- a sample goes left when ``x[feature] <= threshold`` (threshold is float64);
- the forest prediction is the float64 sum of tree predictions, in estimator
  order, divided by the number of trees.
"""

import importlib
import pickle
from pathlib import Path

import numpy as np

TREE_LEAF = -1


class _Stub:
    """Inert stand-in for a pickled scikit-learn object: just keeps its state."""

    def __init__(self, *args):
        self._args = args

    def __setstate__(self, state):
        self.__dict__.update(state)


class _RandomForestRegressor(_Stub):
    pass


class _DecisionTreeRegressor(_Stub):
    pass


class _Tree(_Stub):
    pass


_SKLEARN_CLASSES = {
    ("sklearn.ensemble._forest", "RandomForestRegressor"): _RandomForestRegressor,
    ("sklearn.tree._classes", "DecisionTreeRegressor"): _DecisionTreeRegressor,
    ("sklearn.tree._tree", "Tree"): _Tree,
}

# numpy globals needed to rebuild the node/value arrays. Pickles from numpy 1.x
# reference numpy.core, which numpy 2 renamed to numpy._core.
_NUMPY_GLOBALS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
}


class _ForestUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SKLEARN_CLASSES:
            return _SKLEARN_CLASSES[(module, name)]
        if (module, name) in _NUMPY_GLOBALS:
            if module.startswith("numpy.core"):
                try:
                    mod = importlib.import_module(module.replace("numpy.core", "numpy._core"))
                except ImportError:  # numpy < 2
                    mod = importlib.import_module(module)
            else:
                mod = importlib.import_module(module)
            return getattr(mod, name)
        raise pickle.UnpicklingError(
            f"Refusing to load {module}.{name}: not part of a scikit-learn "
            "RandomForestRegressor pickle"
        )


class Forest:
    """Random-forest regressor rebuilt from a scikit-learn 0.22.2 pickle."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            rf = _ForestUnpickler(f).load()
        if not isinstance(rf, _RandomForestRegressor):
            raise ValueError(f"{path} does not contain a RandomForestRegressor")
        if rf.n_outputs_ != 1:
            raise ValueError(f"expected a single-output forest, got {rf.n_outputs_}")

        self.n_features = int(rf.n_features_)
        self.sklearn_version = rf.__dict__.get("_sklearn_version")
        self.trees = []
        for est in rf.estimators_:
            t = est.tree_
            nodes = t.nodes
            self.trees.append((
                np.asarray(nodes["left_child"], dtype=np.int64),
                np.asarray(nodes["right_child"], dtype=np.int64),
                np.asarray(nodes["feature"], dtype=np.int64),
                np.asarray(nodes["threshold"], dtype=np.float64),
                np.asarray(t.values[:, 0, 0], dtype=np.float64),
            ))

    @property
    def n_estimators(self) -> int:
        return len(self.trees)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} features, got shape {X.shape}"
            )
        rows = np.arange(X.shape[0])
        total = np.zeros(X.shape[0], dtype=np.float64)
        for left, right, feature, threshold, value in self.trees:
            node = np.zeros(X.shape[0], dtype=np.int64)
            while True:
                active = left[node] != TREE_LEAF
                if not active.any():
                    break
                n = node[active]
                go_left = X[rows[active], feature[n]] <= threshold[n]
                node[active] = np.where(go_left, left[n], right[n])
            total += value[node]
        return total / len(self.trees)
