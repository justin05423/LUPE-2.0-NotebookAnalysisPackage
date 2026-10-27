"""
Load the LUPE classifier in any scikit-learn version.

model/model.pkl was pickled with scikit-learn 1.2.1. Newer scikit-learn (1.3+, which is what Google Colab
ships) refuses to unpickle its trees ("node array from the pickle has an incompatible dtype"), and 1.2.1 has
no wheels for Python 3.12. model/model_portable.npz holds the same forest as plain numpy arrays
(written by export_portable_model.py); load_portable_model() rebuilds it with the installed scikit-learn.
"""
import json
import numpy as np


def load_portable_model(path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree._tree import Tree

    with np.load(path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files}
    meta = json.loads(str(data["meta"]))
    # 1.2.1 stored weighted class counts per node; 1.4+ stores proportions and predict_proba no longer
    # normalizes them. Proportions give the same per-tree probabilities in every version.
    values = data["values"]
    totals = values.sum(axis=-1, keepdims=True)
    values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)

    params = dict(meta["params"])
    for k in ("base_estimator", "estimator"):  # renamed across versions; not needed for prediction
        params.pop(k, None)
    model = RandomForestClassifier(**{k: v for k, v in params.items()
                                      if k in RandomForestClassifier().get_params()})
    classes = np.array(meta["classes_"])
    n_classes = len(classes)
    n_features = meta["n_features_in_"]
    n_outputs = meta["n_outputs_"]

    # Node dtype of the installed scikit-learn (1.3+ adds missing_go_to_left; it only matters with NaN inputs)
    probe = DecisionTreeClassifier().fit(np.zeros((2, 1)), [0, 1]).tree_.__getstate__()["nodes"].dtype

    estimators = []
    start = 0
    for node_count, max_depth in zip(meta["node_counts"], meta["max_depths"]):
        stop = start + node_count
        nodes = np.zeros(node_count, dtype=probe)
        for name in probe.names:
            if name in data:
                nodes[name] = data[name][start:stop]
        tree = Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
        tree.__setstate__({"max_depth": max_depth, "node_count": node_count,
                           "nodes": nodes, "values": np.ascontiguousarray(values[start:stop])})
        est = DecisionTreeClassifier(**{k: v for k, v in meta["estimator_params"].items()
                                        if k in DecisionTreeClassifier().get_params()})
        est.tree_ = tree
        est.n_features_in_ = n_features
        est.n_outputs_ = n_outputs
        est.classes_ = classes
        est.n_classes_ = n_classes
        est.max_features_ = max(1, int(np.sqrt(n_features))) if params.get("max_features") == "sqrt" else n_features
        estimators.append(est)
        start = stop

    model.estimators_ = estimators
    try:
        model.estimator_ = DecisionTreeClassifier()
    except AttributeError:  # read-only property in older versions
        pass
    model.n_estimators = len(estimators)
    model.classes_ = classes
    model.n_classes_ = n_classes
    model.n_features_in_ = n_features
    model.n_outputs_ = n_outputs
    return model
