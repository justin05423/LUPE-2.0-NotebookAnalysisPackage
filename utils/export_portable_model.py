"""
Export the LUPE random-forest classifier (model/model.pkl, pickled with scikit-learn 1.2.1) to a
version-independent file of plain numpy arrays, model/model_portable.npz.

Run once, in an environment that can still unpickle model.pkl (scikit-learn 1.2.x, e.g. the LUPE2APP env):
    python utils/export_portable_model.py model/model.pkl model/model_portable.npz   (from the package root)

utils/model_compat.py rebuilds the forest from this file in any newer scikit-learn (e.g. Google Colab).
"""
import sys
import json
import pickle
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    model = pickle.load(f)

fields = ["left_child", "right_child", "feature", "threshold", "impurity",
          "n_node_samples", "weighted_n_node_samples"]
arrays = {}
node_counts, max_depths = [], []
for est in model.estimators_:
    state = est.tree_.__getstate__()
    nodes = state["nodes"]
    for name in fields:
        arrays.setdefault(name, []).append(nodes[name])
    arrays.setdefault("values", []).append(state["values"])
    node_counts.append(int(state["node_count"]))
    max_depths.append(int(state["max_depth"]))

meta = {
    "sklearn_version_exported_from": __import__("sklearn").__version__,
    "params": {k: v for k, v in model.get_params().items()},
    "n_features_in_": int(model.n_features_in_),
    "n_outputs_": int(model.n_outputs_),
    "classes_": model.classes_.tolist(),
    "node_counts": node_counts,
    "max_depths": max_depths,
    "estimator_params": {k: getattr(model.estimators_[0], k) for k in model.estimator_params
                         if k != "monotonic_cst"},
}
np.savez_compressed(dst, meta=json.dumps(meta), **{k: np.concatenate(v) for k, v in arrays.items()})
print(f"Exported {len(node_counts)} trees ({sum(node_counts):,} nodes) to {dst}")
