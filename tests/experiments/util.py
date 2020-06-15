import os
import numpy as np
import scipy
import scipy.io
import xgboost as xgb
from sklearn.datasets import fetch_openml

def load_openml(name, data_id, task="classification", force=False):
    """
    mnist: data_id=554
    covtype: data_id=1596
    """
    if not os.path.exists(f"tests/data/{name}.mat") or force:
        print(f"loading {name} with fetch_openml")
        d = fetch_openml(data_id=data_id)
        X = d["data"]
        if task == "regression":
            y = np.array(list(map(float, d["target"])))
        elif task == "classification":
            y = np.array(list(map(int, d["target"])))
        else:
            raise RuntimeError("invalid task")
        scipy.io.savemat(f"tests/data/{name}.mat", {"X": X, "y": y},
                do_compression=True, format="5")
    else:
        print(f"loading {name} MAT file")
        mat = scipy.io.loadmat(f"tests/data/{name}.mat") # much faster
        X = mat["X"]
        y = mat["y"]

    y = y.reshape((X.shape[0],))

    return X, y

def train_test_indices(num_examples, seed=82394188):
    np.random.seed(seed)
    indices = np.random.permutation(num_examples)

    m = int(num_examples*0.9)
    Itrain = indices[0:m]
    Itest = indices[m:]

    return Itrain, Itest

def double_check_at_output(model, at, X):
    max_diff = 0.0
    tmp = model.predict(xgb.DMatrix(X, missing=None), output_margin=True)
    for i in range(X.shape[0]):
        p = at.predict_single(X[i, :])
        max_diff = max(max_diff, p-tmp[i])
    return max_diff
