import json
import numpy as np
import math

from xgboost.sklearn import XGBModel
from xgboost.core import Booster as xgbbooster

import sklearn.tree as sktree
from sklearn.ensemble import _forest

from lightgbm import LGBMModel
from lightgbm import Booster as lgbmbooster

from sklearn.metrics import mean_absolute_error


def test_model_conversion(model, at, data):
    module_name = getattr(model, '__module__', None)

    # XGB
    if "xgboost" in str(module_name):
        if isinstance(model, XGBModel):
            booster = model.get_booster()
        assert isinstance(
            booster, xgbbooster), f"not xgb.Booster but {type(booster)}"
        param_dump = json.loads(booster.save_config())['learner']
        model_type = param_dump["objective"]["name"]
        if "multi" in model_type:
            return mae_classification(model, at, data, "xgb", multiclass=True)
        elif "logistic" in model_type:
            return mae_classification(model, at, data, "xgb")

        return mae_regression(model, at, data)

    # Sklearn RandomForest
    elif "sklearn.ensemble._forest" in str(module_name):
        type_ = type(model).__name__
        if "Regressor" in type_:
            return mae_regression(model, at, data)
        elif "Classifier" in type_:
            multiclass = True if model.n_classes_ > 2 else False
            return mae_classification(model, at, data, "sklearn", multiclass)

    # LGBM
    elif "lightgbm" in str(module_name):
        if isinstance(model, LGBMModel):
            booster = model.booster_
        assert isinstance(
            booster, lgbmbooster), f"not xgb.Booster but {type(booster)}"
        model_type = booster.dump_model()["objective"]
        if "multi" in model_type:
            return mae_classification(model, at, data, "lgbm", multiclass=True)
        elif "binary" in model_type:
            return mae_classification(model, at, data, "lgbm")

        return mae_regression(model, at, data)


def mae_regression(model, at, data):
    X, y = data

    yhat = model.predict(X)
    sqerr = sum((y - yhat)**2)
    rmse = np.sqrt(sqerr)/len(X)
    yhat_at_pred = at.predict(X)

    return mean_absolute_error(yhat, yhat_at_pred), rmse


def mae_classification(model, ats, data, model_type, multiclass=False):
    X, y = data

    yhat = model.predict(X)
    acc = np.mean(yhat == y)

    if model_type == "sklearn":
        yhatm = model.predict_proba(X) if multiclass else [y1 for [y1,_] in model.predict_proba(X)] 
    elif model_type == "xgb":
        yhatm = model.predict(X, output_margin=True)
    elif model_type == "lgbm":
        yhatm = model.predict(X, raw_score=True)

    if model_type == "sklearn":
        yhatm_at = ats.predict(X)
    else:
        yhatm_at = ats.eval(X).ravel() if not multiclass else ats.eval(X)


    mae = mean_absolute_error(yhatm, yhatm_at)

    if mae > 1e-4:
        find_floating_errors(ats, yhatm, yhatm_at, X, multiclass)

    return mae, acc

# TODO: Make accesable from outside veritas and user-friendly 
def find_floating_errors(ats, yhatm, yhatm_at, X, multiclass=False):

    for example in range(len(X)):
        y = yhatm[example]
        y_mod = yhatm_at[example]

        diff = (abs(y-y_mod) > 1e-6) if not multiclass else any(diff > 1e-6 for diff in abs(y-y_mod))
        if diff:
            # Difference: print(abs(y-y_mod))
            # If difference is always the same, there is a problem with the base_score
            # print("[Warning] Found potential floating error after conversion!")
            # print(f"[Warning] Example: {example}")

            for tree in ats:
                leaf_node = tree.eval_node(X[example], tree.root())
                find_floating_splits(tree, leaf_node, X[example])


def find_floating_splits(tree, node, example):
    while not tree.is_root(node):
        node = tree.parent(node)
        split = tree.get_split(node)
        if math.isclose(example[split.feat_id], split.split_value, abs_tol=1e-8):
            print(
                f"[Warning] Feature {split.feat_id}: {example[split.feat_id]}    Split value: {split.split_value}")
            return
    return
