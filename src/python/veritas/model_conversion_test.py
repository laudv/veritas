import json
import numpy as np

from xgboost.sklearn import XGBModel
from xgboost.core import Booster as xgbbooster

import sklearn.tree as sktree
from sklearn.ensemble import _forest

from lightgbm import LGBMModel
from lightgbm import Booster as lgbmbooster

from sklearn.metrics import mean_absolute_error

from . import AddTree


def test_model(model, at, data):
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
            return mae_classification(model, at, data, "xgb", True)
        elif "logistic" in model_type:
            return mae_classification(model, at, data, "xgb")

        return mae_regression(model, at, data)

    # Sklearn RandomForest / InsulationForest in the Future?
    elif "sklearn.ensemble._forest" in str(module_name):
        type_ = type(model).__name__
        if "Regressor" in type_:
            return mae_regression(model, at, data)
        elif "Classifier" in type_:
            return mae_classification(model, at, data, "sklearn")

    # LGBM
    elif "lightgbm" in str(module_name):
        if isinstance(model, LGBMModel):
            booster = model.booster_
        assert isinstance(
            booster, lgbmbooster), f"not xgb.Booster but {type(booster)}"
        model_type = booster.dump_model()["objective"]
        if "multi" in model_type:
            return mae_classification(model, at, data, "lgbm", True)
        elif "binary" in model_type:
            return mae_classification(model, at, data, "lgbm")

        return mae_regression(model, at, data)


def mae_regression(model, at, data):
    X, y = data
    yhat = model.predict(X)
    sqerr = sum((y - yhat)**2)
    rmse = np.sqrt(sqerr)/len(X)

    return mean_absolute_error(yhat, at.predict(X)), rmse


def mae_classification(model, ats, data, model_type, multiclass=False):
    X, y = data
    yhat = model.predict(X)
    if model_type == "sklearn":
        yhatm = model.predict_proba(X)
    elif model_type == "xgb":
        yhatm = model.predict(X, output_margin=True)
    elif model_type == "lgbm":
        yhatm = model.predict_proba(X, raw_score=True)

    acc = np.mean(yhat == y)
    if model_type == "xgb" or "lgbm":
        if multiclass:
            yhatm_at = np.zeros_like(yhatm)
            for k, at in enumerate(ats):
                yhatm_at[:, k] = at.predict(X).ravel()
        else:
            yhatm_at = ats.predict(X)
    else:
        yhatm_at = ats.predict(X)

    if multiclass:
        yhatm = yhatm.ravel()
        yhatm_at = yhatm_at.ravel()

    return mean_absolute_error(yhatm, yhatm_at), acc
