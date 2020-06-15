import os, pickle, timeit, subprocess
import scipy.io
import numpy as np
import xgboost as xgb

import util
from treeck import *
from treeck.xgb import addtree_from_xgb_model

RESULT_DIR = "tests/experiments/scale"

# - Loading the covertype data set --------------------------------------------

X, y = util.load_openml("covtype", data_id=1596)
y = (y==2)
num_examples, num_features = X.shape
print("balance:", sum(y) / num_examples)
Itrain, Itest = util.train_test_indices(num_examples)



# - Training XGBoost model ----------------------------------------------------

def train_model(lr, num_trees, max_depth=5):
    model_name = f"model-{num_trees}.xgb"
    dtest = xgb.DMatrix(X[Itest], y[Itest], missing=None)
    if not os.path.isfile(os.path.join(RESULT_DIR, model_name)):
        print(f"training model learning_rate={lr}, num_trees={num_trees}")
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": max_depth,
            "learning_rate": lr,
            "eval_metric": "error",
            "seed": 14,
        }

        dtrain = xgb.DMatrix(X[Itrain], y[Itrain], missing=None)
        model = xgb.train(params, dtrain, num_boost_round=num_trees,
                          #early_stopping_rounds=5,
                          evals=[(dtrain, "train"), (dtest, "test")])
        with open(os.path.join(RESULT_DIR, model_name), "wb") as f:
            pickle.dump(model, f)
        #with open(os.path.join(RESULT_DIR, "model.json"), "w") as f:
        #    model.dump_model(f, dump_format="json")
    else:
        print(f"loading model from file: {model_name}")
        with open(os.path.join(RESULT_DIR, model_name), "rb") as f:
            model = pickle.load(f)

    yhat = model.predict(dtest) > 0.5
    acc = sum(y[Itest] != yhat) / X.shape[0]

    return model, acc


# - Optimizer routines --------------------------------------------------------


def get_opt():
    opt = Optimizer(maximize=at)
    opt.enable_smt()

    # hoz. dist to road < 2000, distance to fire source < 1800, not wilderness type 1, soil type 29
    opt.set_smt_program(f"""
(assert (< {opt.xvar(1, 5)} 2000.0))
(assert (< {opt.xvar(1, 9)} 1800.0))
(assert (< {opt.xvar(1, 10)} 0.5))
(assert (> {opt.xvar(1, 42)} 0.5))
    """)
    opt.disable_smt()

    return opt

def astar():
    pass

def arastar():
    pass

def merge():
    pass




# - Robustness for increasing model complexity --------------------------------

for lr, num_trees in [
        (1.0, 1),
        (0.9, 5),
        (0.8, 10),
        (0.75, 20),
        (0.6, 30),
        (0.65, 40),
        (0.6, 50),
        (0.50, 60),
        (0.40, 70),
        (0.35, 80),
        (0.30, 90),
        (0.30, 100),
        ]:

    print(f"\n= num_trees {num_trees}===========")
    model, acc = train_model(lr, num_trees)
    print(f"accuracy: {acc}")
    at = addtree_from_xgb_model(model)
    at.base_score = 0

    print("double check:", util.double_check_at_output(model, at, X[0:100, :]))

    means0 = X[y==0, :].mean(axis=0)
    means1 = X[y==1, :].mean(axis=0)
    for i, (m0, m1) in enumerate(zip(means0, means1)):
        print(i, m0, m1, m1-m0)


    #while opt.num_solutions() < 

    break
