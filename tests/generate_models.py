import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_covtype

#from treeverifier.tree_xgb import additive_tree_from_xgb_model
from treeck.xgb import addtree_from_xgb_model

def generate_california_housing():
    calhouse = fetch_california_housing()

    #print(calhouse["DESCR"])

    X = calhouse["data"]
    y = calhouse["target"]

    # Very Easy
    regr = xgb.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=3,
            learning_rate=1.0,
            n_estimators=2)
    model = regr.fit(X, y)
    at = addtree_from_xgb_model(model)
    sqerr = sum((y - model.predict(X))**2)
    sqcorr = sum((model.predict(X) - at.predict(X))**2)
    print(f"easy: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"easy: rmse model difference {np.sqrt(sqcorr)/len(X)}")

    at.write("tests/models/xgb-calhouse-very-easy.json")

    # Easy
    regr = xgb.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=3,
            learning_rate=0.5,
            n_estimators=10)
    model = regr.fit(X, y)
    at = addtree_from_xgb_model(model)
    sqerr = sum((y - model.predict(X))**2)
    sqcorr = sum((model.predict(X) - at.predict(X))**2)
    print(f"easy: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"easy: rmse model difference {np.sqrt(sqcorr)/len(X)}")

    # edge case test
    feat_id, split_value = at[0].get_split(0)
    Xt = [X[12]]
    Xt[0][feat_id] = split_value
    print("edge case diff: ", model.predict(Xt) - at.predict(Xt))

    at.write("tests/models/xgb-calhouse-easy.json")
    

    # Intermediate
    regr = xgb.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=5,
            learning_rate=0.2,
            n_estimators=20)
    model = regr.fit(X, y)
    at = addtree_from_xgb_model(model)
    sqerr = sum((y - model.predict(X))**2)
    sqcorr = sum((model.predict(X) - at.predict(X))**2)
    print(f"inter: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"inter: rmse model difference {np.sqrt(sqcorr)/len(X)}")

    at.write("tests/models/xgb-calhouse-intermediate.json")


    # Hard
    regr = xgb.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=5,
            learning_rate=0.15,
            n_estimators=100)
    model = regr.fit(X, y)
    at = addtree_from_xgb_model(model)
    sqerr = sum((y - model.predict(X))**2)
    sqcorr = sum((model.predict(X) - at.predict(X))**2)
    print(f"hard: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"hard: rmse model difference {np.sqrt(sqcorr)/len(X)}")

    print(model.predict(X[:10]) - at.predict(X[:10]))

    at.write("tests/models/xgb-calhouse-hard.json")

def generate_covertype():
    covtype = fetch_covtype()

    X = np.array(covtype["data"])
    y = np.array(covtype["target"]) == 2

    # Very Easy
    regr = xgb.XGBClassifier(
            objective="reg:logistic",
            nthread=4,
            tree_method="hist",
            max_depth=4,
            learning_rate=0.5,
            n_estimators=10)
    model = regr.fit(X, y)
    at = addtree_from_xgb_model(model)
    at.base_score = 0.0
    err = sum(y != model.predict(X)) / len(y)
    sqcorr = sum((model.predict(X[:1000], output_margin=True) - at.predict(X[:1000]))**2)
    print(f"easy: error rate {err}")
    print(f"easy: rmse model difference {np.sqrt(sqcorr)/len(X[:1000])}")

    # edge case test
    feat_id, split_value = at[0].get_split(0)
    Xt = [X[12]]
    Xt[0][feat_id] = split_value
    print("edge case diff: ", model.predict(Xt, output_margin=True) - at.predict(Xt))

    at.write("tests/models/xgb-covtype-easy.json")

if __name__ == "__main__":
    #generate_california_housing()
    generate_covertype()
