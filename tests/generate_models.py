import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing

#from treeverifier.tree_xgb import additive_tree_from_xgb_model
from treeck.xgb import addtree_from_xgb_model

def generate_california_housing():
    calhouse = fetch_california_housing()

    #print(calhouse["DESCR"])

    X = calhouse["data"]
    y = calhouse["target"]

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
    sqerr = sum((y - model.predict(X))**2) / len(y)
    sqcorr = sum((model.predict(X) - at.predict(X))**2)
    print(f"easy: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"easy: rmse model difference {np.sqrt(sqcorr)/len(X)}")

    # edge case test
    Xt = [X[12]]
    Xt[0][at[0][0].get_split().feat_id] = at[0][0].get_split().split_value
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
    sqerr = sum((y - model.predict(X))**2) / len(y)
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
    sqerr = sum((y - model.predict(X))**2) / len(y)
    sqcorr = sum((model.predict(X) - at.predict(X))**2)
    print(f"hard: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"hard: rmse model difference {np.sqrt(sqcorr)/len(X)}")

    print(model.predict(X[:10]) - at.predict(X[:10]))

    at.write("tests/models/xgb-calhouse-hard.json")


if __name__ == "__main__":
    generate_california_housing()
