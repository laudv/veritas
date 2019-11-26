import json
import matplotlib.pyplot as plt
import imageio
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
    print(f"very easy calhouse: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"very easy calhouse: rmse model difference {np.sqrt(sqcorr)/len(X)}")

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
    print(f"easy calhouse: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"easy calhouse: rmse model difference {np.sqrt(sqcorr)/len(X)}")

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
    print(f"inter calhouse: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"inter calhouse: rmse model difference {np.sqrt(sqcorr)/len(X)}")

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
    print(f"hard calhouse: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"hard calhouse: rmse model difference {np.sqrt(sqcorr)/len(X)}")

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
    print(f"easy covtype: error rate {err}")
    print(f"easy covtype: rmse model difference {np.sqrt(sqcorr)/len(X[:1000])}")

    # edge case test
    feat_id, split_value = at[0].get_split(0)
    Xt = [X[12]]
    Xt[0][feat_id] = split_value
    print("edge case diff: ", model.predict(Xt, output_margin=True) - at.predict(Xt))

    at.write("tests/models/xgb-covtype-easy.json")

def generate_img():
    img = imageio.imread("tests/data/img.png")
    X = np.array([[x, y] for x in range(100) for y in range(100)])
    y = np.array([img[x, y] for x, y in X])

    regr = xgb.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=6,
            learning_rate=0.5,
            n_estimators=10)
    model = regr.fit(X, y)
    at = addtree_from_xgb_model(model)
    yhat = model.predict(X)
    sqerr = sum((y - yhat)**2)
    sqcorr = sum((model.predict(X[:1000]) - at.predict(X[:1000]))**2)
    print(f"easy img: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"easy img: rmse model difference {np.sqrt(sqcorr)/len(X[:1000])}")

    #print(model.predict(X[10:20]) - at.predict(X[10:20]))

    fig, ax = plt.subplots(1, 2)
    im0 = ax[0].imshow(img)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(yhat.reshape((100,100)))
    fig.colorbar(im1, ax=ax[1])
    plt.show()

    at.write("tests/models/xgb-img-easy.json")
    with open("tests/models/xgb-img-easy-values.json", "w") as f:
        json.dump(list(map(float, yhat)), f)

if __name__ == "__main__":
    #generate_california_housing()
    #generate_covertype()
    generate_img()
