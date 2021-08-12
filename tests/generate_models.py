import json, os
import matplotlib.pyplot as plt
import scipy.io
import imageio
import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_openml

from veritas import *

import timeit

def generate_img():
    img = imageio.imread("tests/data/img.png")
    X = np.array([[x, y] for x in range(100) for y in range(100)])
    y = np.array([img[x, y] for x, y in X])

    X = X.astype(float)

    regr = xgb.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=3,
            learning_rate=1.0,
            n_estimators=3)
    model = regr.fit(X, y)
    at = addtree_from_xgb_model(model)
    yhat = model.predict(X)
    sqerr = sum((y - yhat)**2)

    mae = mean_absolute_error(model.predict(X), at.eval(X))
    print(f"very easy img: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"very easy img: mae model difference {mae}")

    fig, ax = plt.subplots(1, 2)
    im0 = ax[0].imshow(img)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(yhat.reshape((100,100)))
    fig.colorbar(im1, ax=ax[1])
    plt.show()

    #at.write("tests/models/xgb-img-very-easy.json")
    #with open("tests/models/xgb-img-very-easy-values.json", "w") as f:
    #    json.dump(list(map(float, yhat)), f)

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
    mae = mean_absolute_error(model.predict(X), at.eval(X))
    print(f"easy img: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"easy img: mae model difference {mae}")

    fig, ax = plt.subplots(1, 2)
    im0 = ax[0].imshow(img)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(yhat.reshape((100,100)))
    fig.colorbar(im1, ax=ax[1])
    plt.show()

    #at.write("tests/models/xgb-img-easy.json")
    #with open("tests/models/xgb-img-easy-values.json", "w") as f:
    #    json.dump(list(map(float, yhat)), f)

    regr = xgb.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=6,
            learning_rate=0.4,
            n_estimators=50)
    model = regr.fit(X, y)
    at = addtree_from_xgb_model(model)
    yhat = model.predict(X)
    sqerr = sum((y - yhat)**2)
    mae = mean_absolute_error(model.predict(X), at.eval(X))
    print(f"hard img: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"hard img: mae model difference {mae}")

    #print(model.predict(X[10:20]) - at.predict(X[10:20]))

    fig, ax = plt.subplots(1, 2)
    im0 = ax[0].imshow(img)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(yhat.reshape((100,100)))
    fig.colorbar(im1, ax=ax[1])
    plt.show()

    #at.write("tests/models/xgb-img-hard.json")
    #with open("tests/models/xgb-img-hard-values.json", "w") as f:
    #    json.dump(list(map(float, yhat)), f)

def generate_allstate():
    allstate_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "allstate.h5")
    data = pd.read_hdf(allstate_data_path)
    X = data.drop(columns=["loss"]).to_numpy()
    y = data.loss.to_numpy()

    max_depth, num_trees = 5, 100
    print(f"training model depth={max_depth}, num_trees={num_trees}")

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "seed": 14,
        "nthread": 4,
        "max_depth": max_depth
    }

    dtrain = xgb.DMatrix(X, y, missing=None)
    model = xgb.train(params, dtrain, num_boost_round=num_trees,
                      evals=[(dtrain, "train")])
    at = addtree_from_xgb_model(model)
    mae = sum((model.predict(dtrain) - at.eval(X))**2)/len(X)
    print(f"allstate: mae model difference {mae}")
    at.write("tests/models/xgb-allstate.json", compress=False)

#def generate_california_housing():
#    calhouse = fetch_california_housing()
#
#    #print(calhouse["DESCR"])
#
#    X = calhouse["data"]
#    y = calhouse["target"]
#
#    # Very Easy
#    regr = xgb.XGBRegressor(
#            objective="reg:squarederror",
#            nthread=4,
#            tree_method="hist",
#            max_depth=3,
#            learning_rate=1.0,
#            n_estimators=2)
#    model = regr.fit(X, y)
#    at = addtree_from_xgb_model(model)
#    sqerr = sum((y - model.predict(X))**2)
#    mae = mean_absolute_error(model.predict(X), at.predict(X))
#    print(f"very easy calhouse: rmse train {np.sqrt(sqerr)/len(X)}")
#    print(f"very easy calhouse: mae model difference {mae}")
#
#    at.write("tests/models/xgb-calhouse-very-easy.json")
#
#    # Easy
#    regr = xgb.XGBRegressor(
#            objective="reg:squarederror",
#            nthread=4,
#            tree_method="hist",
#            max_depth=3,
#            learning_rate=0.5,
#            n_estimators=10)
#    model = regr.fit(X, y)
#    at = addtree_from_xgb_model(model)
#    sqerr = sum((y - model.predict(X))**2)
#    mae = mean_absolute_error(model.predict(X), at.predict(X))
#    print(f"easy calhouse: rmse train {np.sqrt(sqerr)/len(X)}")
#    print(f"easy calhouse: mae model difference {mae}")
#
#    # edge case test
#    split = at[0].get_split(0)
#    Xt = np.array([X[12]])
#    Xt[0][split.feat_id] = split.split_value
#    print("edge case diff: ", model.predict(Xt) - at.predict(Xt))
#
#    at.write("tests/models/xgb-calhouse-easy.json")
#    
#
#    # Intermediate
#    regr = xgb.XGBRegressor(
#            objective="reg:squarederror",
#            nthread=4,
#            tree_method="hist",
#            max_depth=5,
#            learning_rate=0.2,
#            n_estimators=20)
#    model = regr.fit(X, y)
#    at = addtree_from_xgb_model(model)
#    sqerr = sum((y - model.predict(X))**2)
#    mae = mean_absolute_error(model.predict(X), at.predict(X))
#    print(f"inter calhouse: rmse train {np.sqrt(sqerr)/len(X)}")
#    print(f"inter calhouse: mae model difference {mae}")
#
#    at.write("tests/models/xgb-calhouse-intermediate.json")
#
#
#    # Hard
#    regr = xgb.XGBRegressor(
#            objective="reg:squarederror",
#            nthread=4,
#            tree_method="hist",
#            max_depth=5,
#            learning_rate=0.15,
#            n_estimators=100)
#    model = regr.fit(X, y)
#    at = addtree_from_xgb_model(model)
#    sqerr = sum((y - model.predict(X))**2)
#    mae = mean_absolute_error(model.predict(X), at.predict(X))
#    print(f"hard calhouse: rmse train {np.sqrt(sqerr)/len(X)}")
#    print(f"hard calhouse: mae model difference {mae}")
#
#    print(model.predict(X[:10]) - at.predict(X[:10]))
#
#    at.write("tests/models/xgb-calhouse-hard.json")
#
#def generate_covertype():
#    covtype = fetch_covtype()
#
#    X = np.array(covtype["data"], dtype=float)
#    y = np.array(covtype["target"]) == 2
#
#    # Very Easy
#    clf = xgb.XGBClassifier(
#            objective="reg:logistic",
#            nthread=4,
#            tree_method="hist",
#            max_depth=4,
#            learning_rate=0.5,
#            n_estimators=10)
#    model = clf.fit(X, y)
#    at = addtree_from_xgb_model(model)
#    at.base_score = 0.0
#    err = sum(y != model.predict(X)) / len(y)
#    mae = mean_absolute_error(model.predict(X[:1000], output_margin=True), at.predict(X[:1000]))
#    print(f"easy covtype: error rate {err}")
#    print(f"easy covtype: mae model difference {mae}")
#
#    # edge case test
#    split = at[0].get_split(0)
#    Xt = np.array([X[12]])
#    Xt[0][split.feat_id] = split.split_value
#    print("edge case diff: ", model.predict(Xt, output_margin=True) - at.predict(Xt))
#
#    at.write("tests/models/xgb-covtype-easy.json")
#
#
#def generate_mnist():
#    if not os.path.exists("tests/data/mnist.mat"):
#        print("loading MNIST with fetch_openml")
#        mnist = fetch_openml("mnist_784")
#        X = mnist["data"]
#        y = np.array(list(map(lambda v: int(v), mnist["target"])))
#        scipy.io.savemat("tests/data/mnist.mat", {"X": X, "y": y},
#                do_compression=True, format="5")
#    else:
#        print("loading MNIST MAT file")
#        mat = scipy.io.loadmat("tests/data/mnist.mat") # much faster
#        X = mat["X"]
#        y = mat["y"].reshape((70000,))
#
#
#    print("Training MNIST y==1")
#    y1 = y==1
#    clf = xgb.XGBClassifier(
#            nthread=4,
#            tree_method="hist",
#            max_depth=4,
#            learning_rate=0.5,
#            n_estimators=10)
#    model = clf.fit(X, y1)
#    at = addtree_from_xgb_model(model)
#    at.base_score = 0.0
#    acc = accuracy_score(model.predict(X), y1)
#    print(f"mnist y==1: accuracy y==1: {acc}")
#    mae = mean_absolute_error(model.predict(X[:5000], output_margin=True), at.predict(X[:5000]))
#    print(f"mnist y==1: mae model difference {mae}")
#    at.write("tests/models/xgb-mnist-yis1-easy.json")
#
#    print("Training MNIST y==0")
#    y0 = y==0
#    clf = xgb.XGBClassifier(
#            nthread=4,
#            tree_method="hist",
#            max_depth=4,
#            learning_rate=0.5,
#            n_estimators=10)
#    model = clf.fit(X, y0)
#    at = addtree_from_xgb_model(model)
#    at.base_score = 0.0
#    acc = accuracy_score(model.predict(X), y0)
#    print(f"mnist y==0: accuracy y==0: {acc}")
#    mae = mean_absolute_error(model.predict(X[:5000], output_margin=True), at.predict(X[:5000]))
#    print(f"mnist y==0: mae model difference {mae}")
#    at.write("tests/models/xgb-mnist-yis0-easy.json")
#
#    print("Training MNIST y==0 hard")
#    y0 = y==0
#    clf = xgb.XGBClassifier(
#            nthread=4,
#            tree_method="hist",
#            max_depth=5,
#            learning_rate=0.2,
#            n_estimators=100)
#    model = clf.fit(X, y0)
#    at = addtree_from_xgb_model(model)
#    at.base_score = 0.0
#    acc = accuracy_score(model.predict(X), y0)
#    print(f"mnist y==0: accuracy y==0: {acc}")
#    mae = mean_absolute_error(model.predict(X[:5000], output_margin=True), at.predict(X[:5000]))
#    print(f"mnist y==0: mae model difference {mae}")
#    at.write("tests/models/xgb-mnist-yis0-hard.json")
#
#    print("Exporting instances")
#    with open("tests/models/mnist-instances.json", "w") as f:
#        d = {}
#        for j, v in enumerate(y):
#            if v not in d.keys():
#                d[v] = X[j]
#                #print(f"v={v}")
#                #plt.imshow(X[j].reshape((28, 28)))
#                #plt.show()
#        dd = {}
#        for (k, v) in d.items():
#            dd[str(k)] = list(v)
#        json.dump(dd, f)
#
#def generate_bin_mnist():
#    print("loading MNIST MAT file")
#    mat = scipy.io.loadmat("tests/data/mnist.mat") # much faster
#    X = mat["X"] > 50.0
#    pdX = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]) # xgboost produces binary splits
#    print(pdX.dtypes)
#    y = mat["y"].reshape((70000,))
#
#    print("Training MNIST y==1")
#    y1 = y==1
#    print(type(y1), y1.dtype)
#    clf = xgb.XGBClassifier(
#            nthread=4,
#            tree_method="hist",
#            max_depth=3,
#            learning_rate=0.4,
#            n_estimators=1)
#    model = clf.fit(pdX, y1)
#    pred = model.predict(pdX, output_margin=True)
#    at = addtree_from_xgb_model(model) # broken: https://github.com/dmlc/xgboost/issues/5655
#    at.base_score = 0.0
#    acc = accuracy_score(pred > 0.0, y1)
#    print(f"mnist y==1: accuracy y==1: {acc}")
#    mae = mean_absolute_error(pred[:5000], at.predict(X[:5000]))
#    print(f"mnist y==1: mae model difference {mae}")
#    at.write("tests/models/xgb-mnist-bin-yis1-intermediate.json")


if __name__ == "__main__":
    generate_img()
    #generate_allstate()
    #generate_california_housing()
    #generate_covertype()
    #generate_mnist()
    #generate_bin_mnist()
