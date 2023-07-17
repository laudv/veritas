import json
import os
import matplotlib.pyplot as plt
import scipy.io
import imageio
import numpy as np
from veritas.addtree import get_addtree
import xgboost as xgb
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

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
    at = get_addtree(model)
    yhat = model.predict(X)
    sqerr = sum((y - yhat)**2)

    mae = mean_absolute_error(model.predict(X), at.eval(X))
    print(f"very easy img: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"very easy img: mae model difference {mae}")

    fig, ax = plt.subplots(1, 2)
    im0 = ax[0].imshow(img)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(yhat.reshape((100, 100)))
    fig.colorbar(im1, ax=ax[1])
    plt.show()

    at.write("tests/models/xgb-img-very-easy-new.json")
    # with open("tests/models/xgb-img-very-easy-values.json", "w") as f:
    #    json.dump(list(map(float, yhat)), f)

    regr = xgb.XGBRegressor(
        objective="reg:squarederror",
        nthread=4,
        tree_method="hist",
        max_depth=6,
        learning_rate=0.5,
        n_estimators=10)
    model = regr.fit(X, y)
    at = get_addtree(model)
    yhat = model.predict(X)
    sqerr = sum((y - yhat)**2)
    mae = mean_absolute_error(model.predict(X), at.eval(X))
    print(f"easy img: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"easy img: mae model difference {mae}")

    fig, ax = plt.subplots(1, 2)
    im0 = ax[0].imshow(img)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(yhat.reshape((100, 100)))
    fig.colorbar(im1, ax=ax[1])
    plt.show()

    at.write("tests/models/xgb-img-easy-new.json")
    # with open("tests/models/xgb-img-easy-values.json", "w") as f:
    #    json.dump(list(map(float, yhat)), f)

    regr = xgb.XGBRegressor(
        objective="reg:squarederror",
        nthread=4,
        tree_method="hist",
        max_depth=6,
        learning_rate=0.4,
        n_estimators=50)
    model = regr.fit(X, y)
    at = get_addtree(model)
    yhat = model.predict(X)
    sqerr = sum((y - yhat)**2)
    mae = mean_absolute_error(model.predict(X), at.eval(X))
    print(f"hard img: rmse train {np.sqrt(sqerr)/len(X)}")
    print(f"hard img: mae model difference {mae}")

    # print(model.predict(X[10:20]) - at.predict(X[10:20]))

    fig, ax = plt.subplots(1, 2)
    im0 = ax[0].imshow(img)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(yhat.reshape((100, 100)))
    fig.colorbar(im1, ax=ax[1])
    plt.show()

    at.write("tests/models/xgb-img-hard-new.json")
    # with open("tests/models/xgb-img-hard-values.json", "w") as f:
    #    json.dump(list(map(float, yhat)), f)


def generate_img_multiclass():
    img = imageio.imread("tests/data/img.png")
    X = np.array([[x, y] for x in range(100) for y in range(100)])
    y = np.array([img[x, y] for x, y in X])
    yc = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))  # multiclass
    X = X.astype(float)

    regr = xgb.XGBRegressor(
        objective="multi:softmax",
        num_class=4,
        nthread=4,
        tree_method="hist",
        max_depth=6,
        learning_rate=0.5,
        n_estimators=20)
    model = regr.fit(X, yc)
    ats = get_addtree(model)
    yhat = model.predict(X)
    yhatm = model.predict(X, output_margin=True)
    yhatm_at = np.zeros_like(yhatm)
    for k, at in enumerate(ats):
        at.set_base_score(0, 0.5)
        yhatm_at[:, k] = at.eval(X).ravel()
    acc = np.mean(yhat == yc)
    print(f"mult: acc train {acc*100:.1f}%")
    mae = mean_absolute_error(yhatm.ravel(), yhatm_at.ravel())
    print(f"mult: mae model difference {mae}")

    # print(model.predict(X[10:20]) - at.predict(X[10:20]))

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title("actual")
    ax[0].imshow(yc.reshape((100, 100)))
    ax[1].set_title("predicted")
    ax[1].imshow(yhat.reshape((100, 100)))
    ax[2].set_title("errors")
    ax[2].imshow((yc != yhat).reshape((100, 100)))
    fig.suptitle("XGB")

    fig, ax = plt.subplots(1, 5)
    for k in range(4):
        ax[k].set_title(f"prob class {k}")
        ax[k].imshow(yhatm[:, k].reshape((100, 100)))
    ax[4].imshow(yc.reshape((100, 100)))
    ax[4].set_title("actual")
    fig.suptitle("XGB")

    at = ats[0].make_multiclass(0, 4)
    for k in range(1, 4):
        at.add_trees(ats[k], k)

    yhatm_at2 = at.eval(X)
    mae = mean_absolute_error(yhatm.ravel(), yhatm_at2.ravel())
    print(f"mult: multiclass mae model difference {mae}")

    print(at)

    at.write("tests/models/xgb-img-multiclass.json")

    ################

    clf = RandomForestClassifier(max_depth=8, random_state=0, n_estimators=25)
    clf.fit(X, yc)
    yhat = clf.predict(X)
    acc = np.mean(yhat == yc)
    at = get_addtree(clf)
    print("num_leaf_values", at.num_leaf_values())

    print(f"mult: acc train RF {acc*100:.1f}")

    yhatm = clf.predict_proba(X)
    yhatm_at = at.eval(X)
    mae = mean_absolute_error(yhatm.ravel(), yhatm_at.ravel())
    print(f"mult: mae train RF {mae:.2f}")

    fig, ax = plt.subplots(1, 3)
    fig.suptitle("RF")
    ax[0].set_title("actual")
    ax[0].imshow(yc.reshape((100, 100)))
    ax[1].set_title("predicted")
    ax[1].imshow(yhat.reshape((100, 100)))
    ax[2].set_title("errors")
    ax[2].imshow((yc != yhat).reshape((100, 100)))

    fig, ax = plt.subplots(1, 5)
    fig.suptitle("RF")
    for k in range(4):
        ax[k].set_title(f"prob class {k}")
        ax[k].imshow(yhatm[:, k].reshape((100, 100)))
    ax[4].imshow(yc.reshape((100, 100)))
    ax[4].set_title("actual")

    at.write("tests/models/rf-img-multiclass.json")

    plt.show()


if __name__ == "__main__":
    generate_img()
    generate_img_multiclass()
