import json, os
import matplotlib.pyplot as plt
import scipy.io
import imageio
import numpy as np
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_openml


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
    clf = xgb.XGBClassifier(
            objective="reg:logistic",
            nthread=4,
            tree_method="hist",
            max_depth=4,
            learning_rate=0.5,
            n_estimators=10)
    model = clf.fit(X, y)
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

def generate_mnist():
    if not os.path.exists("tests/data/mnist.mat"):
        print("loading MNIST with fetch_openml")
        mnist = fetch_openml("mnist_784")
        X = mnist["data"]
        y = np.array(list(map(lambda v: int(v), mnist["target"])))
        scipy.io.savemat("tests/data/mnist.mat", {"X": X, "y": y},
                do_compression=True, format="5")
    else:
        print("loading MNIST MAT file")
        mat = scipy.io.loadmat("tests/data/mnist.mat") # much faster
        X = mat["X"]
        y = mat["y"].reshape((70000,))


    print("Training MNIST17")
    m17 = np.logical_or(y==1, y==7)
    X17 = X[m17]
    y17 = y[m17]==1

    clf = xgb.XGBClassifier(
            nthread=4,
            tree_method="hist",
            max_depth=4,
            learning_rate=0.5,
            n_estimators=10)
    model = clf.fit(X17, y17)
    at = addtree_from_xgb_model(model)
    at.base_score = 0.0
    acc = accuracy_score(model.predict(X17), y17)
    print(f"mnist17: accuracy 17: {acc}")
    mae = mean_absolute_error(model.predict(X17, output_margin=True), at.predict(X17))
    print(f"mnist17: mae model difference {mae}")
    at.write("tests/models/xgb-mnist17-easy.json")

    print("Training MNIST08")
    m08 = np.logical_or(y==0, y==8)
    X08 = X[m08]
    y08 = y[m08]==0

    clf = xgb.XGBClassifier(
            nthread=4,
            tree_method="hist",
            max_depth=4,
            learning_rate=0.5,
            n_estimators=10)
    model = clf.fit(X08, y08)
    at = addtree_from_xgb_model(model)
    at.base_score = 0.0
    acc = accuracy_score(model.predict(X08), y08)
    print(f"mnist08: accuracy 08: {acc}")
    mae = mean_absolute_error(model.predict(X08, output_margin=True), at.predict(X08))
    print(f"mnist08: mae model difference {mae}")
    at.write("tests/models/xgb-mnist08-easy.json")

    print("Exporting instances")
    with open("tests/models/mnist-instances.json", "w") as f:
        d = {}
        for j, v in enumerate(y):
            if v not in d.keys():
                d[v] = X[j]
                #print(f"v={v}")
                #plt.imshow(X[j].reshape((28, 28)))
                #plt.show()
        dd = {}
        for (k, v) in d.items():
            dd[str(k)] = list(v)
        json.dump(dd, f)

if __name__ == "__main__":
    #generate_california_housing()
    #generate_covertype()
    #generate_img()
    generate_mnist()
