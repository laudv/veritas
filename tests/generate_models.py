import matplotlib.pyplot as plt
import imageio.v3 as imageio
import numpy as np
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

import veritas

def get_img_data():
    img = imageio.imread("tests/data/img.png")
    X = np.array([[x, y] for x in range(100) for y in range(100)])
    yr = np.array([img[x, y] for x, y in X]).astype(float)
    y2 = yr > np.median(yr) # binary clf
    y4 = np.digitize(yr, np.quantile(yr, [0.25, 0.5, 0.75])) # multiclass
    X = X.astype(float)

    return img, X, yr, y2, y4

def generate_img_regression():
    img, X, y, _, _ = get_img_data()

    params = {
        "xgb-img-very-easy-new": {
            "max_depth": 3,
            "learning_rate": 1.0,
            "n_estimators": 3
        },
        "xgb-img-easy-new": {
            "max_depth": 6,
            "learning_rate": 0.5,
            "n_estimators": 10
        },
        "xgb-img-hard-new": {
            "max_depth": 6,
            "learning_rate": 0.25,
            "n_estimators": 40
        },
    }

    for k, ps in params.items():
        print("TRAIN", k)
        regr = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            **ps)
        model = regr.fit(X, y)
        at = veritas.get_addtree(model)
        yhat_model = model.predict(X)
        sqerr = sum((y - yhat_model)**2)

        veritas.test_conversion(at, X, yhat_model)

        print(f"{k}: rmse train {np.sqrt(sqerr)/len(X)}")

        fig, ax = plt.subplots(1, 2)
        im0 = ax[0].imshow(img)
        fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(yhat_model.reshape((100, 100)))
        fig.colorbar(im1, ax=ax[1])
        plt.show()

        at.write(f"tests/models/{k}.json")
        # with open("tests/models/xgb-img-very-easy-values.json", "w") as f:
        #    json.dump(list(map(float, yhat)), f)

def generate_img_multiclass():
    img, X, _, _, y = get_img_data()

    clfs = {
        "xgb-img-multiclass": xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=4,
            tree_method="hist",
            max_depth=6,
            learning_rate=0.5,
            n_estimators=20),

        "xgb-img-multiclass-multivalue": xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=4,
            multi_strategy="multi_output_tree",
            tree_method="hist",
            max_depth=6,
            learning_rate=0.5,
            n_estimators=20),

        "rf-img-multiclass": RandomForestClassifier(
            max_depth=8,
            random_state=0,
            n_estimators=25)
    }

    for k, model in clfs.items(): 
        print("TRAIN MULT", k)
        model.fit(X, y)
        at = veritas.get_addtree(model)
        yhat_model = model.predict(X)
        ypred_model = model.predict_proba(X)

        acc = np.mean(yhat_model == y)
        print(f"{k} mult: acc train {acc*100:.1f}%")

        veritas.test_conversion(at, X, ypred_model)

        fig, ax = plt.subplots(1, 3)
        ax[0].set_title("actual")
        ax[0].imshow(y.reshape((100, 100)))
        ax[1].set_title("predicted")
        ax[1].imshow(yhat_model.reshape((100, 100)))
        ax[2].set_title("errors")
        ax[2].imshow((y != yhat_model).reshape((100, 100)))
        fig.suptitle(k)
        plt.show()

        at.write(f"tests/models/{k}.json")

if __name__ == "__main__":
    generate_img_regression()
    generate_img_multiclass()
