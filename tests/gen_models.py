import unittest
import matplotlib.pyplot as plt
import imageio
import numpy as np
from veritas.addtree import get_addtree
import xgboost as xgb
import lightgbm as lgbm
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from veritas import *

import timeit


class TestAddTree(unittest.TestCase):

    def test_regression(self):
        img = imageio.imread("tests/data/img.png")
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])

        X = X.astype(float)

        print("############# Regression Tests #############")

        ############# XGB #############
        print("XGB - Regression:")
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

        mae = mean_absolute_error(yhat, at.predict(X))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {np.sqrt(sqerr)/len(X)}")
        print(f"very easy img: mae model difference {mae}")

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

        mae = mean_absolute_error(yhat, at.predict(X))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {np.sqrt(sqerr)/len(X)}")
        print(f"very easy img: mae model difference {mae}")

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

        mae = mean_absolute_error(yhat, at.predict(X))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {np.sqrt(sqerr)/len(X)}")
        print(f"very easy img: mae model difference {mae}")

        at.write("tests/models/xgb-img-hard-new.json")
        print()
        # with open("tests/models/xgb-img-hard-values.json", "w") as f:
        #    json.dump(list(map(float, yhat)), f)

        ############# SkLearn #############
        print("SkLearn - Regression:")
        clf = RandomForestRegressor(
            max_depth=6,
            random_state=0,
            n_estimators=50)
        model = clf.fit(X, y)
        yhat = model.predict(X)
        at = get_addtree(model)
        yhat = model.predict(X)
        sqerr = sum((y - yhat)**2)

        mae = mean_absolute_error(yhat, at.predict(X))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {np.sqrt(sqerr)/len(X)}")
        print(f"very easy img: mae model difference {mae}")

        at.write("tests/models/sklearn-img-hard-new.json")
        print()

        ############# LGBM #############
        print("LGBM - Regression:")
        regr = lgbm.LGBMRegressor(
            objective="regression",
            num_leaves=10,
            nthread=4,
            max_depth=3,
            learning_rate=1,
            n_estimators=3,
            verbose=-1)
        model = regr.fit(X, y)
        at = get_addtree(model)
        yhat = model.predict(X)
        sqerr = sum((y - yhat)**2)

        mae = mean_absolute_error(yhat, at.predict(X))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {np.sqrt(sqerr)/len(X)}")
        print(f"very easy img: mae model difference {mae}")

        at.write("tests/models/lgbm-img-very-easy-new.json")

        regr = lgbm.LGBMRegressor(
            objective="regression",
            nthread=4,
            num_leaves=65,
            max_depth=6,
            learning_rate=0.5,
            n_estimators=10,
            verbose=-1)
        model = regr.fit(X, y)
        at = get_addtree(model)
        yhat = model.predict(X)
        sqerr = sum((y - yhat)**2)

        mae = mean_absolute_error(yhat, at.predict(X))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {np.sqrt(sqerr)/len(X)}")
        print(f"very easy img: mae model difference {mae}")

        at.write("tests/models/lgbm-img-easy-new.json")
        # with open("tests/models/xgb-img-easy-values.json", "w") as f:
        #    json.dump(list(map(float, yhat)), f)

        regr = lgbm.LGBMRegressor(
            objective="regression",
            nthread=4,
            num_leaves=65,
            max_depth=6,
            learning_rate=0.4,
            n_estimators=50,
            verbose=-1)
        model = regr.fit(X, y)
        at = get_addtree(model)
        yhat = model.predict(X)
        sqerr = sum((y - yhat)**2)

        mae = mean_absolute_error(yhat, at.predict(X))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {np.sqrt(sqerr)/len(X)}")
        print(f"very easy img: mae model difference {mae}")

        at.write("tests/models/lgbm-img-hard-new.json")
        print()

    def test_multiclass(self):
        full_set = load_digits()
        X = full_set.data
        y = full_set.target

        print("############# Multiclass Tests #############")

        # ############# XGB #############
        print("XGB - Multiclass:")
        clf = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=10,
            nthread=4,
            tree_method="hist",
            max_depth=6,
            learning_rate=0.5,
            n_estimators=20)
        model = clf.fit(X, y)
        ats = get_addtree(model)
        yhat = model.predict(X)
        yhatm = model.predict(X, output_margin=True)

        yhatm_at = np.zeros_like(yhatm)
        for k, at in enumerate(ats):
            yhatm_at[:, k] = at.predict(X).ravel()

        acc = np.mean(yhat == y)
        print(f"multi: acc train {acc*100:.1f}%")
        mae = mean_absolute_error(yhatm.ravel(), yhatm_at.ravel())
        print(f"multi: mae model difference {mae}")
        print()

        # ############# SkLearn #############
        print("SkLearn - Multiclass:")
        clf = RandomForestClassifier(
            max_depth=8, random_state=0, n_estimators=20)
        model = clf.fit(X, y)

        yhat = clf.predict(X)
        acc = np.mean(yhat == y)
        at = get_addtree(clf)
        print(f"mult: acc train RF {acc*100:.1f}%")

        yhatm = clf.predict_proba(X)
        yhatm_at = at.eval(X)
        mae = mean_absolute_error(yhatm.ravel(), yhatm_at.ravel())
        print(f"mult: mae train RF {mae:.2f}")
        print()

        # ############# LGBM #############
        print("LGBM - Multiclass:")
        clf = lgbm.LGBMClassifier(
            objective="multiclass",
            num_leaves=31,
            nthread=4,
            max_depth=3,
            learning_rate=0.5,
            n_estimators=20,
            verbose=-1)
        model = clf.fit(X, y)
        ats = get_addtree(model)
        yhat = model.predict(X)
        yhatm = model.predict_proba(X, raw_score=True)

        yhatm_at = np.zeros_like(yhatm)
        for k, at in enumerate(ats):
            yhatm_at[:, k] = at.predict(X).ravel()

        acc = np.mean(yhat == y)
        print(f"multi: acc train {acc*100:.1f}%")
        mae = mean_absolute_error(yhatm.ravel(), yhatm_at.ravel())
        print(f"multi: mae model difference {mae}")
        print()
        # img = imageio.imread("tests/data/img.png")
        # X = np.array([[x, y] for x in range(100) for y in range(100)])
        # y = np.array([img[x, y] for x, y in X])
        # yc = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))  # multiclass
        # X = X.astype(float)

        # print("############# Multiclass Tests #############")

        # ############# XGB #############
        # print("XGB - Multi Classification:")
        # clf = xgb.XGBClassifier(
        #     objective="multi:softmax",
        #     num_class=4,
        #     nthread=4,
        #     tree_method="hist",
        #     max_depth=6,
        #     learning_rate=0.5,
        #     n_estimators=20)
        # model = clf.fit(X, yc)
        # ats = get_addtree(model)
        # yhat = model.predict(X)
        # yhatm = model.predict(X, output_margin=True)
        # yhatm_at = np.zeros_like(yhatm)
        # for k, at in enumerate(ats):
        #     at.set_base_score(0, 0.5)
        #     yhatm_at[:, k] = at.eval(X).ravel()
        # acc = np.mean(yhat == yc)
        # print(f"mult: acc train {acc*100:.1f}%")
        # mae = mean_absolute_error(yhatm.ravel(), yhatm_at.ravel())
        # print(f"mult: mae model difference {mae}")

        # # print(model.predict(X[10:20]) - at.predict(X[10:20]))

        # fig, ax = plt.subplots(1, 3)
        # ax[0].set_title("actual")
        # ax[0].imshow(yc.reshape((100, 100)))
        # ax[1].set_title("predicted")
        # ax[1].imshow(yhat.reshape((100, 100)))
        # ax[2].set_title("errors")
        # ax[2].imshow((yc != yhat).reshape((100, 100)))
        # fig.suptitle("XGB")

        # fig, ax = plt.subplots(1, 5)
        # for k in range(4):
        #     ax[k].set_title(f"prob class {k}")
        #     ax[k].imshow(yhatm[:, k].reshape((100, 100)))
        # ax[4].imshow(yc.reshape((100, 100)))
        # ax[4].set_title("actual")
        # fig.suptitle("XGB")

        # at = ats[0].make_multiclass(0, 4)
        # for k in range(1, 4):
        #     at.add_trees(ats[k], k)

        # yhatm_at2 = at.eval(X)
        # mae = mean_absolute_error(yhatm.ravel(), yhatm_at2.ravel())
        # print(f"mult: multiclass mae model difference {mae}")

        # print(at)

        # at.write("tests/models/xgb-img-multiclass.json")
        # print()

        # ############# SkLearn #############
        # print("SkLearn - Multi Classification:")
        # clf = RandomForestClassifier(max_depth=8, random_state=0, n_estimators=25)
        # clf.fit(X, yc)
        # yhat = clf.predict(X)
        # acc = np.mean(yhat == yc)
        # at = get_addtree(clf)
        # print("num_leaf_values", at.num_leaf_values())

        # print(f"mult: acc train RF {acc*100:.1f}")

        # yhatm = clf.predict_proba(X)
        # yhatm_at = at.eval(X)
        # mae = mean_absolute_error(yhatm.ravel(), yhatm_at.ravel())
        # print(f"mult: mae train RF {mae:.2f}")

        # fig, ax = plt.subplots(1, 3)
        # fig.suptitle("RF")
        # ax[0].set_title("actual")
        # ax[0].imshow(yc.reshape((100, 100)))
        # ax[1].set_title("predicted")
        # ax[1].imshow(yhat.reshape((100, 100)))
        # ax[2].set_title("errors")
        # ax[2].imshow((yc != yhat).reshape((100, 100)))

        # fig, ax = plt.subplots(1, 5)
        # fig.suptitle("RF")
        # for k in range(4):
        #     ax[k].set_title(f"prob class {k}")
        #     ax[k].imshow(yhatm[:, k].reshape((100, 100)))
        # ax[4].imshow(yc.reshape((100, 100)))
        # ax[4].set_title("actual")

        # at.write("tests/models/rf-img-multiclass.json")

        # plt.show()
        # print()

        # ############# LGBM #############
        # # print("LGBM - Multi Classification:")
        # # clf = lgbm.LGBMClassifier(
        # #     objective="multiclass",
        # #     num_class=4,
        # #     nthread=4,
        # #     max_depth=6,
        # #     learning_rate=0.5,
        # #     n_estimators=20,
        # #     verbose=-1)
        # # model = clf.fit(X, yc)
        # # ats = get_addtree(model)
        # # yhat = model.predict(X)
        # # yhatm = model.predict(X, output_margin=True)
        # # yhatm_at = np.zeros_like(yhatm)
        # # for k, at in enumerate(ats):
        # #     yhatm_at[:, k] = at.eval(X).ravel()
        # # acc = np.mean(yhat == yc)
        # # print(f"mult: acc train {acc*100:.1f}%")
        # # mae = mean_absolute_error(yhatm.ravel(), yhatm_at.ravel())
        # # print(f"mult: mae model difference {mae}")

        # # fig, ax = plt.subplots(1, 3)
        # # ax[0].set_title("actual")
        # # ax[0].imshow(yc.reshape((100, 100)))
        # # ax[1].set_title("predicted")
        # # ax[1].imshow(yhat.reshape((100, 100)))
        # # ax[2].set_title("errors")
        # # ax[2].imshow((yc != yhat).reshape((100, 100)))
        # # fig.suptitle("LGBM")

        # # fig, ax = plt.subplots(1, 5)
        # # for k in range(4):
        # #     ax[k].set_title(f"prob class {k}")
        # #     ax[k].imshow(yhatm[:, k].reshape((100, 100)))
        # # ax[4].imshow(yc.reshape((100, 100)))
        # # ax[4].set_title("actual")
        # # fig.suptitle("LGBM")

        # # at = ats[0].make_multiclass(0, 4)
        # # for k in range(1, 4):
        # #     at.add_trees(ats[k], k)

        # # yhatm_at2 = at.eval(X)
        # # mae = mean_absolute_error(yhatm.ravel(), yhatm_at2.ravel())
        # # print(f"mult: multiclass mae model difference {mae}")

        # # at.write("tests/models/xgb-img-multiclass.json")
        # # print()

    def test_binary_classification(self):

        full_set = load_breast_cancer()
        X = full_set.data
        y = full_set.target

        print("############# Binary Classification Tests #############")

        ############# XGB #############
        print("XGB - Binary Classification:")
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            nthread=4,
            tree_method="hist",
            max_depth=4,
            learning_rate=1,
            n_estimators=3)
        model = clf.fit(X, y)
        at = get_addtree(model)

        err = sum(y != model.predict(X)) / len(y)
        mae = mean_absolute_error(model.predict(
            X, output_margin=True), at.predict(X))

        print(f"easy bc: error rate {err}")
        print(f"easy bc: mae model difference {mae}")
        self.assertAlmostEqual(mae, 0.0, delta=1e-2)  # Good enough?
        at.write("tests/models/xgb-bc-easy.json")
        print()

        ############# SkLearn #############
        print("SkLearn - Binary Classification:")
        clf = RandomForestClassifier(
            max_depth=6,
            random_state=0,
            n_estimators=50,)
        model = clf.fit(X, y)
        at = get_addtree(model)

        err = sum(y != model.predict(X)) / len(y)
        mae = mean_absolute_error(model.predict_proba(
            X), at.predict(X))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)
        print(f"easy bc: error rate {err}")
        print(f"easy bc: mae model difference {mae}")

        at.write("tests/models/sklearn-bc-easy.json")
        print()

        ############# LGBM #############
        print("LGBM - Binary Classification:")
        clf = lgbm.LGBMClassifier(
            objective="binary",
            num_leaves=31,
            nthread=4,
            max_depth=3,
            learning_rate=1,
            n_estimators=3,
            verbose=-1)
        model = clf.fit(X, y)
        at = get_addtree(model)

        err = sum(y != model.predict(X)) / len(y)
        mae = mean_absolute_error(model.predict(
            X, raw_score=True), at.predict(X))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)
        print(f"easy bc: error rate {err}")
        print(f"easy bc: mae model difference {mae}")
        at.write("tests/models/lgbm-bc-easy.json")
        print()


if __name__ == "__main__":
    unittest.main()
    # test_regression()
    # test_binary_classification()
    # test_multiclass()
