# Copyright 2024 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import unittest
import imageio.v3 as imageio
import numpy as np

import xgboost as xgb
import lightgbm as lgb
import veritas

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

def get_img_data():
    img = imageio.imread("tests/data/img.png")
    X = np.array([[x, y] for x in range(100) for y in range(100)])
    yr = np.array([img[x, y] for x, y in X]).astype(float)
    y2 = yr > np.median(yr) # binary clf
    y4 = np.digitize(yr, np.quantile(yr, [0.25, 0.5, 0.75])) # multiclass

    # build y for multi-target regression
    yr2 = np.array([img[99-x, 99-y] for x, y in X]).astype(float) # bottom-up
    yr_mt = np.array([[c1, c2, c3] for c1,c2,c3 in zip(yr,yr2,y4)])

    X = X.astype(float) / 100.0

    return X, yr, y2, y4, yr_mt

class TestConverters(unittest.TestCase):
    def test_xgb_binary(self):
        X, _, y, _, _ = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            max_depth=5,
            learning_rate=0.25,
            n_estimators=10)
        model.fit(X, y)
        ypred_model = model.predict_proba(X)[:,1]

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)

        #print(at[0])
        #print(model.get_booster().get_dump()[0])

        #x = X[[2063], :]
        #print("at:", at[0].eval_node(x))
        #print("xgb:", model.apply(x))
        #print("at:", at.eval(x)[0])
        #print("xgb:", model.predict(x, output_margin=True))
        #print("at:", at.eval(x)[0])
        #print("diff:", model.predict(x, output_margin=True)-at.eval(x)[0])
        #print(x)

        #print(f"split_value {at[0].get_split(1).split_value:.16f}")
        #print(f"xvalue      {x[0,0]:.16f}")

        self.assertTrue(is_correct)

    def test_xgb_multiclass(self):
        X, _, _, y, _ = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=4,
            tree_method="hist",
            max_depth=5,
            learning_rate=0.4,
            n_estimators=10)
        model.fit(X, y)
        ypred_model = model.predict_proba(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_multiclass_multioutput(self):
        X, _, _, y, _ = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=4,
            multi_strategy="multi_output_tree",
            tree_method="hist",
            max_depth=5,
            learning_rate=0.25,
            n_estimators=10)
        model.fit(X, y)
        ypred_model = model.predict_proba(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_regression(self):
        X, y, _, _, _ = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            max_depth=5,
            learning_rate=1.0,
            n_estimators=10)
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_regression_with_feature_names(self):
        import pandas as pd

        feature_names = ["feature 1", "feature 2"]
        X, y, _, _, _ = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        df = pd.DataFrame(X, columns=feature_names)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            max_depth=2,
            learning_rate=1.0,
            n_estimators=2)
        model.fit(df, y)
        ypred_model = model.predict(df)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, df, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_regression_multioutput(self):
        X, _, _, _, y = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            multi_strategy="multi_output_tree",
            num_target=y.shape[1],
            tree_method="hist",
            max_depth=5,
            learning_rate=0.5,
            n_estimators=10)
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model, single_rel_tol=1e-4)
        self.assertTrue(is_correct)

    def test_xgb_regression_multioutput_onepertree(self):
        X, _, _, _, y = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            multi_strategy = "one_output_per_tree",
            num_target=y.shape[1],
            tree_method="hist",
            max_depth=5,
            learning_rate=0.5,
            n_estimators=10)
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model, single_rel_tol=1e-4)
        self.assertTrue(is_correct)

    def test_rf_binary(self):
        X, _, y, _, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        clf = RandomForestClassifier(
            max_depth=6,
            random_state=0,
            n_estimators=2)
        clf.fit(X, y)
        ypred_model = clf.predict_proba(X)[:,1]

        at = veritas.get_addtree(clf)
        is_correct = veritas.test_conversion(at, X, ypred_model)

        #print([t.eval_node(X[[4191], :])[0] for t in at])
        #print(clf.apply(X[[4191], :])[0])

        #for t in at:
        #    print(t)

        #import sklearn
        #for t in clf.estimators_:
        #    r = sklearn.tree.export_text(t, feature_names=['F0', 'F1'],
        #                                 decimals=4, spacing=2)
        #    print(r)
        self.assertTrue(is_correct)

    def test_rf_multiclass(self):
        X, _, _, y, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        clf = RandomForestClassifier(
            max_depth=8,
            random_state=0,
            n_estimators=25)
        clf.fit(X, y)
        ypred_model = clf.predict_proba(X)

        at = veritas.get_addtree(clf)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_rf_regression(self):
        X, y, _, _, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        clf = RandomForestRegressor(
            max_depth=8,
            random_state=0,
            n_estimators=25)
        clf.fit(X, y)
        ypred_model = clf.predict(X)

        #import sklearn
        #for x in clf.estimators_:
        #    r = sklearn.tree.export_text(x, feature_names=['F0', 'F1'],
        #                                 decimals=4, spacing=2)
        #    print(r)

        at = veritas.get_addtree(clf)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_rf_regression_multioutput(self):
        X, _, _, _, y = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        model = RandomForestRegressor(
            max_depth=2,
            random_state=0,
            n_estimators=25)
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)


    def test_lgb_binary(self):
        X, _, y, _, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        model = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=64,
            learning_rate=0.25,
            n_estimators=10)
        model.fit(X, y)
        ypred_model = model.predict_proba(X)[:,1]

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_lgb_multiclass(self):
        X, _, _, y, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            num_leaves=64,
            learning_rate=0.25,
            n_estimators=10)
        model.fit(X, y)
        ypred_model = model.predict_proba(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

        print("at base scores", 
              [at.get_base_score(k) for k in range(at.num_leaf_values())])

    def test_lgb_regression(self):
        X, y, _, _, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        model = lgb.LGBMRegressor(
            objective="regression_l2",
            max_depth=5,
            learning_rate=1.0,
            n_estimators=10,
            n_jobs=1)
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

if __name__ == "__main__":
    unittest.main()
