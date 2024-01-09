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
    X = X.astype(float)

    return X, yr, y2, y4

class TestConverters(unittest.TestCase):
    def test_xgb_binary(self):
        X, _, y, _ = get_img_data()
        X = X / 100.0
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
        self.assertTrue(is_correct)

    def test_xgb_multiclass(self):
        X, _, _, y = get_img_data()
        X = X / 100.0
        model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=4,
            tree_method="hist",
            max_depth=5,
            learning_rate=0.4,
            n_estimators=5)
        model.fit(X, y)
        ypred_model = model.predict_proba(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_multiclass_multioutput(self):
        X, _, _, y = get_img_data()
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
        X, y, _, _ = get_img_data()
        X /= 100.0
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
        X, y, _, _ = get_img_data()
        X /= 100.0
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

    def test_rf_binary(self):
        X, _, y, _ = get_img_data()
        clf = RandomForestClassifier(
            max_depth=8,
            random_state=0,
            n_estimators=25)
        clf.fit(X, y)
        ypred_model = clf.predict_proba(X)[:,1]

        at = veritas.get_addtree(clf)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_rf_multiclass(self):
        X, _, _, y = get_img_data()
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
        X, y, _, _ = get_img_data()
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

    def test_lgb_binary(self):
        X, _, y, _ = get_img_data()
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
        X, _, _, y = get_img_data()
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
        X, y, _, _ = get_img_data()
        X /= 100.0
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
