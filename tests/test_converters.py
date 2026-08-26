# Copyright 2024 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import unittest

import imageio.v3 as imageio
import numpy as np
import pytest
import veritas

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import sklearn.ensemble as skensemble
    import sklearn.tree as sktree
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except ImportError:
    RandomForestClassifier = None
    RandomForestRegressor = None
    skensemble = None
    sktree = None


def get_img_data():
    img = imageio.imread("tests/data/img.png")
    X = np.array([[x, y] for x in range(100) for y in range(100)])
    yr = np.array([img[x, y] for x, y in X]).astype(float)
    y2 = yr > np.median(yr)  # binary clf
    y4 = np.digitize(yr, np.quantile(yr, [0.25, 0.5, 0.75]))  # multiclass

    # build y for multi-target regression
    yr2 = np.array([img[99 - x, 99 - y] for x, y in X]).astype(float)  # bottom-up
    yr_mt = np.array([[c1, c2, c3] for c1, c2, c3 in zip(yr, yr2, y4)])

    X = X.astype(float) / 100.0

    return X, yr, y2, y4, yr_mt


def get_missing_data(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, (200, 4))

    # Downscale X's precision because tree packages might internally use float32
    X = X.astype(np.float32).astype(np.float64)

    mask = rng.choice([True, False], size=X.shape, p=[0.2, 0.8])
    X[mask] = np.nan

    y = np.sum(np.nan_to_num(X), axis=1) + rng.normal(0, 1, 200).astype(np.float32)
    # Binary classification labels
    y_bin = (y > np.median(y)).astype(int)
    # Multiclass labels
    y_multi = np.digitize(y, np.quantile(y, [0.33, 0.66]))

    return X, y, y_bin, y_multi


@pytest.mark.xgboost
class TestXgbConverters(unittest.TestCase):
    def test_xgb_binary(self):
        X, _, y, _, _ = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            max_depth=5,
            learning_rate=0.25,
            n_estimators=10,
        )
        model.fit(X, y)
        ypred_model = model.predict_proba(X)[:, 1]

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)

        # print(at[0])
        # print(model.get_booster().get_dump()[0])

        # x = X[[2063], :]
        # print("at:", at[0].eval_node(x))
        # print("xgb:", model.apply(x))
        # print("at:", at.eval(x)[0])
        # print("xgb:", model.predict(x, output_margin=True))
        # print("at:", at.eval(x)[0])
        # print("diff:", model.predict(x, output_margin=True)-at.eval(x)[0])
        # print(x)

        # print(f"split_value {at[0].get_split(1).split_value:.16f}")
        # print(f"xvalue      {x[0,0]:.16f}")

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
            n_estimators=10,
        )
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
            n_estimators=10,
        )
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
            n_estimators=10,
        )
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_regression_ordinal(self):
        X, y, _, _, _ = get_img_data()
        X *= 100.0
        X = X.astype(int)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            max_depth=5,
            learning_rate=1.0,
            n_estimators=10,
        )
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

        splits = at.get_splits()
        fid0 = splits[0][10]
        fid1 = splits[1][10]
        X_force_exacteq_split0 = np.array([[fid0 - 1e10, fid1 - 1e10]])
        X_force_exacteq_split1 = np.array([[fid0, fid1]])

        for eps in [-1e-10, 0, +1e-10]:
            X_force_exacteq_split0 = np.array([[fid0 + eps, fid1 + eps]])
            pred_xgb = model.predict(X_force_exacteq_split0)
            pred_at = at.predict(X_force_exacteq_split0).ravel()
            self.assertTrue(np.isclose(pred_xgb[0], pred_at[0]))

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
            n_estimators=2,
        )
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
            n_estimators=10,
        )
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
            multi_strategy="one_output_per_tree",
            num_target=y.shape[1],
            tree_method="hist",
            max_depth=5,
            learning_rate=0.5,
            n_estimators=10,
        )
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model, single_rel_tol=1e-4)
        self.assertTrue(is_correct)

    def test_xgb_base_score_random_data(self):
        # Regression test for the base_score parsing bug: newer XGBoost
        # versions encode `base_score` in the saved JSON model as a
        # stringified array (e.g. "[4.89E-1]") instead of a bare float.
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(200, 5)).astype(np.float32).astype(np.float64)
        y = rng.uniform(size=200)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            max_depth=3,
            learning_rate=0.5,
            n_estimators=5,
        )
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_regressor_missing(self):
        X, y, _, _ = get_missing_data()
        model = xgb.XGBRegressor(max_depth=3, n_estimators=5, random_state=42)
        model.fit(X, y)

        ypred_model = model.predict(X, output_margin=True)

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_regressor_missing2(self):
        X, y, _, _, _ = get_img_data()

        # also downscale X's precision because XGBoost uses float32
        X = X.astype(np.float32).astype(np.float64)

        rng = np.random.default_rng(42)
        mask = rng.random(X.shape) < 0.2
        X[mask] = np.nan

        model = xgb.XGBRegressor(max_depth=3, n_estimators=1, random_state=42, learning_rate=1.0)
        model.fit(X, y)

        ypred_model = model.predict(X)

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_classifier_binary_missing(self):
        X, _, y, _ = get_missing_data()
        model = xgb.XGBClassifier(objective="binary:logistic", max_depth=3, n_estimators=5, random_state=42)
        model.fit(X, y)

        ypred_model = model.predict_proba(X)[:, 1]

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)

    def test_xgb_classifier_multiclass_missing(self):
        X, _, _, y = get_missing_data()
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=3,
            n_estimators=5,
            random_state=42,
        )
        model.fit(X, y)

        ypred_model = model.predict_proba(X)

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)

    def test_missing_values_error_handling(self):
        X, y, _, _ = get_missing_data()
        model = xgb.XGBRegressor(max_depth=2, n_estimators=1, random_state=42)
        model.fit(X, y)

        at = veritas.get_addtree(model, handle_missing=True)

        with self.assertRaises(ValueError) as ctx:
            at.predict(X)
        self.assertIn("Input data contains missing values", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            at.eval(X)
        self.assertIn("Input data contains missing values", str(ctx.exception))

        tree = at[0]
        with self.assertRaises(ValueError) as ctx:
            tree.eval(X)
        self.assertIn("Input data contains missing values", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            tree.eval_node(X)
        self.assertIn("Input data contains missing values", str(ctx.exception))

    def test_xgb_conversion_error_issue13(self):
        """Model is last tree of model trained in Issue 12 with 0.0 base score (`regr.get_booster()[2:]`)"""
        model_bytes = bytearray(
            b'{"learner":{"attributes":{},"feature_names":[],"feature_types":[],"gradient_booster":{"model":{"cats":{"enc":[],"feature_segments":[],"sorted_idx":[]},"gbtree_model_param":{"num_parallel_tree":"1","num_trees":"1"},"iteration_indptr":[0,1],"tree_info":[0],"trees":[{"base_weights":[8.0104626E-5,-3.069526E-1,5.026176E-1,-1.7521982E-1,-4.1596647E-2,2.7186227E-1,7.3281884E-2],"categories":[],"categories_nodes":[],"categories_segments":[],"categories_sizes":[],"default_left":[0,0,0,0,0,0,0],"id":0,"left_children":[1,3,5,-1,-1,-1,-1],"loss_changes":[2.548035E3,4.7799884E2,6.5279224E2,0E0,0E0,0E0,0E0],"parents":[2147483647,0,0,1,1,2,2],"right_children":[2,4,6,-1,-1,-1,-1],"split_conditions":[4.0625E0,2.5139E0,2.7E0,-1.7521982E-1,-4.1596647E-2,2.7186227E-1,7.3281884E-2],"split_indices":[0,0,5,0,0,0,0],"split_type":[0,0,0,0,0,0,0],"sum_hessian":[1.6512E4,1.025E4,6.262E3,3.872E3,6.378E3,2.443E3,3.819E3],"tree_param":{"num_deleted":"0","num_feature":"8","num_nodes":"7","size_leaf_vector":"1"}}]},"name":"gbtree"},"learner_model_param":{"base_score":"[0.0E0]","boost_from_average":"1","num_class":"0","num_feature":"8","num_target":"1"},"objective":{"name":"reg:squarederror","reg_loss_param":{"scale_pos_weight":"1"}}},"version":[3,4,1]}'
        )
        regr = xgb.XGBRegressor()
        regr.load_model(model_bytes)
        at = veritas.get_addtree(regr)

        # import io
        # with io.StringIO() as f:
        #    regr.get_booster().dump_model(f, dump_format="text")
        #    f.seek(0)
        #    print(f.read())
        # print(at[0])

        import json
        xgb_json = json.loads(model_bytes)
        print(json.dumps(xgb_json, indent=4))
        print(xgb_json["learner"]["gradient_booster"]["model"]["trees"][0]["split_conditions"])

        for dtype, conversion_error in ((np.float32, False), (np.float64, True)):
            X = np.array([[4.0625, 0.0, 0.0, 0.0, 0.0, 2.7, 0.0, 0.0]], dtype=dtype)
            # X.astype(np.float32).astype(np.float64)
            xgb_pred = regr.predict(X)

            print("XGB:", xgb_pred)
            print("VER:", at.predict(X)[0])

            self.assertEqual(not conversion_error, veritas.test_conversion(at, X, xgb_pred, silent=False))


@pytest.mark.sklearn
class TestSklearnConverters(unittest.TestCase):
    def test_rf_binary(self):
        X, _, y, _, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        clf = RandomForestClassifier(max_depth=6, random_state=0, n_estimators=2)
        clf.fit(X, y)
        ypred_model = clf.predict_proba(X)[:, 1]

        at = veritas.get_addtree(clf)
        is_correct = veritas.test_conversion(at, X, ypred_model)

        # print([t.eval_node(X[[4191], :])[0] for t in at])
        # print(clf.apply(X[[4191], :])[0])

        # for t in at:
        #    print(t)

        # import sklearn
        # for t in clf.estimators_:
        #    r = sklearn.tree.export_text(t, feature_names=['F0', 'F1'],
        #                                 decimals=4, spacing=2)
        #    print(r)
        self.assertTrue(is_correct)

    def test_rf_multiclass(self):
        X, _, _, y, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        clf = RandomForestClassifier(max_depth=8, random_state=0, n_estimators=25)
        clf.fit(X, y)
        ypred_model = clf.predict_proba(X)

        at = veritas.get_addtree(clf)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_rf_regression(self):
        X, y, _, _, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        clf = RandomForestRegressor(max_depth=8, random_state=0, n_estimators=25)
        clf.fit(X, y)
        ypred_model = clf.predict(X)

        # import sklearn
        # for x in clf.estimators_:
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

        model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=25)
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_sklearn_rf_regressor_missing(self):
        X, y, _, _ = get_missing_data()
        model = skensemble.RandomForestRegressor(max_depth=3, n_estimators=5, random_state=42)
        model.fit(X, y)

        ypred_model = model.predict(X)

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)

    def test_sklearn_rf_classifier_missing(self):
        X, _, y, _ = get_missing_data()
        model = skensemble.RandomForestClassifier(max_depth=3, n_estimators=5, random_state=42)
        model.fit(X, y)

        ypred_model = model.predict_proba(X)[:, 1]

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)

    def test_sklearn_tree_classifier_missing(self):
        X, _, y, _ = get_missing_data()
        model = sktree.DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)

        ypred_model = model.predict_proba(X)[:, 1]

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)

    def test_sklearn_tree_regressor_missing(self):
        X, y, _, _ = get_missing_data()
        model = sktree.DecisionTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)

        ypred_model = model.predict(X)

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)


@pytest.mark.lightgbm
class TestLgbConverters(unittest.TestCase):
    def test_lgb_binary(self):
        X, _, y, _, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        model = lgb.LGBMClassifier(objective="binary", num_leaves=64, learning_rate=0.25, n_estimators=10)
        model.fit(X, y)
        ypred_model = model.predict_proba(X)[:, 1]

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
            n_estimators=10,
        )
        model.fit(X, y)
        ypred_model = model.predict_proba(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

        print(
            "at base scores",
            [at.get_base_score(k) for k in range(at.num_leaf_values())],
        )

    def test_lgb_regression(self):
        X, y, _, _, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        model = lgb.LGBMRegressor(
            objective="regression_l2",
            max_depth=5,
            learning_rate=1.0,
            n_estimators=10,
            n_jobs=1,
        )
        model.fit(X, y)
        ypred_model = model.predict(X)

        at = veritas.get_addtree(model)
        is_correct = veritas.test_conversion(at, X, ypred_model)
        self.assertTrue(is_correct)

    def test_lgb_regressor_missing(self):
        X, y, _, _ = get_missing_data()
        model = lgb.LGBMRegressor(max_depth=3, n_estimators=5, random_state=42, verbose=-1)
        model.fit(X, y)

        ypred_model = model.predict(X)

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)

    def test_lgb_classifier_binary_missing(self):
        X, _, y, _ = get_missing_data()
        model = lgb.LGBMClassifier(objective="binary", max_depth=3, n_estimators=5, random_state=42, verbose=-1)
        model.fit(X, y)

        ypred_model = model.predict_proba(X)[:, 1]

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)

    def test_lgb_classifier_multiclass_missing(self):
        X, _, _, y = get_missing_data()
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            max_depth=3,
            n_estimators=5,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)

        ypred_model = model.predict_proba(X)

        at = veritas.get_addtree(model, handle_missing=True)
        X_trans = veritas.transform_data_for_missing(X)

        is_correct = veritas.test_conversion(at, X_trans, ypred_model)
        self.assertTrue(is_correct)


@pytest.mark.sklearn
class TestGrootConverter(unittest.TestCase):
    def test_groot_binary(self):
        try:
            from groot.model import GrootRandomForestClassifier
        except ModuleNotFoundError:
            print("skipping test on groot")
            return

        X, _, y, _, _ = get_img_data()
        X = X.astype(np.float32).astype(np.float64)
        clf = GrootRandomForestClassifier(max_depth=6, random_state=0, n_estimators=2, attack_model=[0.1, 0.1])
        clf.fit(X, y)
        ypred_model = clf.predict_proba(X)[:, 1]

        at = veritas.get_addtree(clf)
        is_correct = veritas.test_conversion(at, X, ypred_model)


if __name__ == "__main__":
    unittest.main()
