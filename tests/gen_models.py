import unittest
import imageio
import numpy as np

# from veritas.add_tree import get_addtree
from veritas.model_conversion_test import test_model_conversion
import xgboost
import lightgbm as lgbm

from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_moons

from veritas import *

import timeit


class Test_AddTree_Regression(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        ############# Load Regression Data #############
        img = imageio.imread("data/img.png")
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        X = X.astype(float)
        self.regr_data = (X, y)

    def test_xgboost_regression(self):
        X, y = self.regr_data

        ############# XGB #############
        print("XGB - Regression:")
        regr = xgboost.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=3,
            learning_rate=1.0,
            n_estimators=3)
        model = regr.fit(X, y)
        at = get_addtree(model)

        mae, rmse = test_model_conversion(
            model, at, (X, y))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {rmse}")
        print(f"very easy img: mae model difference {mae}")

        at.write("models/xgboost-img-very-easy-new.json")

        regr = xgboost.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=6,
            learning_rate=0.5,
            n_estimators=10)
        model = regr.fit(X, y)
        at = get_addtree(model)

        mae, rmse = test_model_conversion(
            model, at, (X, y))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {rmse}")
        print(f"very easy img: mae model difference {mae}")

        at.write("models/xgboost-img-easy-new.json")

        regr = xgboost.XGBRegressor(
            objective="reg:squarederror",
            nthread=4,
            tree_method="hist",
            max_depth=6,
            learning_rate=0.4,
            n_estimators=50)
        model = regr.fit(X, y)
        at = get_addtree(model)

        mae, rmse = test_model_conversion(
            model, at, (X, y))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {rmse}")
        print(f"very easy img: mae model difference {mae}")

        at.write("models/xgboost-img-hard-new.json")
        print()

    def test_sklearn_regression(self):
        X, y = self.regr_data

        ############# SkLearn #############
        print("SkLearn - Regression:")
        clf = RandomForestRegressor(
            max_depth=6,
            random_state=0,
            n_estimators=50)
        model = clf.fit(X, y)
        at = get_addtree(model)

        mae, rmse = test_model_conversion(model, at, (X, y))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {rmse}")
        print(f"very easy img: mae model difference {mae}")

        at.write("models/sklearn-img-hard-new.json")
        print()

    def test_lgbm_regression(self):
        X, y = self.regr_data

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

        mae, rmse = test_model_conversion(model, at, (X, y))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {rmse}")
        print(f"very easy img: mae model difference {mae}")

        at.write("models/lgbm-img-very-easy-new.json")

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

        mae, rmse = test_model_conversion(model, at, (X, y))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {rmse}")
        print(f"very easy img: mae model difference {mae}")

        at.write("models/lgbm-img-easy-new.json")
        # with open("models/xgboost-img-easy-values.json", "w") as f:
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

        mae, rmse = test_model_conversion(model, at, (X, y))
        self.assertAlmostEqual(mae, 0.0, delta=1e-4)

        print(f"very easy img: rmse train {rmse}")
        print(f"very easy img: mae model difference {mae}")

        at.write("models/lgbm-img-hard-new.json")
        print()


class Test_AddTree_BinaryClassification(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        ############# Load Binary Data #############
        full_set = load_breast_cancer()
        self.bin_data = (full_set.data, full_set.target)

    def test_xgb_binary_class(self):
        ############# XGB #############
        print("XGB - Binary Classification:")

        print("Make moons") 
        (X,Y) = make_moons(100)

        clf = xgboost.XGBClassifier(
            objective="binary:logistic",
            nthread=4,
            tree_method="hist",
            max_depth=4,
            learning_rate=1,
            n_estimators=1)

        trained_model = clf.fit(X, Y)

        # Convert the XGBoost model to a Veritas tree ensemble
        addtree = get_addtree(trained_model)

        mae, model_acc = test_model_conversion(trained_model, addtree, (X, Y))

        print(f"easy bc: accuracy {model_acc}")
        print(f"easy bc: mae model difference {mae}")
        print()

        print("Dataset") 
        X, y = self.bin_data

        clf = xgboost.XGBClassifier(
            objective="binary:logistic",
            nthread=4,
            tree_method="hist",
            max_depth=4,
            learning_rate=1,
            n_estimators=1)
        model = clf.fit(X, y)

        at = get_addtree(model)

        mae, model_acc = test_model_conversion(model, at, (X, y))

        print(f"easy bc: accuracy {model_acc}")
        print(f"easy bc: mae model difference {mae}")
        print()
        
        # NOT GOOD ENOUGH ! Show floating error
        self.assertAlmostEqual(mae, 0.0, delta=1e-2)

        at.write("models/xgboost-bc-easy.json")

    def test_sklearn_binary_class(self):
        X, y = self.bin_data

        ############# SkLearn #############
        print("SkLearn - Binary Classification:")
        clf = RandomForestClassifier(
            max_depth=6,
            random_state=0,
            n_estimators=50,)
        model = clf.fit(X, y)
        at = get_addtree(model)

        mae, acc = test_model_conversion(model, at, (X, y))

        self.assertAlmostEqual(mae, 0.0, delta=1e-6)
        print(f"easy bc: accuracy {acc}")
        print(f"easy bc: mae model difference {mae}")

        at.write("models/sklearn-bc-easy.json")
        print()

    def test_lgbm_binary_class(self):
        X, y = self.bin_data

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

        mae, acc = test_model_conversion(model, at, (X, y))

        self.assertAlmostEqual(mae, 0.0, delta=1e-6)
        print(f"easy bc: accuracy {acc}")
        print(f"easy bc: mae model difference {mae}")

        at.write("models/lgbm-bc-easy.json")
        print()


class Test_AddTree_MultiClass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        ############# Load Multiclass Data #############
        full_set = load_digits()
        self.multi_data = (full_set.data, full_set.target)

    def test_xgb_multiclass(self):
        X, y = self.multi_data

        ############# XGB #############
        print("XGB - Multiclass:")
        clf = xgboost.XGBClassifier(
            objective="multi:softmax",
            num_class=10,
            nthread=4,
            tree_method="hist",
            max_depth=6,
            learning_rate=0.5,
            n_estimators=10)
        model = clf.fit(X, y)
        ats = get_addtree(model)

        mae, acc = test_model_conversion(model, ats, (X, y))

        self.assertAlmostEqual(mae, 0.0, delta=1e-6)
        print(f"multi: acc train {acc*100:.1f}%")
        print(f"multi: mae model difference {mae}")
        print()

    def test_sklearn_multiclass(self):
        X, y = self.multi_data

        ############# SkLearn #############
        print("SkLearn - Multiclass:")
        clf = RandomForestClassifier(
            max_depth=8, random_state=0, n_estimators=20)
        model = clf.fit(X, y)
        ats = get_addtree(model)

        mae, acc = test_model_conversion(model, ats, (X, y))

        self.assertAlmostEqual(mae, 0.0, delta=1e-6)
        print(f"multi: acc train RF {acc*100:.1f}%")
        print(f"multi: mae mae model difference {mae}")
        print()

    def test_lgbm_multiclass(self):
        X, y = self.multi_data

        ############## LGBM #############
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

        mae, acc = test_model_conversion(model, ats, (X, y))

        self.assertAlmostEqual(mae, 0.0, delta=1e-6)
        print(f"multi: acc train {acc*100:.1f}%")
        print(f"multi: mae model difference {mae}")
        print()


# TODO: test for using boosters with DMatrix
if __name__ == "__main__":
    # xgboost = unittest.TestSuite()
    # xgboost.addTest(Test_AddTree_Regression('test_xgb_regression'))
    # xgboost.addTest(Test_AddTree_BinaryClassification('test_xgb_binary_class'))
    # xgboost.addTest(Test_AddTree_MultiClass('test_xgb_multiclass'))

    # sklearn = unittest.TestSuite()
    # sklearn.addTest(Test_AddTree_Regression('test_sklearn_regression'))
    # sklearn.addTest(Test_AddTree_BinaryClassification(
    #     'test_sklearn_binary_class'))
    # sklearn.addTest(Test_AddTree_MultiClass('test_sklearn_multiclass'))

    # lgbm = unittest.TestSuite()
    # lgbm.addTest(Test_AddTree_Regression('test_lgbm_regression'))
    # lgbm.addTest(Test_AddTree_BinaryClassification('test_lgbm_binary_class'))
    # lgbm.addTest(Test_AddTree_MultiClass('test_lgbm_multiclass'))

    unittest.main()
