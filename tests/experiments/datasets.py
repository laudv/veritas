import os, json
import util
import pickle
import numpy as np
import pandas as pd

import sklearn.metrics as metrics
from sklearn import preprocessing

from veritas import addtree_from_xgb_model, addtrees_from_multiclass_xgb_model

class Dataset:
    models_dir = "tests/experiments/models"

    def __init__(self, special_name_tag=""):
        self.special_tag = special_name_tag # special parameters, name indication
        self.X = None
        self.y = None

    def load_dataset(self): # populate X, y
        raise RuntimeError("not implemented")

    def load_model(self, num_trees, tree_depth): # populate self.model, self.at, self.feat2id
        """ populate self.model, self.at """
        raise RuntimeError("not implemented")

    def get_model_name(self, num_trees, tree_depth):
        return f"{type(self).__name__}{self.special_tag}-{num_trees}-{tree_depth}"

    def minmax_normalize(self):
        X = self.X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=self.X.columns)
        self.X = df

class Calhouse(Dataset):

    def __init__(self):
        super().__init__()
        self.params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "seed": 14,
            "nthread": 1,
        }
    
    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = util.load_openml("calhouse", data_id=537)
            self.y = np.log(self.y)

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, raw_yhat): #maximized
                return -metrics.mean_squared_error(y, raw_yhat)

            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0

class Allstate(Dataset):
    def __init__(self):
        super().__init__()
        self.params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "seed": 14,
            "nthread": 1,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            allstate_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "allstate.h5")
            data = pd.read_hdf(allstate_data_path)
            self.X = data.drop(columns=["loss"])
            self.y = data.loss

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, raw_yhat): #maximized
                return -metrics.mean_squared_error(y, raw_yhat)

            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0

class Covtype(Dataset):
    def __init__(self):
        super().__init__()
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "tree_method": "hist",
            "seed": 235,
            "nthread": 1,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = util.load_openml("covtype", data_id=1596)
            self.y = (self.y==2)

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, raw_yhat):
                return metrics.accuracy_score(y, raw_yhat > 0)

            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0
        
class CovtypeNormalized(Covtype):
    def __init__(self):
        super().__init__()

    def load_dataset(self):
        if self.X is None or self.y is None:
            super().load_dataset()
            self.minmax_normalize()

class Higgs(Dataset):
    def __init__(self):
        super().__init__()
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "tree_method": "hist",
            "seed": 220,
            "nthread": 1,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            higgs_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "higgs.h5")
            self.X = pd.read_hdf(higgs_data_path, "X")
            self.y = pd.read_hdf(higgs_data_path, "y")

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, raw_yhat):
                return metrics.accuracy_score(y, raw_yhat > 0)

            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0

class LargeHiggs(Dataset):
    def __init__(self):
        super().__init__()
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "tree_method": "hist",
            "seed": 220,
            "nthread": 1,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            higgs_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "higgs_large.h5")
            data = pd.read_hdf(higgs_data_path)
            self.y = data[0]
            self.X = data.drop(columns=[0])
            columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.X.columns = columns
            self.minmax_normalize()

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, raw_yhat):
                return metrics.accuracy_score(y, raw_yhat > 0)

            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0


class Mnist(Dataset):

    def __init__(self):
        super().__init__()
        self.params = {
            "num_class": 10,
            "objective": "multi:softmax",
            "tree_method": "hist",
            "eval_metric": "merror",
            "seed": 53589,
            "nthread": 4,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = util.load_openml("mnist", data_id=554)

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, yhat): #maximized
                return metrics.accuracy_score(y, yhat)
            
            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)
            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtrees_from_multiclass_xgb_model(self.model, 10, feat2id_map=self.feat2id)
        for at in self.at:
            at.base_score = 0

class MnistNormalized(Mnist):
    def __init__(self):
        super().__init__()

    def load_dataset(self):
        if self.X is None or self.y is None:
            super().load_dataset()
            self.minmax_normalize()

class Mnist2v6(Mnist):
    def __init__(self):
        super().__init__()
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "tree_method": "hist",
            "seed": 235,
            "nthread": 4,
            "subsample": 0.5,
            "colsample_bytree": 0.8,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            super().load_dataset()
            self.X = self.X.loc[(self.y==2) | (self.y==6), :]
            self.y = self.y[(self.y==2) | (self.y==6)]
            self.y = (self.y == 2.0).astype(float)
            self.X.reset_index(inplace=True, drop=True)
            self.y.reset_index(inplace=True, drop=True)

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, raw_yhat):
                return metrics.accuracy_score(y, raw_yhat > 0)

            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0

class FashionMnist(Dataset):
    def __init__(self):
        super().__init__()
        self.params = {
            "num_class": 10,
            "objective": "multi:softmax",
            "tree_method": "hist",
            "eval_metric": "merror",
            "seed": 132955,
            "nthread": 1,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            self.X, self.y = util.load_openml("fashion_mnist", data_id=40996)
            #self.minmax_normalize()

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, yhat): #maximized
                return metrics.accuracy_score(y, yhat)
            
            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)
            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtrees_from_multiclass_xgb_model(self.model, 10, feat2id_map=self.feat2id)
        for at in self.at:
            at.base_score = 0

class FashionMnist2v6(FashionMnist):
    def __init__(self):
        super().__init__()
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "tree_method": "hist",
            "seed": 235,
            "nthread": 4,
            "subsample": 0.5,
            "colsample_bytree": 0.8,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            super().load_dataset()
            self.X = self.X.loc[(self.y==2) | (self.y==6), :]
            self.y = self.y[(self.y==2) | (self.y==6)]
            self.y = (self.y == 2.0).astype(float)
            self.X.reset_index(inplace=True, drop=True)
            self.y.reset_index(inplace=True, drop=True)

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, raw_yhat):
                return metrics.accuracy_score(y, raw_yhat > 0)

            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0

class Ijcnn1(Dataset):
    def __init__(self):
        super().__init__()
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "tree_method": "hist",
            "seed": 235,
            "nthread": 1,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            ijcnn1_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "ijcnn1.h5")
            self.X = pd.read_hdf(ijcnn1_data_path, "Xtrain")
            self.Xtest = pd.read_hdf(ijcnn1_data_path, "Xtest")
            columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.X.columns = columns
            self.Xtest.columns = columns
            self.y = pd.read_hdf(ijcnn1_data_path, "ytrain")
            self.ytest = pd.read_hdf(ijcnn1_data_path, "ytest")
            self.minmax_normalize()

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, raw_yhat):
                return metrics.accuracy_score(y, raw_yhat > 0)

            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0

class Webspam(Dataset):
    def __init__(self):
        super().__init__()
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "error",
            "tree_method": "hist",
            "seed": 732,
            "nthread": 1,
        }

    def load_dataset(self):
        if self.X is None or self.y is None:
            data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "webspam_wc_normalized_unigram.h5")
            self.X = pd.read_hdf(data_path, "X")
            self.X.columns = [f"a{i}" for i in range(self.X.shape[1])]
            self.y = pd.read_hdf(data_path, "y")
            self.minmax_normalize()

    def load_model(self, num_trees, tree_depth):
        model_name = self.get_model_name(num_trees, tree_depth)
        if not os.path.isfile(os.path.join(self.models_dir, f"{model_name}.xgb")):
            self.load_dataset()
            print(f"training model depth={tree_depth}, num_trees={num_trees}")

            def metric(y, raw_yhat):
                return metrics.accuracy_score(y, raw_yhat > 0)

            self.params["max_depth"] = tree_depth
            self.model, lr, metric_value = util.optimize_learning_rate(self.X,
                    self.y, self.params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(self.X.columns)}

            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.models_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0
