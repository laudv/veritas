import os, json
import util
import pickle

import sklearn.metrics as metrics

from veritas import Optimizer
from veritas.xgb import addtree_from_xgb_model, addtrees_from_multiclass_xgb_model

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

    def get_at(self):
        raise RuntimeError("not implemented")

    def get_model_name(self, num_trees, tree_depth):
        return f"{type(self).__name__}{self.special_tag}-{num_trees}-{tree_depth}"

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
