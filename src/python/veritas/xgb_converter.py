# Copyright 2023 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos, Alexander Schoeters
#
# This requires `xgboost` to be installed.

import os
import json
import tempfile
import numpy as np

from . import AddTree, AddTreeType, AddTreeConverter
from . import InapplicableAddTreeConverter
from . import FloatT

class XGBAddTreeConverter(AddTreeConverter):
    def convert(self, model):
        try:
            from xgboost import XGBModel
            from xgboost.core import Booster as XGBBooster
        except ModuleNotFoundError:
            raise InapplicableAddTreeConverter("xgb not installed")

        if isinstance(model, XGBModel):
            model = model.get_booster()

        if not isinstance(model, XGBBooster):
            raise InapplicableAddTreeConverter("not an xgb model")

        #param_dump = json.loads(model.save_config())["learner"]
        booster_json = get_booster_json(model)
        trees = booster_json["learner"]["gradient_booster"]["model"]["trees"]
        feature_names = booster_json["learner"]["feature_names"]
        if len(feature_names) == 0:
            feature_names = None
        objective = booster_json["learner"]["objective"]["name"]
        base_score = float(booster_json["learner"]["learner_model_param"]["base_score"])
        num_class = int(booster_json["learner"]["learner_model_param"]["num_class"])
        num_target = int(booster_json["learner"]["learner_model_param"]["num_target"])
        grad_boost_name = booster_json["learner"]["gradient_booster"]["name"]
        tree_info = booster_json["learner"]["gradient_booster"]["model"]["tree_info"]

        #print("OBJECTIVE", objective)
        #print("FEATURES", feature_names)
        #print("BASE_SCORE", base_score)
        #print("NUM_CLASS", num_class)
        #print("TREE_INFO", tree_info)

        #cpy = __import__('copy').deepcopy(booster_json)
        #cpy["learner"]["gradient_booster"]["model"]["trees"] = "The TREES"
        #__import__('pprint').pprint(cpy)

        if grad_boost_name != "gbtree":
            raise RuntimeError(f"Tree type {grad_boost_name} not supported")

        at_type = AddTreeType.REGR
        if "multi" in objective or "logistic" in objective:
            at_type = AddTreeType.CLF_SOFTMAX

        size_leaf_vector = int(trees[0]["tree_param"]["size_leaf_vector"])
        num_leaf_values = max(size_leaf_vector, num_class, num_target, 1)
        at = AddTree(num_leaf_values, at_type)

        version = booster_json['version'][0]
        for tree_json, clazz in zip(trees, tree_info):
            convert_xgb_json_tree(at, tree_json, clazz, version)

        single_rel_tol = 1e-5
        rel_tol = single_rel_tol * len(at)
        base_score_manual, base_score_diff_std = try_determine_base_score(
                model, at, feature_names)
        err = np.abs(base_score_manual-base_score)
        print()
        print("| XGBOOST's base_score")
        print("|   base_score diff std     ", base_score_diff_std,
              "(!) NOT OK" if np.any(base_score_diff_std > 1e-5) else "OK")
        print("|   base_score reported     ", base_score)
        print("|   versus manually detected", base_score_manual)
        print("|   abs err                 ", err)
        print("|   rel err                 ", err/base_score)
        if not np.all(np.isclose(base_score_manual, base_score, rtol=rel_tol)):
            msg = "(!) base_score NOT THE SAME"
            base_score = base_score_manual
        else:
            msg = "base_score OK"
        print(f"|   {msg} with relative tolerance {rel_tol:g}")
        print()

        if isinstance(base_score, float):
            base_score = np.full(num_leaf_values, base_score)
        for k in range(num_leaf_values):
            at.set_base_score(k, base_score[k])

        return at

    def predict(self, model, X):
        from xgboost import XGBModel, DMatrix
        from xgboost.core import Booster as XGBBooster
        if isinstance(model, XGBModel):
            model = model.get_booster()
        if not isinstance(model, XGBBooster):
            raise InapplicableAddTreeConverter("not an xgb model")

        pred = model.predict(DMatrix(X), output_margin=True)
        return pred

def get_booster_json(booster):
    # Write to a temp file using XGBoost's save_model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as fp:
        fp.close() # not deleted because delete=False
        booster.save_model(fp.name)

        with open(fp.name) as f:
            # XGBoost dumps floats as float32, so we also need to parse them as
            # float32s, even though Veritas uses float64
            # e.g.
            #   - the real float32 is 0.123452
            #   - the JSON (a string) contains 0.12345
            #   - the closest float32 is 0.123452
            #   - the closest float64 might be 0.123450001 (!! lower than 0.123452)
            booster_json = json.load(f, parse_float=np.float32)

        # clean up our mess
        os.remove(fp.name)
    return booster_json

def convert_xgb_json_tree(at, tree_json, clazz, version):
    t = at.add_tree()
    slv = int(tree_json["tree_param"]["size_leaf_vector"])
    nlv = at.num_leaf_values()

    if len(tree_json["categories"]) > 0:
        raise RuntimeError("xgb categories not supported")
    #if any(map(lambda v: v!=0, tree_json["default_left"])):
    #    raise RuntimeError("none default_left not supported")
    if any(map(lambda v: v!=0, tree_json["split_type"])):
        raise RuntimeError("split_type not supported")

    lefts = tree_json["left_children"]
    rights = tree_json["right_children"]
    thresholds = tree_json["split_conditions"]
    feat_ids = tree_json["split_indices"]
    leafvals = tree_json["base_weights"]

    # versions <2 put even leaf values into tree_json['split_conditions']
    # (?) what is in tree_json['base_weights'] then?
    # https://github.com/dmlc/xgboost/blob/4de866211d5bba706f6b94d1ba4a102fe885c1b9/demo/json-model/json_parser.py#L132 
    if version < 2:
        leafvals = thresholds 

    stack = [(0, t.root())]
    while len(stack) > 0:
        i, j = stack.pop()
        left, right = lefts[i], rights[i]

        # internal
        if left != -1 and right != -1:
            t.split(j, feat_ids[i], thresholds[i])
            stack.append((right, t.right(j)))
            stack.append((left, t.left(j)))
        elif slv == 1:
            t.set_leaf_value(j, clazz, leafvals[i])
        else:
            for k in range(nlv):
                u = i * nlv + k
                t.set_leaf_value(j, k, leafvals[u])


def try_determine_base_score(booster, at, feature_names, seed=472934901, n=1000):
    import xgboost as xgb
    num_features = booster.num_features()

    rng = np.random.default_rng(seed)
    x = rng.random((n, num_features), dtype=FloatT)

    # use the split values to generate a random dataset
    for k, vs in at.get_splits().items():
        vmin, vmax = min(vs), max(vs)
        vdiff = vmax-vmin
        vmin -= 0.05 * vdiff
        vmax += 0.05 * vdiff
        x[:, k] = x[:, k] * (vmax-vmin) + vmin

    dmat = xgb.DMatrix(x, feature_names=feature_names)
    pred0 = booster.predict(dmat, output_margin=True)
    pred1 = at.eval(x)

    if pred1.shape[1] == 1:
        pred1 = pred1.reshape(pred1.shape[0])

    diff = (pred0 - pred1).astype(FloatT)

    # All the errors should be the same, otherwise it is not a mistake in the
    # base_score, but one in the structure of the trees! That is why we also report
    # the np.std.

    return np.mean(diff, axis=0), np.std(diff, axis=0)
