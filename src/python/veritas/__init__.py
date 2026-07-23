# Copyright 2026 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

# ruff: noqa: F401, E402

import gzip as _gzip

import numpy as _np

from .veritas_core import (
    BOOL_SPLIT_VALUE,
    FALSE_INTERVAL,
    TRUE_INTERVAL,
    AddTree,
    AddTreeType,
    Bounds,
    Config,
    FeatMap,
    HeuristicType,
    Interval,
    IntervalPair,
    LtSplit,
    Search,
    Solution,
    Statistics,
    StopReason,
    Tree,
)

# PACKAGE META

__version__ = "0.3.1"
__title__ = "veritas"
__description__ = "Versatile Verification of Tree Ensembles"
__url__ = "https://github.com/laudv/veritas"
__doc__ = __description__ + " <" + __url__ + ">"

__author__ = "Laurens Devos"
__email__ = ""

__license__ = "Apache-2.0"
__copyright__ = "Copyright (c) 2026 DTAI Research Group, KU Leuven"


def __interval_hash(self):
    return hash((self.lo, self.hi))


Interval.hash = __interval_hash


def __addtree_write(self, f, compressed=False):
    if compressed:
        with _gzip.open(f, "wb") as fh:
            json = self.to_json()
            fh.write(json.encode("utf-8"))
    else:
        with open(f, "w") as fh:
            fh.write(self.to_json())


def __addtree_read(f, compressed=False):
    if compressed:
        with _gzip.open(f, "rb") as fh:
            json = fh.read()
            return AddTree.from_json(json.decode("utf-8"))
    else:
        with open(f, "r") as fh:
            return AddTree.from_json(fh.read())


def __addtree_iter(self):
    for i in range(len(self)):
        yield self[i]


FloatT = _np.float64

__addtree_eval_cpp = AddTree.eval


def __addtree_eval(self, data):
    data = _np.array(data, dtype=FloatT)
    if _np.isnan(data).any():
        raise ValueError(
            "Input data contains missing values (NaNs). Veritas trees do not natively "
            "evaluate NaNs. Please transform your dataset using `veritas.transform_data_for_missing(X)` "
            "and convert your model using `veritas.get_addtree(model, handle_missing=True)`."
        )
    return __addtree_eval_cpp(self, data)


__addtree_predict_cpp = AddTree.predict


def __addtree_predict(self, data):
    data = _np.array(data, dtype=FloatT)
    if _np.isnan(data).any():
        raise ValueError(
            "Input data contains missing values (NaNs). Veritas trees do not natively "
            "evaluate NaNs. Please transform your dataset using `veritas.transform_data_for_missing(X)` "
            "and convert your model using `veritas.get_addtree(model, handle_missing=True)`."
        )
    return __addtree_predict_cpp(self, data)


AddTree.write = __addtree_write
AddTree.read = __addtree_read
AddTree.__iter__ = __addtree_iter
AddTree.eval = __addtree_eval
AddTree.predict = __addtree_predict

__tree_eval_cpp = Tree.eval


def __tree_eval(self, data, nid=None):
    data = _np.array(data, dtype=FloatT)
    if _np.isnan(data).any():
        raise ValueError(
            "Input data contains missing values (NaNs). Veritas trees do not natively "
            "evaluate NaNs. Please transform your dataset using `veritas.transform_data_for_missing(X)` "
            "and convert your model using `veritas.get_addtree(model, handle_missing=True)`."
        )
    if nid is None:
        nid = self.root()
    return __tree_eval_cpp(self, data, nid)


__tree_eval_node_cpp = Tree.eval_node


def __tree_eval_node(self, data, nid=None):
    data = _np.array(data, dtype=FloatT)
    if _np.isnan(data).any():
        raise ValueError(
            "Input data contains missing values (NaNs). Veritas trees do not natively "
            "evaluate NaNs. Please transform your dataset using `veritas.transform_data_for_missing(X)` "
            "and convert your model using `veritas.get_addtree(model, handle_missing=True)`."
        )
    if nid is None:
        nid = self.root()
    return __tree_eval_node_cpp(self, data, nid)


Tree.eval = __tree_eval
Tree.eval_node = __tree_eval_node


################################################################################
#              Imports that rely on initialized Veritas core below             #
################################################################################

from .util import get_closest_example, transform_data_for_missing

try:  # fails when gurobipy not installed
    from .kantchelian import (
        KantchelianAttack,
        KantchelianBase,
        KantchelianOutputOpt,
        KantchelianTargetedAttack,
    )
except ModuleNotFoundError:
    pass

try:
    from .smt import Verifier, VerifierTimeout
except ModuleNotFoundError:
    pass

try:
    from .z3backend import Z3Backend
except ModuleNotFoundError:
    pass

from .addtree_conversion import (
    AddTreeConverter,
    InapplicableAddTreeConverter,
    NoRegisteredConverterException,
    add_addtree_converter,
    get_addtree,
    test_conversion,
)
from .lgbm_converter import LGBMAddTreeConverter
from .robustness import (
    MilpRobustnessSearch,
    RobustnessSearch,
    SMTRobustnessSearch,
    VeritasRobustnessSearch,
)
from .sklearn_converter import (
    SklGbdtAddTreeConverter,
    SklRfAddTreeConverter,
    SklTreeAddTreeConverter,
)

# try: # fails when groot not installed
#     from .groot import \
#         addtree_from_groot_ensemble
# except ModuleNotFoundError:
#     pass
################################################################################
#                  Initialize the AddTree converter registry                   #
################################################################################
from .xgb_converter import XGBAddTreeConverter

add_addtree_converter(LGBMAddTreeConverter())
add_addtree_converter(SklRfAddTreeConverter())
add_addtree_converter(SklGbdtAddTreeConverter())
add_addtree_converter(SklTreeAddTreeConverter())
add_addtree_converter(XGBAddTreeConverter())

try:
    from .groot import GrootAddTreeConverter

    add_addtree_converter(GrootAddTreeConverter())
except ModuleNotFoundError:
    pass
