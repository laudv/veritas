# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import gzip
import numpy as np

from .pyveritas import *
del pyveritas

try: # fails when xgboost not installed
    from .xgb import \
        addtree_from_xgb_model, \
        addtrees_from_multiclass_xgb_model
    del xgb
except ModuleNotFoundError as e: pass

try: # fails when sklearn not installed
    from .sklearn import \
        addtree_from_sklearn_ensemble, \
        addtrees_from_multiclass_sklearn_ensemble
    del sklearn
except ModuleNotFoundError as e: pass

try: # fails when groot not installed
    from .groot import \
        addtree_from_groot_ensemble
    del groot
except ModuleNotFoundError as e: pass



# PACKAGE META

__version__ = "0.1"
__title__ = "veritas"
__description__ = "Versatile Verification of Tree Ensembles"
__url__ = "https://github.com/laudv/veritas"
__doc__ = __description__ + " <" + __url__ + ">"

__author__ = "Laurens Devos"
__email__ = ""

__license__ = "Apache 2.0"
__copyright__ = "Copyright (c) 2022 DTAI Research Group, KU Leuven"





def __domain_hash(self):
    return hash((self.lo, self.hi))

setattr(Domain, "hash", __domain_hash)

def __addtree_write(self, f, compress=False):
    if compress:
        with gzip.open(f, "wb") as fh:
            json = self.to_json()
            fh.write(json.encode("utf-8"))
    else:
        with open(f, "w") as fh:
            fh.write(self.to_json())

def __addtree_read(f, compressed=False):
    if compressed:
        with gzip.open(f, "rb") as fh:
            json = fh.read()
            return AddTree.from_json(json.decode("utf-8"))
    else:
        with open(f, "r") as fh:
            return AddTree.from_json(fh.read())

def __addtree_iter(self):
    for i in range(len(self)):
        yield self[i]

__addtree_eval_cpp = AddTree.eval
def __addtree_eval(self, data):
    data = np.array(data, dtype=np.float32)
    return __addtree_eval_cpp(self, data)

setattr(AddTree, "write", __addtree_write)
setattr(AddTree, "read", __addtree_read)
setattr(AddTree, "__iter__", __addtree_iter)
setattr(AddTree, "eval", __addtree_eval)

__tree_eval_cpp = Tree.eval
def __tree_eval(self, data):
    data = np.array(data, dtype=np.float32)
    return __tree_eval_cpp(self, data)

__tree_eval_node_cpp = Tree.eval_node
def __tree_eval_node(self, data):
    data = np.array(data, dtype=np.float32)
    return __tree_eval_node_cpp(self, data)

setattr(Tree, "eval", __tree_eval)
setattr(Tree, "eval_node", __tree_eval_node)

from .util import *
del util

try: # fails when gurobipy not installed
    from . import kantchelian
except ModuleNotFoundError as e: pass

from . import robustness
del robustness
