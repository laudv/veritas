# Copyright 2020 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import gzip, types
from io import StringIO

from .pyveritas import *

from .xgb import \
    addtree_from_xgb_model, \
    addtrees_from_multiclass_xgb_model
del xgb

from .sklearn import \
    addtree_from_sklearn_ensemble, \
    addtrees_from_multiclass_sklearn_ensemble
del sklearn

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

setattr(AddTree, "write", __addtree_write)
setattr(AddTree, "read", __addtree_read)
setattr(AddTree, "__iter__", __addtree_iter)

from .util import *
del util

try:
    from . import kantchelian
except:
    print("Veritas: install `gurobipy` for MILP support")

from . import robustness
