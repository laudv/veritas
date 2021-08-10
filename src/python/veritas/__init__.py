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
setattr(AddTree, "write", __addtree_write)
setattr(AddTree, "read", __addtree_read)

def get_closest_example(solution, example, guard=1e-4, featmap=None):
    num_attributes = len(example)
    #intervals = self.solution_to_intervals(solution, num_attributes)[instance]

    if featmap is None:
        featmap = {i: [i] for i in range(num_attributes)}
    else:
        featmap = featmap.get_indices_map()

    closest = example.copy()

    for index, dom in solution.box().items():
        for feat_id in featmap[index]:
            x = example[feat_id]
            if dom.lo <= x and x < dom.hi:
                continue # keep the value x

            dist_lo = abs(dom.lo - x)
            dist_hi = abs(x - dom.hi)
            if dist_lo > dist_hi:
                closest[feat_id] = dom.hi - guard
            else:
                closest[feat_id] = dom.lo

            #print(f"dom {feat_id}:", dom, x, "->", closest[feat_id])

    return closest

try:
    from . import kantchelian
except:
    print("Veritas: install `gurobipy` for MILP support")

from . import robustness

#
#def __realdomain__str(self):
#    return "[{:.3g}, {:.3g}]".format(self.lo, self.hi)
#
#def __realdomain__eq(self, o):
#    return self.lo == o.lo and self.hi == o.hi
#
#def __realdomain__hash(self):
#    return hash((self.lo, self.hi))
#
#def __booldomain__str(self):
#    return "{True}" if self.is_true() \
#        else "{False}" if self.is_false() \
#        else "{False, True}"
#
#def __booldomain__eq(self, o):
#    return self._value == o._value
#
#def __booldomain__hash(self):
#    return hash(self._value)
#
#RealDomain.__str__ = __realdomain__str
#RealDomain.__eq__ = __realdomain__eq
#RealDomain.__hash__ = __realdomain__hash
#BoolDomain.__str__ = __booldomain__str
#BoolDomain.__eq__ = __booldomain__eq
#BoolDomain.__hash__ = __booldomain__hash
#
#def __tree_predict_leaf(self, example):
#    node = self.root()
#    while not self.is_leaf(node):
#        split = self.get_split(node)
#        value = example[split.feat_id]
#        if isinstance(split, LtSplit):
#            #assert isinstance(value, float), f"is {type(value)} instead"
#            go_left = split.test(value)
#        elif isinstance(split, BoolSplit):
#            #assert isinstance(value, bool), f"is {type(value)} instead"
#            go_left = split.test(value)
#        node = self.left(node) if go_left else self.right(node)
#    return node
#
#def __tree_predict_single(self, example):
#    node = __tree_predict_leaf(self, example)
#    return self.get_leaf_value(node)
#
#def __tree_predict(self, examples):
#    return list(map(self.predict_single, examples))
#
#Tree.predict_leaf = __tree_predict_leaf
#Tree.predict_single = __tree_predict_single
#Tree.predict = __tree_predict
#
#def __addtree_iter(self):
#    for i in range(len(self)):
#        yield self[i]
#
#def __addtree_predict_single(self, example):
#    result = self.base_score
#    for tree in self:
#        result += tree.predict_single(example)
#    return result
#
#def __addtree_predict(self, examples):
#    it = enumerate(examples)
#    try:
#        import pandas as pd
#        if isinstance(examples, pd.DataFrame):
#            it = examples.iterrows()
#    except: pass
#
#    predictions = []
#    #print("predicting...", end="")
#    for i, example in it:
#        #print("\rpredicting...", i, "/", len(examples), end="")
#        predictions.append(self.predict_single(example))
#    #print("\rdone                    ")
#    return predictions
#
#def __addtree_write(self, f):
#    with gzip.open(f, "wb") as fh:
#        json = self.to_json()
#        fh.write(json.encode("utf-8"))
#
#def __addtree_read(f):
#    try:
#        with gzip.open(f, "rb") as fh:
#            json = fh.read()
#            return AddTree.from_json(json.decode("utf-8"))
#    except OSError as e: # if file is not gzip encoded
#        with open(f, "r") as fh:
#            return AddTree.from_json(fh.read())
#
#AddTree.__iter__ = __addtree_iter
#AddTree.predict_single = __addtree_predict_single
#AddTree.predict = __addtree_predict
#AddTree.write = __addtree_write
#AddTree.read = __addtree_read
#
#def __solution_output(self, instance):
#    if instance==0:
#        return self.output0
#    elif instance==1:
#        return self.output1
#    else:
#        return (self.output0, self.output1)
#
#Solution.output = __solution_output
#
#
#class AddTreeFeatureTypes:
#    def __init__(self, at):
#        self._types = dict()
#
#        for tree_index in range(len(at)):
#            tree = at[tree_index]
#            self._check_types(tree, tree.root())
#
#        # sort by key
#        self._types = {fid: self._types[fid] for fid in sorted(self._types)}
#
#    def _check_types(self, tree, node):
#        l, r = tree.left(node), tree.right(node)
#
#        if not tree.is_internal(node): return
#
#        split = tree.get_split(node)
#        split_type = type(split)
#
#        if split.feat_id in self._types and self._types[split.feat_id] != split_type:
#            raise RuntimeError(f"AddTree split type error for feat_id {split.feat_id}")
#
#        self._types[split.feat_id] = split_type
#
#        if not tree.is_leaf(l): self._check_types(tree, l)
#        if not tree.is_leaf(r): self._check_types(tree, r)
#
#    def feat_ids(self):
#        yield from self._types.keys()
#
#    def __iter__(self):
#        yield from self._types.items()
#
#    def __getitem__(self, feat_id):
#        if feat_id not in self._types:
#            raise KeyError(f"unknown feat_id {feat_id}")
#        return self._types[feat_id]
#
#def get_xvar_id_map(feat_info=None, instance=None):
#    """ deprecated """
#    if isinstance(feat_info, Optimizer):
#        feat_info = feat_info.feat_info
#    if instance is not None:
#        feat_ids = feat_info.feat_ids0() if instance==0 else feat_info.feat_ids1()
#        return { feat_info.get_id(instance, fid) : fid for fid in feat_ids }
#    else:
#        idmap = {}
#        for fid in feat_info.feat_ids0():
#            idmap[feat_info.get_id(0, fid)] = fid
#        for fid in feat_info.feat_ids1():
#            idmap[feat_info.get_id(1, fid)] = fid
#        return idmap
#
#def get_closest_example(xvar_id_map, base_example, doms, delta=1e-5):
#    """ deprecated: use Optimizer.get_closest_example """
#    example = base_example.copy()
#
#    for id, dom in doms.items():
#        if not isinstance(dom, RealDomain):
#            dom = RealDomain(dom[0], dom[1])
#        feat_id = xvar_id_map[id]
#        v = example[feat_id]
#        if dom.contains(v):
#            continue # keep the value
#
#        dist_lo = abs(dom.lo - v)
#        dist_hi = abs(v - dom.hi)
#        if dist_lo > dist_hi:
#            example[feat_id] = dom.hi - delta # hi is not included
#        else:
#            example[feat_id] = dom.lo
#
#    return example
#
#def get_example_box_smt(opt, instance, example, eps):
#    smt = StringIO()
#    if not isinstance(instance, list):
#        instance = [instance]
#
#    if 0 in instance:
#        for feat_id in opt.feat_info.feat_ids0():
#            v = example[feat_id]
#            print(f"(assert (< {{f{feat_id}}} {v+eps}))", file=smt)
#            print(f"(assert (>= {{f{feat_id}}} {v-eps}))", file=smt)
#    if 1 in instance:
#        for feat_id in opt.feat_info.feat_ids1():
#            v = example[feat_id]
#            print(f"(assert (< {{g{feat_id}}} {v+eps}))", file=smt)
#            print(f"(assert (>= {{g{feat_id}}} {v-eps}))", file=smt)
#
#    return smt.getvalue()
#
#TRUE_DOMAIN = RealDomain.from_lo(1.0);
#FALSE_DOMAIN = RealDomain.from_hi_exclusive(1.0);
#
#
##class Optimizer:
#    #def parallel(self, num_threads):
#    #    """ use with with-statement """
#    #    return ParallelOptimizer(self, num_threads)
#
#
##class ParallelOptimizer:
##    def __init__(self, opt, num_threads):
##        self.opt = opt
##        self.paropt = self.opt.opt.parallel(num_threads)
##        self.paropt.set_box_adjuster(self.opt.adjuster)
##        self.bounds = opt.bounds + [self.paropt.current_bounds()]
##        self.memory = opt.memory + [self.paropt.current_memory()]
##        self.clique_count = opt.clique_count + [self.num_candidate_cliques()]
##        self.start_time = opt.start_time + opt.times[-1]
##        self.times = opt.times + [opt.times[-1]]
##
##    def __enter__(self):
##        return self
##    def __exit__ (self, type, value, tb):
##        self.paropt.join_all()
##
##    def num_threads(self): return self.paropt.num_threads()
##    def redistribute_work(self): self.paropt.redistribute_work()
##    def num_solutions(self): return self.paropt.num_solutions()
##    def num_new_valid_solutions(self): return self.paropt.num_new_valid_solutions()
##    def num_candidate_cliques(self): return self.paropt.num_candidate_cliques()
##    def current_bounds(self): return self.paropt.current_bounds()
##    def current_memory(self): return self.paropt.current_memory()
##    def join_all(self): self.paropt.join_all()
##    def get_eps(self): return self.paropt.get_eps()
##    def set_eps(self, new_eps): self.paropt.set_eps(new_eps)
##    def worker_opt(self, i): return self.paropt.worker_opt(i)
##    def steps_for(self, millis, **kwargs):
##        self.paropt.steps_for(millis, **kwargs)
##        self.bounds.append(self.paropt.current_bounds())
##        self.memory.append(self.paropt.current_memory())
##        self.times.append(timeit.default_timer() - self.start_time)
##
##    def solutions(self):
##        solutions = []
##        for i in range(self.num_threads()):
##            wopt = self.worker_opt(i)
##            solutions += wopt.solutions
##        solutions.sort(key=lambda s: s.time)
##        return solutions
#
