# Copyright 2019 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import timeit
from io import StringIO

from .pytreeck import *

def __realdomain__str(self):
    return "[{:.3g}, {:.3g})".format(self.lo, self.hi)

def __realdomain__eq(self, o):
    return self.lo == o.lo and self.hi == o.hi

def __realdomain__hash(self):
    return hash((self.lo, self.hi))

def __booldomain__str(self):
    return "{True}" if self.is_true() \
        else "{False}" if self.is_false() \
        else "{False, True}"

def __booldomain__eq(self, o):
    return self._value == o._value

def __booldomain__hash(self):
    return hash(self._value)

RealDomain.__str__ = __realdomain__str
RealDomain.__eq__ = __realdomain__eq
RealDomain.__hash__ = __realdomain__hash
BoolDomain.__str__ = __booldomain__str
BoolDomain.__eq__ = __booldomain__eq
BoolDomain.__hash__ = __booldomain__hash

def __tree_predict_leaf(self, example):
    node = self.root()
    while not self.is_leaf(node):
        split = self.get_split(node)
        value = example[split.feat_id]
        if isinstance(split, LtSplit):
            #assert isinstance(value, float), f"is {type(value)} instead"
            go_left = split.test(value)
        elif isinstance(split, BoolSplit):
            #assert isinstance(value, bool), f"is {type(value)} instead"
            go_left = split.test(value)
        node = self.left(node) if go_left else self.right(node)
    return node

def __tree_predict_single(self, example):
    node = __tree_predict_leaf(self, example)
    return self.get_leaf_value(node)

def __tree_predict(self, examples):
    return list(map(self.predict_single, examples))

Tree.predict_leaf = __tree_predict_leaf
Tree.predict_single = __tree_predict_single
Tree.predict = __tree_predict

def __addtree_iter(self):
    for i in range(len(self)):
        yield self[i]

def __addtree_predict_single(self, example):
    result = self.base_score
    for tree in self:
        result += tree.predict_single(example)
    return result

def __addtree_predict(self, examples):
    it = enumerate(examples)
    try:
        import pandas as pd
        if isinstance(examples, pd.DataFrame):
            it = examples.iterrows()
    except: pass

    predictions = []
    #print("predicting...", end="")
    for i, example in it:
        #print("\rpredicting...", i, "/", len(examples), end="")
        predictions.append(self.predict_single(example))
    #print("\rdone                    ")
    return predictions

def __addtree_write(self, f):
    with open(f, "w") as fh:
        json = self.to_json()
        fh.write(json)

def __addtree_read(f):
    with open(f, "r") as fh:
        json = fh.read()
        return AddTree.from_json(json)

AddTree.__iter__ = __addtree_iter
AddTree.predict_single = __addtree_predict_single
AddTree.predict = __addtree_predict
AddTree.write = __addtree_write
AddTree.read = __addtree_read


class AddTreeFeatureTypes:
    def __init__(self, at):
        self._types = dict()

        for tree_index in range(len(at)):
            tree = at[tree_index]
            self._check_types(tree, tree.root())

        # sort by key
        self._types = {fid: self._types[fid] for fid in sorted(self._types)}

    def _check_types(self, tree, node):
        l, r = tree.left(node), tree.right(node)

        if not tree.is_internal(node): return

        split = tree.get_split(node)
        split_type = type(split)

        if split.feat_id in self._types and self._types[split.feat_id] != split_type:
            raise RuntimeError(f"AddTree split type error for feat_id {split.feat_id}")

        self._types[split.feat_id] = split_type

        if not tree.is_leaf(l): self._check_types(tree, l)
        if not tree.is_leaf(r): self._check_types(tree, r)

    def feat_ids(self):
        yield from self._types.keys()

    def __iter__(self):
        yield from self._types.items()

    def __getitem__(self, feat_id):
        if feat_id not in self._types:
            raise KeyError(f"unknown feat_id {feat_id}")
        return self._types[feat_id]

def get_xvar_id_map(opt, instance):
    feat_ids = opt.feat_info.feat_ids0() if instance==0 else opt.feat_info.feat_ids1()
    return { opt.feat_info.get_id(instance, fid) : fid for fid in feat_ids }

def get_closest_example(xvar_id_map, base_example, doms, delta=1e-5):
    example = base_example.copy()

    for id, dom in doms.items():
        assert isinstance(dom, RealDomain)
        feat_id = xvar_id_map[id]
        v = example[feat_id]
        if dom.contains(v):
            continue # keep the value

        dist_lo = abs(dom.lo - v)
        dist_hi = abs(v - dom.hi)
        if dist_lo > dist_hi:
            example[feat_id] = dom.hi - delta # hi is not included
        else:
            example[feat_id] = dom.lo

    return example

def get_example_box_smt(opt, instance, example, eps):
    smt = StringIO()
    if not isinstance(instance, list):
        instance = [instance]

    if 0 in instance:
        for feat_id in opt.feat_info.feat_ids0():
            v = example[feat_id]
            print(f"(assert (< {{f{feat_id}}} {v+eps}))", file=smt)
            print(f"(assert (>= {{f{feat_id}}} {v-eps}))", file=smt)
    if 1 in instance:
        for feat_id in opt.feat_info.feat_ids1():
            v = example[feat_id]
            print(f"(assert (< {{g{feat_id}}} {v+eps}))", file=smt)
            print(f"(assert (>= {{g{feat_id}}} {v-eps}))", file=smt)

    return smt.getvalue()

class Optimizer:
    def __init__(self, **kwargs):
        self.at0 = AddTree(); # dummy tree
        self.at1 = AddTree(); # dummy tree

        if "minimize" in kwargs: self.at0 = kwargs["minimize"]
        if "maximize" in kwargs: self.at1 = kwargs["maximize"]

        matches = set()
        match_is_reuse = True
        if "matches" in kwargs:
            matches = kwargs["matches"]
        if "match_is_reuse" in kwargs:
            match_is_reuse = kwargs["match_is_reuse"]
        self.feat_info = FeatInfo(self.at0, self.at1, matches, match_is_reuse)

        self.g0 = KPartiteGraph(self.at0, self.feat_info, 0)
        self.g1 = KPartiteGraph(self.at1, self.feat_info, 1)

        self.max_memory = 1024*1024*1024*1 # 1 gb default
        if "max_memory" in kwargs:
            self.max_memory = kwargs["max_memory"]
            self.g0.set_max_mem_size(mem)
            self.g1.set_max_mem_size(mem)

        self.ara_eps = 1.0 # just A*
        self.ara_eps_incr = 0.0
        if "ara_eps" in kwargs:
            self.ara_eps = kwargs["ara_eps"]
        if "ara_eps_incr" in kwargs:
            self.ara_eps_incr = kwargs["ara_eps_incr"]

        self.use_dyn_prog_heuristic = False
        if "use_dyn_prog_heuristic" in kwargs:
            self.use_dyn_prog_heuristic = kwargs["use_dyn_prog_heuristic"]

        self.reset_optimizer()

    def reset_optimizer(self):
        self.opt = KPartiteGraphOptimize(self.g0, self.g1)
        self.opt.set_max_mem_size(self.max_memory)
        self.opt.set_eps(self.ara_eps, self.ara_eps_incr)
        if self.use_dyn_prog_heuristic:
            self.opt.use_dyn_prog_heuristic()
        self.bounds = []
        self.memory = []
        self.times = []
        self.start_time = timeit.default_timer()

    def merge(self, K):
        self.g0.merge(K)
        self.g1.merge(K)
        self.reset_optimizer()

    def prune_example(self, example, delta):
        self.g0.prune_example(self.feat_info, example, delta)
        self.g1.prune_example(self.feat_info, example, delta)
        self.reset_optimizer()

    def prune_smt(self, smt, var_prefix0="f", var_prefix1="g"): # (assert (< {<var_prefix><feat_id> ...}))
        solver = SMTSolver(self.feat_info, self.at0, self.at1)
        #print("before", smt)
        b = StringIO()
        var_open, var_close = -1, -1
        for i, c in enumerate(smt):
            if c == "{":
                var_open = i+1
            elif c == "}":
                assert var_open != -1
                var_close = i
                var = smt[var_open:var_close]
                if var.startswith(var_prefix0):
                    feat_id = int(var[len(var_prefix0):])
                    #print("var0:", var, feat_id)
                    b.write(solver.xvar_name(0, feat_id))
                elif var.startswith(var_prefix1):
                    feat_id = int(var[len(var_prefix1):])
                    #print("var1:", var, feat_id)
                    b.write(solver.xvar_name(1, feat_id))
                else:
                    raise RuntimeError(f"invalid SMT variable {var}, prefixes are {var_prefix0} and {var_prefix1}")
                var_open, var_close = -1, -1
            elif var_open == -1 and var_close == -1:
                b.write(c)
        smt = b.getvalue()
        #print("after", smt)
        solver.parse_smt(smt)
        self.g0.prune_smt(solver)
        self.g1.prune_smt(solver)
        self.reset_optimizer()

    def num_solutions(self): return self.opt.num_solutions()
    def solutions(self): return self.opt.solutions
    def epses(self): return self.opt.epses
    def num_steps(self): return self.opt.num_steps
    def num_update_fails(self): return self.opt.num_update_fails
    def num_rejected(self): return self.opt.num_rejected
    def num_box_filter_calls(self): return self.opt.num_box_filter_calls
    def current_bounds(self): return self.opt.current_bounds()
    def current_memory(self):
        return self.g0.get_mem_size() + self.g1.get_mem_size() + self.opt.get_mem_size()

    def steps(self, num_steps, **kwargs):
        value = self.opt.steps(num_steps, **kwargs)
        self.bounds.append(self.current_bounds())
        self.memory.append(self.current_memory())
        self.times.append(timeit.default_timer() - self.start_time)
        return value

    def parallel(self, num_threads):
        return ParallelOptimizer(self, num_threads)
        

class ParallelOptimizer:
    def __init__(self, opt, num_threads):
        self.opt = opt
        self.paropt = self.opt.opt.parallel(num_threads)
        self.bounds = []
        self.memory = []
        self.times = []
        self.start_time = timeit.default_timer()

    def num_threads(self): return self.paropt.num_threads()
    def redistribute_work(self): self.paropt.redistribute_work()
    def num_solutions(self): return self.paropt.num_solutions()
    def current_bounds(self): return self.paropt.current_bounds()
    def current_memory(self): return self.paropt.current_memory()
    def join_all(self): self.paropt.join_all()
    def worker_opt(self, i): return self.paropt.worker_opt(i)
    def steps_for(self, millis, **kwargs):
        self.paropt.steps_for(millis, **kwargs)
        self.bounds.append(self.paropt.current_bounds())
        self.memory.append(self.paropt.current_memory())
        self.times.append(timeit.default_timer() - self.start_time)

    def solutions(self):
        solutions = []
        epses = []
        times = []
        for i in range(self.num_threads()):
            wopt = self.worker_opt(i)
            solutions += wopt.solutions
            epses += wopt.epses
            times += wopt.times
        t = tuple(map(list, zip(*sorted(zip(solutions, epses, times), key=lambda p: p[2]))))
        solutions, epses, times = t

        print(len(solutions), len(epses), len(times))
        return solutions, epses, times

