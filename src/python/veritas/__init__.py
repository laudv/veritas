# Copyright 2020 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import timeit, gzip, types
from io import StringIO

from .pyveritas import *

def __realdomain__str(self):
    return "[{:.3g}, {:.3g}]".format(self.lo, self.hi)

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
    with gzip.open(f, "wb") as fh:
        json = self.to_json()
        fh.write(json.encode("utf-8"))

def __addtree_read(f):
    try:
        with gzip.open(f, "rb") as fh:
            json = fh.read()
            return AddTree.from_json(json.decode("utf-8"))
    except OSError as e: # if file is not gzip encoded
        with open(f, "r") as fh:
            return AddTree.from_json(fh.read())

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

def get_xvar_id_map(feat_info=None, instance=None):
    if isinstance(feat_info, Optimizer):
        feat_info = feat_info.feat_info
    if instance is not None:
        feat_ids = feat_info.feat_ids0() if instance==0 else feat_info.feat_ids1()
        return { feat_info.get_id(instance, fid) : fid for fid in feat_ids }
    else:
        idmap = {}
        for fid in feat_info.feat_ids0():
            idmap[feat_info.get_id(0, fid)] = fid
        for fid in feat_info.feat_ids1():
            idmap[feat_info.get_id(1, fid)] = fid
        return idmap

def get_closest_example(xvar_id_map, base_example, doms, delta=1e-5):
    example = base_example.copy()

    for id, dom in doms.items():
        if not isinstance(dom, RealDomain):
            dom = RealDomain(dom[0], dom[1])
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

TRUE_DOMAIN = RealDomain.from_lo(1.0);
FALSE_DOMAIN = RealDomain.from_hi_exclusive(1.0);

class Optimizer:
    def __init__(self, **kwargs):
        self.at0 = AddTree(); # dummy tree
        self.at1 = AddTree(); # dummy tree

        if "minimize" in kwargs:
            self.at0 = kwargs["minimize"]
            del kwargs["minimize"]
        if "maximize" in kwargs:
            self.at1 = kwargs["maximize"]
            del kwargs["maximize"]

        matches = set()
        match_is_reuse = True
        if "matches" in kwargs:
            matches = kwargs["matches"]
            del kwargs["matches"]
        if "match_is_reuse" in kwargs:
            match_is_reuse = kwargs["match_is_reuse"]
            del kwargs["match_is_reuse"]
        self.feat_info = FeatInfo(self.at0, self.at1, matches, match_is_reuse)

        self.box_checker = BoxChecker(self.feat_info.num_ids(), 5)

        self.g0 = KPartiteGraph(self.at0, self.feat_info, 0)
        self.g1 = KPartiteGraph(self.at1, self.feat_info, 1)

        self.max_memory = 1024*1024*1024*1 # 1 gb default
        if "max_memory" in kwargs:
            self.max_memory = kwargs["max_memory"]
            self.g0.set_max_mem_size(self.max_memory)
            self.g1.set_max_mem_size(self.max_memory)
            del kwargs["max_memory"]

        self.ara_eps = 1.0 # just A*
        if "ara_eps" in kwargs:
            self.ara_eps = kwargs["ara_eps"]
            del kwargs["ara_eps"]

        self.heuristic = KPartiteGraphOptimizeHeuristic.RECOMPUTE
        if "use_dyn_prog_heuristic" in kwargs:
            if kwargs["use_dyn_prog_heuristic"]:
                self.heuristic = KPartiteGraphOptimizeHeuristic.DYN_PROG
            del kwargs["use_dyn_prog_heuristic"]

        for k, v in kwargs.items():
            raise RuntimeError(f"invalid argument '{k}'")

        self.reset_optimizer()

    def reset_optimizer(self):
        self.opt = KPartiteGraphOptimize(self.g0, self.g1, self.heuristic)
        self.opt.set_max_mem_size(self.max_memory)
        self.opt.set_eps(self.ara_eps)
        self.start_time = timeit.default_timer()
        bb = self.current_basic_bounds()
        self.bounds = [(bb[0][0], bb[1][1])]
        self.memory = [self.current_memory()]
        self.clique_count = [self.num_candidate_cliques()]
        self.times = [0.0]

    def prune_example(self, example, delta):
        self.g0.prune_example(self.feat_info, example, delta)
        self.g1.prune_example(self.feat_info, example, delta)
        self.reset_optimizer()

    def prune_box(self, box, instance):
        if instance == 0:
            self.g0.prune_box(self.feat_info, box, 0)
        elif instance == 1:
            self.g1.prune_box(self.feat_info, box, 1)
        else: raise RuntimeError("invalid instance")
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
    def num_steps(self): return self.opt.num_steps
    def num_update_fails(self): return self.opt.num_update_fails
    def num_rejected(self): return self.opt.num_rejected
    def num_box_checks(self): return self.opt.num_box_checks
    def num_candidate_cliques(self): return self.opt.num_candidate_cliques()
    def current_bounds(self): return self.opt.current_bounds()
    def current_basic_bounds(self): return (self.g0.basic_bound(), self.g1.basic_bound())
    def get_mem_size(self): return self.opt.get_mem_size()
    def get_used_mem_size(self): return self.opt.get_used_mem_size()
    def get_eps(self): return self.opt.get_eps()
    def set_eps(self, new_eps): self.opt.set_eps(new_eps)
    def current_memory(self):
        return self.g0.get_used_mem_size() + self.g1.get_used_mem_size() + self.opt.get_used_mem_size()

    def steps(self, num_steps, **kwargs):
        value = self.opt.steps(num_steps, box_checker=self.box_checker, **kwargs)
        self.bounds.append(self.current_bounds())
        self.memory.append(self.current_memory())
        self.clique_count.append(self.num_candidate_cliques())
        self.times.append(timeit.default_timer() - self.start_time)
        return value

    def astar(self, max_time=10,
            min_num_steps=10, max_num_steps=1000,
            steps_kwargs={}):
        done = False
        oom = False
        num_steps = min_num_steps
        start = timeit.default_timer()
        stop = start + max_time
        while not done and self.num_solutions() == 0 \
                and timeit.default_timer() < stop:
            try:
                done = not self.steps(num_steps, **steps_kwargs)
                num_steps = min(max_num_steps, num_steps * 2)
            except Exception as e:
                print("A* OUT OF MEMORY", type(e))
                done = True
                oom = True
        dur = timeit.default_timer() - start
        return dur, oom

    def arastar(self, max_time=10,
            eps_start=0.1, eps_incr=0.1,
            min_num_steps=10, max_num_steps=1000,
            steps_kwargs={}):

        if isinstance(eps_incr, float):
            eps_incr_fun = lambda eps: min(1.0, eps+eps_incr)
        elif isinstance(eps_incr, types.FunctionType):
            eps_incr_fun = eps_incr
        else: raise ValueError("invalid eps_incr parameter")

        self.set_eps(eps_start)

        done = False
        oom = False
        num_steps = min_num_steps
        start = timeit.default_timer()
        stop = start + max_time
        solution_count = 0
        while not done and timeit.default_timer() < stop \
                and not (self.get_eps() == 1.0 \
                and solution_count < self.num_solutions()):
            if solution_count < self.num_solutions():
                eps = self.get_eps()
                self.set_eps(eps_incr_fun(eps))
                print(f"ARA* eps: {eps} -> {self.get_eps()}")
                solution_count = self.num_solutions()
                num_steps = min_num_steps
            try:
                done = not self.steps(num_steps, **steps_kwargs)
                num_steps = min(max_num_steps, num_steps * 2)
            except Exception as e:
                print("ARA* OUT OF MEMORY", type(e))
                done = True
                oom = True
        dur = timeit.default_timer() - start
        return dur, oom

    def arastar_multiple_solutions(self, num_solutions, eps_margin=0.05,
            max_time=10, min_num_steps=10, max_num_steps=1000,
            steps_kwargs={}, **arastar_kwargs):

        dur, oom = self.arastar(max_time=max_time, min_num_steps=min_num_steps,
                max_num_steps=max_num_steps, steps_kwargs=steps_kwargs,
                **arastar_kwargs)

        if self.get_eps() != 1.0:
            # we found one solution for the previous eps, not certain we will
            # also be able to do that for the current eps
            self.set_eps(self.get_eps() - eps_margin)

        done = False
        if oom:
            done = True

        num_steps = min_num_steps * 4
        start = timeit.default_timer()
        stop = start + max_time
        while not done \
                and self.num_solutions() < num_solutions \
                and timeit.default_timer() < stop:
            print("generating more solutions...", self.num_solutions())
            try:
                self.steps(num_steps, **steps_kwargs)
                num_steps = min(max_num_steps, num_steps * 2)
            except Exception as e:
                print("ARA* OUT OF MEMORY", type(e))
                done = True
                oom = True
        print("done generating more solutions", self.num_solutions())

        dur += timeit.default_timer() - start

        return dur, oom

    def solution_to_intervals(self, solution, num_attributes):
        box = solution.box()
        intervals = [[None for i in range(num_attributes)], [None for i in range(num_attributes)]]
        for instance in [0, 1]:
            for feat_id in range(num_attributes):
                i = self.feat_info.get_id(instance, feat_id) # internal VERITAS id for instance
                if i != -1 and i in box:
                    if self.feat_info.is_real(i):
                        interval = (box[i].lo, box[i].hi)
                    else:
                        interval = box[i].lo == 1.0 # (-inf, 1.0) == False, (1.0, inf) == True
                    intervals[instance][feat_id] = interval
        return intervals

    #def parallel(self, num_threads):
    #    """ use with with-statement """
    #    return ParallelOptimizer(self, num_threads)
        

#class ParallelOptimizer:
#    def __init__(self, opt, num_threads):
#        self.opt = opt
#        self.paropt = self.opt.opt.parallel(num_threads)
#        self.paropt.set_box_adjuster(self.opt.adjuster)
#        self.bounds = opt.bounds + [self.paropt.current_bounds()]
#        self.memory = opt.memory + [self.paropt.current_memory()]
#        self.clique_count = opt.clique_count + [self.num_candidate_cliques()]
#        self.start_time = opt.start_time + opt.times[-1]
#        self.times = opt.times + [opt.times[-1]]
#
#    def __enter__(self):
#        return self
#    def __exit__ (self, type, value, tb):
#        self.paropt.join_all()
#
#    def num_threads(self): return self.paropt.num_threads()
#    def redistribute_work(self): self.paropt.redistribute_work()
#    def num_solutions(self): return self.paropt.num_solutions()
#    def num_new_valid_solutions(self): return self.paropt.num_new_valid_solutions()
#    def num_candidate_cliques(self): return self.paropt.num_candidate_cliques()
#    def current_bounds(self): return self.paropt.current_bounds()
#    def current_memory(self): return self.paropt.current_memory()
#    def join_all(self): self.paropt.join_all()
#    def get_eps(self): return self.paropt.get_eps()
#    def set_eps(self, new_eps): self.paropt.set_eps(new_eps)
#    def worker_opt(self, i): return self.paropt.worker_opt(i)
#    def steps_for(self, millis, **kwargs):
#        self.paropt.steps_for(millis, **kwargs)
#        self.bounds.append(self.paropt.current_bounds())
#        self.memory.append(self.paropt.current_memory())
#        self.times.append(timeit.default_timer() - self.start_time)
#
#    def solutions(self):
#        solutions = []
#        for i in range(self.num_threads()):
#            wopt = self.worker_opt(i)
#            solutions += wopt.solutions
#        solutions.sort(key=lambda s: s.time)
#        return solutions

