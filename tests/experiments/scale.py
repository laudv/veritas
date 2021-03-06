import os, sys, json, io
import time, timeit
import pickle
import multiprocessing as mp

import xgboost as xgb
import pandas as pd
import numpy as np

import sklearn.metrics as metrics
from scipy.special import logit, expit as logistic

from veritas import *
from veritas.xgb import addtree_from_xgb_model

import util

class ScaleExperiment:

    def __init__(self,
            max_memory=1024*1024*1024*1,
            max_time = 60,
            num_threads = 1):
        self.result_dir = type(self).result_dir
        self.max_memory = max_memory
        self.num_threads = num_threads
        self.max_time = max_time

        self.min_num_steps = 10
        self.max_num_steps = 10000
        self.min_time_per_step = 25
        self.max_time_per_step = 2000
        self.do_merge = True
        self.merge_only = False

        self.steps_kwargs = {}

        self.results = {}

    def load_model(self):
        raise RuntimeError("override")

    def get_opt(self):
        raise RuntimeError("override")

    def astar_single(self):
        opt = self.get_opt()
        done = False
        oom = False
        num_steps = self.min_num_steps
        start = timeit.default_timer()
        stop = start + self.max_time
        while not done and opt.num_solutions() == 0 \
                and timeit.default_timer() < stop:
            try:
                done = not opt.steps(num_steps, **self.steps_kwargs)
                num_steps = min(self.max_num_steps, num_steps * 2)
            except Exception as e:
                print("A* OUT OF MEMORY", type(e))
                done = True
                oom = True
        dur = timeit.default_timer() - start
        return opt, dur, oom

    def astar_parallel(self):
        opt = self.get_opt()

        time_per_step = self.min_time_per_step
        start = timeit.default_timer()
        stop = start + self.max_time
        oom = False

        # just a few steps so we have some work to share between workers
        opt.steps(self.min_num_steps, **self.steps_kwargs)

        try:
            paropt = opt.parallel(self.num_threads)
            while paropt.num_solutions() == 0 \
                    and not paropt.num_candidate_cliques() == 0 \
                    and timeit.default_timer() < stop:
                try:
                    paropt.steps_for(time_per_step, **self.steps_kwargs)
                    time_per_step = min(self.max_time_per_step, int(time_per_step * 1.5))
                except Exception as e:
                    print("A* OUT OF MEMORY", type(e))
                    oom = True
                    break
        finally:
            paropt.join_all()

        dur = timeit.default_timer() - start
        return paropt, opt, dur, oom

    def arastar_single(self, start_eps, incr_eps_fun):
        opt = self.get_opt()
        opt.set_eps(start_eps)
        done = False
        oom = False
        num_steps = self.min_num_steps
        start = timeit.default_timer()
        stop = start + self.max_time
        solution_count = 0
        while not done and timeit.default_timer() < stop \
                and not (opt.get_eps() == 1.0 \
                and solution_count < opt.num_solutions()):
            if solution_count < opt.num_solutions():
                eps = opt.get_eps()
                new_eps = incr_eps_fun(eps)
                opt.set_eps(new_eps)
                print(f"ARA* eps: {eps} -> {opt.get_eps()}")
                solution_count = opt.num_solutions()
                num_steps = self.min_num_steps
            try:
                done = not opt.steps(num_steps, **self.steps_kwargs)
                num_steps = min(self.max_num_steps, num_steps * 2)
            except Exception as e:
                print("ARA* OUT OF MEMORY", type(e))
                done = True
                oom = True
        dur = timeit.default_timer() - start
        return opt, dur, oom

    def arastar_single_multiple_solutions(self, start_eps, incr_eps_fun, num_solutions):
        opt, dur, oom = self.arastar_single(start_eps, incr_eps_fun)
        if opt.get_eps() != 1.0:
            # we found one solution for the previous eps, not certain we will
            # also be able to do that for the current eps
            opt.set_eps(opt.get_eps() - 0.05)

        done = False
        if oom:
            done = True

        num_steps = self.min_num_steps * 4
        current_num_solutions = opt.num_solutions()
        start = timeit.default_timer()
        stop = start + self.max_time
        while not done \
                and opt.num_solutions() - current_num_solutions < num_solutions \
                and timeit.default_timer() < stop:
            print("generating more solutions...", opt.num_solutions())
            try:
                opt.steps(num_steps, **self.steps_kwargs)
                num_steps = min(self.max_num_steps, num_steps * 2)
            except Exception as e:
                print("ARA* OUT OF MEMORY", type(e))
                done = True
                oom = True
        print("done generating more solutions", opt.num_solutions())

        dur += timeit.default_timer() - start

        info = self._extract_info(opt, dur, oom)
        sols = opt.solutions()[current_num_solutions:]
        info["ara_solutions"] = [(s.output0, s.output1) for s in sols]
        info["ara_solution_boxes"] = [{i: (d.lo, d.hi) for i, d in s.box().items()}
                                                       for s in sols]
        return info
    
    def arastar_parallel(self, start_eps, incr_eps_fun):
        opt = self.get_opt()
        opt.set_eps(start_eps)

        time_per_step = self.min_time_per_step
        start = timeit.default_timer()
        stop = start + self.max_time
        oom = False

        # just a few steps so we have some work to share between workers
        opt.steps(self.min_num_steps, **self.steps_kwargs)
        try:
            paropt = opt.parallel(self.num_threads)
            while not paropt.num_candidate_cliques() == 0 \
                    and timeit.default_timer() < stop \
                    and not (paropt.get_eps() == 1.0 \
                    and paropt.num_new_valid_solutions() > 0):
                if paropt.num_new_valid_solutions() > 0:
                    eps = paropt.get_eps()
                    paropt.set_eps(incr_eps_fun(eps))
                    print(f"ARA* eps: {eps} -> {paropt.get_eps()}")
                    time_per_step = self.min_time_per_step
                try:
                    paropt.steps_for(time_per_step, **self.steps_kwargs)
                    time_per_step = min(self.max_time_per_step, int(time_per_step * 1.5))
                except Exception as e:
                    print("ARA* OUT OF MEMORY", type(e))
                    oom = True
                    break
        finally:
            paropt.join_all()

        dur = timeit.default_timer() - start
        return paropt, opt, dur, oom

    def merge_worker_fun(self, conn):
        opt = self.get_opt()

        g = opt.g1
        g.add_with_negated_leaf_values(opt.g0)
        
        t = 0.0
        b = g.basic_bound()
        m = g.get_used_mem_size()
        v = g.num_vertices()
        conn.send(("point", t, b, m, v))
        start = timeit.default_timer()
        try:
            while True:
                try:
                    #print("MERGE worker: num_independent_sets:", g.num_independent_sets(), g.num_vertices())
                    g.merge(2)
                except Exception as e:
                    m = g.get_used_mem_size()
                    print(f"MERGE worker: OUT OF MEMORY: {m/(1024*1024):.2f} MiB", type(e))
                    conn.send(("oom", m))
                    break

                t = timeit.default_timer() - start
                b = g.basic_bound()
                m = g.get_used_mem_size()
                v = g.num_vertices()
                conn.send(("point", t, b, m, v))

                if g.num_independent_sets() <= 1:
                    conn.send(("optimal",))
                    break
        finally:
            #print("MERGE worker: closing")
            del g
            del opt
            conn.close()

    def merge(self):
        cparent, cchild = mp.Pipe()
        p = mp.Process(target=ScaleExperiment.merge_worker_fun, name="Merger", args=(self, cchild,))
        p.start()
        start = timeit.default_timer()
        times, bounds, memory, vertices = [], [], [], []
        data = {"oot": True, "oom": False, "optimal": False}
        #print("MERGE host: runtime", self.max_time)
        while timeit.default_timer() - start < self.max_time:
            has_data = cparent.poll(1)
            if has_data:
                msg = cparent.recv()
                if msg[0] == "point":
                    t, b, m, v = msg[1:]
                    #print("MERGE host: data", t, b, m, v)
                    times.append(t)
                    bounds.append(b)
                    memory.append(m)
                    vertices.append(v)
                elif msg[0] == "optimal":
                    #print("MERGE host: optimal found")
                    data["optimal"] = True
                elif msg[0] == "oom":
                    m = msg[1]
                    #print(f"MERGE host: oom ({m/(1024*1024):.2f} MiB)")
                    data["oom"] = True
                    data["oom_value"] = m
            elif p.exitcode is not None:
                data["oot"] = False
                break

        dur = timeit.default_timer() - start

        if data["oot"]:
            print("MERGE host: timeout")

        #print("MERGE host: terminating")
        p.terminate()
        cparent.close()

        data["times"] = times
        data["bounds"] = bounds
        data["memory"] = memory
        data["vertices"] = vertices
        data["total_time"] = dur

        return data

    def _extract_info(self, opt, dur, oom):
        sols = util.filter_solutions(opt)
        data = {
            "solutions": [(s.output0, s.output1) for s in sols],
            "bounds": opt.bounds,
            "bounds_times": opt.times,
            "memory": opt.memory,
            "cliques": opt.clique_count,
            "sol_times": [s.time for s in sols],
            "epses": [s.eps for s in sols],
            "total_time": dur,
            "oom": oom,
        }
        if len(sols) > 0:
            best_sol = max(sols, key=lambda s: s.output_difference())
            data["best_solution_box"] = {i: (d.lo, d.hi) for i, d in best_sol.box().items()}
        if isinstance(opt, Optimizer):
            data["num_vertices0"] = opt.g0.num_vertices()
            data["num_vertices1"] = opt.g1.num_vertices()
        #if isinstance(opt, ParallelOptimizer):
        #    data["num_vertices0"] = opt.opt.g0.num_vertices()
        #    data["num_vertices1"] = opt.opt.g1.num_vertices()
        return data

    def astar(self):
        if self.num_threads == 1:
            opt, dur, oom = self.astar_single()
            return self._extract_info(opt, dur, oom)
        else:
            paropt, opt, dur, oom = self.astar_parallel()
            return self._extract_info(paropt, dur, oom)

    def arastar(self, start_eps, eps_incr):
        if self.num_threads == 1:
            opt, dur, oom = self.arastar_single(start_eps, eps_incr)
            return self._extract_info(opt, dur, oom)
        else:
            paropt, opt, dur, oom = self.arastar_parallel(start_eps, eps_incr)
            return self._extract_info(paropt, dur, oom)

    def run(self, name, data={}, start_eps=0.5):
        def eps_incr(eps):
            return eps + (1.0-eps)/10.0 if eps < 0.99 else 1.0

        if name not in self.results:
            self.results[name] = []
        self.results[name].append(data)

        data["num_threads"] = self.num_threads
        data["max_memory"] = self.max_memory
        data["max_time"] = self.max_time

        if not self.merge_only:
            print(f"\n -- A* {time.ctime()} --")
            data["a*"] = self.astar()
            print(f"\n -- ARA* {time.ctime()} --")
            data["ara*"] = self.arastar(start_eps, eps_incr)
        if self.do_merge or self.merge_only:
            print(f"\n -- MERGE {time.ctime()} --")
            data["merge"] = self.merge()

    def confirm_write_results(self, name):
        filename = os.path.join(self.result_dir, name)
        exists = os.path.exists(filename)
        print("writing output to", filename, "(file exists!)" if exists else "")
        if input("OK? ") != "y":
            sys.exit()

    def write_results(self):
        for key, data in self.results.items():
            filename = os.path.join(self.result_dir, key)
            print("writing output to", filename)
            with open(filename, "w") as f:
                json.dump(data, f)

class CalhouseScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/calhouse"

    def load_model(self, num_trees, depth):
        print("\n=== NEW CALHOUSE MODEL ===")
        model_name = f"calhouse-{num_trees}-{depth}"
        if not os.path.isfile(os.path.join(self.result_dir, f"{model_name}.xgb")):
            X, y = util.load_openml("calhouse", data_id=537)
            print(f"training model depth={depth}, num_trees={num_trees}")
            params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "max_depth": depth,
                "seed": 14,
                "nthread": 1,
            }

            def metric(y, raw_yhat): #maximized
                return -metrics.mean_squared_error(y, raw_yhat)

            self.model, lr, metric_value = util.optimize_learning_rate(X, y,
                    params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": list(X.columns)}

            with open(os.path.join(self.result_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        return opt

class CalhouseRandomScaleExperiment(CalhouseScaleExperiment):
    def get_opt(self):
        if not hasattr(self, "X"):
            self.X, self.y = util.load_openml("calhouse", data_id=537)
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        util.randomly_prune_opt(self.X, opt, seed=self.constraint_seed)
        return opt

class CovtypeScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/covtype"

    def load_model(self, num_trees, depth):
        print("\n=== NEW COVTYPE MODEL ===")
        model_name = f"covtype-{num_trees}-{depth}"
        if not os.path.isfile(os.path.join(self.result_dir, f"{model_name}.xgb")):
            X, y = util.load_openml("covtype", data_id=1596)
            y = (y==2)
            print(f"training model depth={depth}, num_trees={num_trees}")
            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "max_depth": depth,
                "eval_metric": "error",
                "seed": 235,
                "nthread": 1,
            }

            def metric(y, raw_yhat):
                return metrics.accuracy_score(y, raw_yhat > 0)

            self.model, lr, metric_value = util.optimize_learning_rate(X, y,
                    params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value}

            with open(os.path.join(self.result_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        self.at = addtree_from_xgb_model(self.model)
        self.at.base_score = 0

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        return opt

class CovtypeRandomScaleExperiment(CovtypeScaleExperiment):
    def get_opt(self):
        if not hasattr(self, "X"):
            self.X, self.y = util.load_openml("covtype", data_id=1596)
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        util.randomly_prune_opt(self.X, opt, seed=self.constraint_seed)
        return opt

class ConstrainedCovtypeScaleExperiment1(CovtypeScaleExperiment):
    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        feat_ids = opt.feat_info.feat_ids1()

        smt = io.StringIO()
        print("(assert (> {g0} 3200.0))", file=smt) # elevation
        print("(assert (< {g5} 1800.0))", file=smt) # hoz dist to road
        #print(f"(assert (> {opt.xvar(1, 9)} 1800.0))", file=smt) # hoz dist fire road
        for i in set(range(10, 14)).intersection(feat_ids):
            op = ">" if i == 13 else "<"
            print(f"(assert ({op} {{g{i}}} 0.5))", file=smt) # Wilderness_Area
        for i in set(range(14, 54)).intersection(feat_ids):
            op = ">" if i == 36 else "<"
            print(f"(assert ({op} {{g{i}}} 0.5))", file=smt) # Soil_Type

        print("before num_vertices", opt.g1.num_vertices())
        opt.prune_smt(smt.getvalue())
        print("after num_vertices", opt.g1.num_vertices())

        return opt

class MnistXvallScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/mnist"

    def load_model(self, num_trees, depth, label):
        print(f"\n=== NEW MNIST MODEL label {label} ===")
        self.num_trees = num_trees
        self.tree_depth = depth
        model_name = f"mnist-{label}vall-{num_trees}-{depth}"
        if not hasattr(self, "X"):
            X, y = util.load_openml("mnist", data_id=554)
        else:
            X, y = self.X, self.y
        ybin = (y==int(label))
        self.label = label # {label} vs all
        if not os.path.isfile(os.path.join(self.result_dir, f"{model_name}.xgb")):
            print(f"training model depth={depth}, num_trees={num_trees}, label={label}")
            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "max_depth": depth,
                "eval_metric": "error",
                "seed": 53589,
                "nthread": 1,
            }

            def metric(y, raw_yhat): #maximized
                return metrics.accuracy_score(y, raw_yhat > 0)
            
            #print(ybin)
            self.model, lr, metric_value = util.optimize_learning_rate(X, ybin,
                    params, num_trees, metric)
            self.meta = {"lr": lr, "metric": metric_value, "columns": list(X.columns)}

            with open(os.path.join(self.result_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        return opt

class MnistXvallPrunedScaleExperiment(MnistXvallScaleExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X, self.y = util.load_openml("mnist", data_id=554)
        self.target_at = None

    def load_example(self, example_i, delta):
        self.example_i = example_i
        self.delta = delta

        self.example = self.X.iloc[self.example_i, :]
        self.example_label = self.y[self.example_i]

    def get_opt(self):
        if self.target_at is not None:
            # share all variables
            opt = Optimizer(minimize=self.at, maximize=self.target_at, matches=set(),
                    match_is_reuse=False, max_memory=self.max_memory)
        else:
            opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        print("MNIST before num_vertices", opt.g1.num_vertices())
        opt.prune_example(list(self.example), self.delta)
        print("MNIST after num_vertices", opt.g1.num_vertices())
        return opt

class AllstateScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/allstate"

    def load_data(self):
        allstate_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "allstate.h5")
        data = pd.read_hdf(allstate_data_path)
        X = data.drop(columns=["loss"]).to_numpy(dtype=float)
        y = data.loss.to_numpy(dtype=float)
        return X, y

    def load_model(self, num_trees, depth):
        print("\n=== NEW ALLSTATE MODEL ===")
        model_name = f"allstate-{num_trees}-{depth}"
        if not os.path.isfile(os.path.join(self.result_dir, f"{model_name}.xgb")):
            print(f"training model: {model_name}")
            X, y = self.load_data()
            params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "max_depth": depth,
                "seed": 14,
                "nthread": 1,
            }

            def metric(y, raw_yhat): #maximized
                return -metrics.mean_squared_error(y, raw_yhat)
            self.model, lr, metric_value = util.optimize_learning_rate(X, y,
                    params, num_trees, metric)
            self.meta = {"lr": lr, "metric": float(metric_value), "columns": list(X.columns)}

            with open(os.path.join(self.result_dir, f"{model_name}.xgb"), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "w") as f:
                json.dump(self.meta, f)
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, f"{model_name}.xgb"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "r") as f:
                self.meta = json.load(f)

        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.at = addtree_from_xgb_model(self.model, feat2id_map=self.feat2id)
        self.at.base_score = 0

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        return opt

class AllstateRandomScaleExperiment(AllstateScaleExperiment):
    def get_opt(self):
        if not hasattr(self, "X"):
            self.X, self.y = self.load_data()
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        util.randomly_prune_opt(self.X, opt, seed=self.constraint_seed)
        return opt

class SoccerScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/soccer"

    action0 = ['type_pass_a0', 'type_cross_a0', 'type_throw_in_a0',
            'type_freekick_crossed_a0', 'type_freekick_short_a0',
            'type_corner_crossed_a0', 'type_corner_short_a0',
            'type_take_on_a0', 'type_foul_a0', 'type_tackle_a0',
            'type_interception_a0', 'type_shot_a0', 'type_shot_penalty_a0',
            'type_shot_freekick_a0', 'type_keeper_save_a0',
            'type_keeper_claim_a0', 'type_keeper_punch_a0',
            'type_keeper_pick_up_a0', 'type_clearance_a0',
            'type_bad_touch_a0', 'type_non_action_a0', 'type_dribble_a0',
            'type_goalkick_a0', 'type_receival_a0', 'type_out_a0',
            'type_offside_a0', 'type_goal_a0', 'type_owngoal_a0',
            'type_yellow_card_a0', 'type_red_card_a0']
    action1 = ['type_pass_a1', 'type_cross_a1', 'type_throw_in_a1',
            'type_freekick_crossed_a1', 'type_freekick_short_a1',
            'type_corner_crossed_a1', 'type_corner_short_a1',
            'type_take_on_a1', 'type_foul_a1', 'type_tackle_a1',
            'type_interception_a1', 'type_shot_a1', 'type_shot_penalty_a1',
            'type_shot_freekick_a1', 'type_keeper_save_a1',
            'type_keeper_claim_a1', 'type_keeper_punch_a1',
            'type_keeper_pick_up_a1', 'type_clearance_a1',
            'type_bad_touch_a1', 'type_non_action_a1', 'type_dribble_a1',
            'type_goalkick_a1', 'type_receival_a1', 'type_out_a1',
            'type_offside_a1', 'type_goal_a1', 'type_owngoal_a1',
            'type_yellow_card_a1', 'type_red_card_a1']
    bodypart0 = ['bodypart_foot_a0', 'bodypart_head_a0', 'bodypart_other_a0']
    bodypart1 = ['bodypart_foot_a1', 'bodypart_head_a1', 'bodypart_other_a1']

    # cannot be TRUE at the same time (at-most-1)
    bodypart_constraints = [
            ['type_tackle_a0', 'bodypart_head_a0'],
            ['type_tackle_a0', 'bodypart_other_a0'],
            ['type_tackle_a1', 'bodypart_head_a1'],
            ['type_tackle_a1', 'bodypart_other_a1'],
            ['type_throw_in_a0', 'bodypart_foot_a0'],
            ['type_throw_in_a0', 'bodypart_head_a0'],
            ['type_throw_in_a1', 'bodypart_foot_a1'],
            ['type_throw_in_a1', 'bodypart_head_a1'],
            ['type_shot_penalty_a0', 'bodypart_other_a0'],
            ['type_shot_penalty_a0', 'bodypart_head_a0'],
            ['type_shot_penalty_a1', 'bodypart_other_a1'],
            ['type_shot_penalty_a1', 'bodypart_head_a1'],
            ['type_freekick_crossed_a0', 'bodypart_other_a0'],
            ['type_freekick_crossed_a0', 'bodypart_head_a0'],
            ['type_freekick_crossed_a1', 'bodypart_other_a1'],
            ['type_freekick_crossed_a1', 'bodypart_head_a1'],
            ['type_freekick_short_a0', 'bodypart_other_a0'],
            ['type_freekick_short_a0', 'bodypart_head_a0'],
            ['type_freekick_short_a1', 'bodypart_other_a1'],
            ['type_freekick_short_a1', 'bodypart_head_a1'],
            ['type_corner_crossed_a0', 'bodypart_other_a0'],
            ['type_corner_crossed_a0', 'bodypart_head_a0'],
            ['type_corner_crossed_a1', 'bodypart_other_a1'],
            ['type_corner_crossed_a1', 'bodypart_head_a1'],
            ['type_corner_short_a0', 'bodypart_other_a0'],
            ['type_corner_short_a0', 'bodypart_head_a0'],
            ['type_corner_short_a1', 'bodypart_other_a1'],
            ['type_corner_short_a1', 'bodypart_head_a1'],
            ['type_dribble_a0', 'bodypart_other_a0'],
            ['type_dribble_a0', 'bodypart_head_a0'],
            ['type_dribble_a1', 'bodypart_other_a1'],
            ['type_dribble_a1', 'bodypart_head_a1'],
        ]

    def load_data(self):
        soccer_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "soccerdata.h5")
        X = pd.read_hdf(soccer_data_path, "features")
        #X["goalscore_team"] = X["goalscore_team"].astype(np.float32)
        #X["goalscore_opponent"] = X["goalscore_opponent"].astype(np.float32)
        #X["goalscore_diff"] = X["goalscore_diff"].astype(np.float32)
        y = pd.read_hdf(soccer_data_path, "labels").astype(np.float32)["scores"]

        X = X.drop(columns=["team_1", "dist_to_goal_a0", "dist_to_goal_a1",
            "angle_to_goal_a0", "angle_to_goal_a1"])
        X = X.drop(columns=["time_seconds_a0", "time_seconds_a1"])
        X = X.drop(columns=["dx_a0", "dy_a0", "dx_a1", "dy_a1"])
        X = X.drop(columns=["period_id_a0", "period_id_a1", "time_delta_1"])
        X = X.drop(columns=["goalscore_opponent", "goalscore_diff", "goalscore_team"])
        X = X.drop(columns=["time_seconds_overall_a1"])

        columns = list(X.columns)

        y = y.to_numpy(dtype=np.float32)
        X = X.to_numpy(dtype=np.float32)

        return X, y, columns

    def load_model(self, num_trees, depth, lr):
        print("\n=== NEW MODEL ===")
        model_name = f"soccer-{num_trees}-{depth}-{lr}.xgb"
        meta_name = f"soccer-{num_trees}-{depth}-{lr}.meta"
        at_name = f"soccer-{num_trees}-{depth}-{lr}.at"
        if not os.path.isfile(os.path.join(self.result_dir, model_name)):
            print(f"training model learning_rate={lr}, depth={depth}, num_trees={num_trees}")
            X, y, columns = self.load_data()

            num_examples, num_features = X.shape
            Itrain, Itest = util.train_test_indices(num_examples, seed=1)
            Xtrain = X[Itrain, :]
            ytrain = y[Itrain]
            Xtest = X[Itest, :]
            ytest = y[Itest]
            dtrain = xgb.DMatrix(Xtrain, label=ytrain, missing=None)
            dtest = xgb.DMatrix(Xtest, label=ytest, missing=None)
            ybar = ytrain.mean()
            base_score = 0.5*np.log((1.0+ybar) / (1.0-ybar))

            params = {
                "learning_rate": lr,
                "max_depth": depth,
                "objective": "binary:logistic",
                "base_score": base_score,
                "eval_metric": "auc",
                "tree_method": "hist",
                "colsample_bytree": 1.0,
                "subsample": 1.0,
                "seed": 1,
                "nthread": 4,
            }
            self.model = xgb.train(params, dtrain, num_trees, [(dtrain, "train"), (dtest, "test")],
                    early_stopping_rounds=10)
            ytest_hat = self.model.predict(dtest)

            self.feat2id_dict = dict([(v, i) for i, v in enumerate(columns)])
            self.feat2id = lambda x: self.feat2id_dict[x]
            self.id2feat = lambda i: columns[i]

            self.at = addtree_from_xgb_model(self.model)
            self.at.base_score = 0.0

            pred = self.model.predict(dtest, output_margin=True)
            pred_at = np.array(self.at.predict(Xtest[:1000]))

            brier_score = sum((logistic(pred) - ytest)**2) / len(pred)
            print("brier score", brier_score)

            mae = metrics.mean_absolute_error(pred[:1000], pred_at)
            print(f"mae model difference {mae} before base_score")

            self.at.base_score = -mae
            pred_at = np.array(self.at.predict(Xtest[:1000, :]))
            mae = metrics.mean_absolute_error(pred[:1000], pred_at)
            print(f"mae model difference {mae}")

            with open(os.path.join(self.result_dir, model_name), "wb") as f:
                pickle.dump(self.model, f)
            self.at.write(os.path.join(self.result_dir, at_name))
            self.meta = {"columns": columns}
            with open(os.path.join(self.result_dir, meta_name), "w") as f:
                json.dump(self.meta, f)

        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, model_name), "rb") as f:
                self.model = pickle.load(f)
            self.at = AddTree.read(os.path.join(self.result_dir, at_name))
            with open(os.path.join(self.result_dir, meta_name), "r") as f:
                self.meta = json.load(f)

        self.feat2id_dict = dict([(v, i) for i, v in enumerate(self.meta["columns"])])
        self.feat2id = lambda x: self.feat2id_dict[x]
        self.id2feat = lambda i: self.meta["columns"][i]

        #print(num_examples, num_features)

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory, use_dyn_prog_heuristic=False)

        u = self.feat2id
        w = lambda x: opt.feat_info.get_id(1, u(x))
        action0_ids = [w(name) for name in SoccerScaleExperiment.action0 if w(name) != -1]
        action1_ids = [w(name) for name in SoccerScaleExperiment.action1 if w(name) != -1]
        bodypart0_ids = [w(name) for name in SoccerScaleExperiment.bodypart0 if w(name) != -1]
        bodypart1_ids = [w(name) for name in SoccerScaleExperiment.bodypart1 if w(name) != -1]

        print(action0_ids)
        print(action1_ids)
        print(bodypart0_ids)
        print(bodypart1_ids)

        opt.adjuster.add_one_out_of_k(action0_ids, len(action0_ids) == len(SoccerScaleExperiment.action0))
        opt.adjuster.add_one_out_of_k(action1_ids, len(action1_ids) == len(SoccerScaleExperiment.action1))
        opt.adjuster.add_one_out_of_k(bodypart0_ids, len(bodypart0_ids) == len(SoccerScaleExperiment.bodypart0))
        opt.adjuster.add_one_out_of_k(bodypart1_ids, len(bodypart1_ids) == len(SoccerScaleExperiment.bodypart1))

        for c in SoccerScaleExperiment.bodypart_constraints:
            ids = [w(name) for name in c]
            opt.adjuster.add_at_most_k(ids, 1)

        #opt.adjuster.add_at_most_k(action0_ids, 1)
        #opt.adjuster.add_at_most_k(action1_ids, 1)
        #opt.adjuster.add_at_most_k(bodypart0_ids, 1)
        #opt.adjuster.add_at_most_k(bodypart1_ids, 1)

        box = [RealDomain() for i in range(len(self.meta["columns"]))]
        box[u("type_goal_a0")] = RealDomain(-np.inf, 1.0)
        box[u("type_goal_a1")] = RealDomain(-np.inf, 1.0)
        box[u("type_foul_a0")] = RealDomain(-np.inf, 1.0)
        box[u("type_foul_a1")] = RealDomain(-np.inf, 1.0)
        box[u("type_keeper_claim_a0")] = RealDomain(-np.inf, 1.0)
        box[u("type_keeper_claim_a1")] = RealDomain(-np.inf, 1.0)
        box[u("type_keeper_save_a0")] = RealDomain(-np.inf, 1.0)
        box[u("type_keeper_save_a1")] = RealDomain(-np.inf, 1.0)
        box[u("type_yellow_card_a1")] = RealDomain(-np.inf, 1.0)
        box[u("time_seconds_overall_a0")] = RealDomain(600.0, 1200.0)
        print("before prune", opt.g1.num_vertices())
        opt.prune_box(box, 1)
        print("after prune", opt.g1.num_vertices())
        return opt

class SoccerScaleExperiment_Backwards(SoccerScaleExperiment):
    def get_opt(self):
        opt = super().get_opt()
        u = self.feat2id
        w = lambda x: opt.feat_info.get_id(1, u(x))
        opt.adjuster.add_less_than(w("x_a0"), w("x_a1"))
        box = [RealDomain() for i in range(len(self.meta["columns"]))]
        box[u("x_a0")] = RealDomain(50, 80)
        box[u("x_a1")] = RealDomain(95, 100)
        box[u("y_a0")] = RealDomain(5, 63)
        box[u("y_a1")] = RealDomain(5, 63)

        # a0 is shot
        for a in SoccerScaleExperiment.action0:
            if a == "type_shot_a0":
                box[u(a)] = RealDomain(1.0, np.inf)
            else:
                box[u(a)] = RealDomain(-np.inf, 1.0)

        # a1 is pass
        for a in SoccerScaleExperiment.action1:
            if a == "type_pass_a1":
                box[u(a)] = RealDomain(1.0, np.inf)
            else:
                box[u(a)] = RealDomain(-np.inf, 1.0)

        # a0 is foot
        for a in SoccerScaleExperiment.bodypart0:
            if a == "bodypart_foot_a0":
                box[u(a)] = RealDomain(1.0, np.inf)
            else:
                box[u(a)] = RealDomain(-np.inf, 1.0)

        # a1 is foot
        for a in SoccerScaleExperiment.bodypart1:
            if a == "bodypart_foot_a1":
                box[u(a)] = RealDomain(1.0, np.inf)
            else:
                box[u(a)] = RealDomain(-np.inf, 1.0)

        print("before prune", opt.g1.num_vertices())
        opt.prune_box(box, 1)
        print("after prune", opt.g1.num_vertices())
        return opt

class SoccerScaleExperiment_Optimize(SoccerScaleExperiment):
    def get_opt(self):
        opt = super().get_opt()
        u = self.feat2id
        box = [RealDomain() for i in range(len(self.meta["columns"]))]
        #box[u("x_a0")] = RealDomain(80, 90)
        box[u("y_a0")] = RealDomain(20, 25)
        box[u("x_a1")] = RealDomain(50, 55)
        box[u("y_a1")] = RealDomain(5, 10)

        box[u("type_shot_penalty_a0")] = RealDomain(-np.inf, 1.0)
        box[u("type_corner_short_a0")] = RealDomain(-np.inf, 1.0)
        box[u("type_corner_crossed_a0")] = RealDomain(-np.inf, 1.0)
        box[u("type_freekick_short_a0")] = RealDomain(-np.inf, 1.0)
        box[u("type_freekick_crossed_a0")] = RealDomain(-np.inf, 1.0)
        box[u("type_shot_freekick_a0")] = RealDomain(-np.inf, 1.0)

        # a1 is pass
        for a in SoccerScaleExperiment.action1:
            if a == "type_pass_a1":
                box[u(a)] = RealDomain(1.0, np.inf)
            else:
                box[u(a)] = RealDomain(-np.inf, 1.0)

        # a1 is foot
        for a in SoccerScaleExperiment.bodypart1:
            if a == "bodypart_foot_a1":
                box[u(a)] = RealDomain(1.0, np.inf)
            else:
                box[u(a)] = RealDomain(-np.inf, 1.0)

        print("before prune", opt.g1.num_vertices())
        opt.prune_box(box, 1)
        print("after prune", opt.g1.num_vertices())
        return opt

class HiggsScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/higgs"

    def load_data(self):
        higgs_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "higgs.h5")
        X = pd.read_hdf(higgs_data_path, "X").to_numpy(dtype=float)
        y = pd.read_hdf(higgs_data_path, "y").to_numpy(dtype=float)
        return X, y

    def load_model(self, num_trees, depth):
        print("\n=== NEW HIGGS MODEL ===")
        model_name = f"higgs-{num_trees}-{depth}.xgb"
        meta_name = f"higgs-{num_trees}-{depth}.meta"
        at_name = f"higgs-{num_trees}-{depth}.at"

        if not os.path.isfile(os.path.join(self.result_dir, model_name)):
            print(f"training model: {model_name}")
            X, y = self.load_data()

            #def optimize_learning_rate(X, y, params, num_trees, metric):
            params = {
                "max_depth": depth,
                "objective": "binary:logistic",
                #"base_score": base_score,
                "eval_metric": "error",
                "tree_method": "hist",
                "seed": 1,
                "nthread": 1,
            }

            def metric(y, raw_yhat):
                return metrics.accuracy_score(y, raw_yhat > 0)

            self.model, lr, metric_value = util.optimize_learning_rate(X, y,
                    params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value}

            with open(os.path.join(self.result_dir, model_name), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.result_dir, meta_name), "w") as f:
                json.dump(self.meta, f)

        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, model_name), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.result_dir, meta_name), "r") as f:
                self.meta = json.load(f)

        self.at = addtree_from_xgb_model(self.model)
        self.at.base_score = 0

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory, use_dyn_prog_heuristic=False)
        return opt

class HiggsRandomScaleExperiment(HiggsScaleExperiment):
    def get_opt(self):
        if not hasattr(self, "X"):
            self.X, self.y = self.load_data()
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        util.randomly_prune_opt(self.X, opt, seed=self.constraint_seed)
        return opt

class YouTubeScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/youtube"

    def __init__(self, *args, **kwargs):
        if "fixed_words" in kwargs:
            self.fixed_words = kwargs["fixed_words"]
            del kwargs["fixed_words"]
        else:
            self.fixed_words = []
        self.at_most_k = 2
        super().__init__(*args, **kwargs)

    def load_data(self):
        yt_data_path = os.path.join(os.environ["VERITAS_DATA_DIR"], "youtube.h5")
        data = pd.read_hdf(yt_data_path)
        y = data["viewslg"].to_numpy(dtype=float)
        dropcolumns = [c for c in data.columns if c.count("_") > 1 and c.startswith("txt_")]
        dropcolumns += [c for c in data.columns if not c.startswith("txt_")]
        data = data.drop(columns=dropcolumns)
        X = data.to_numpy(dtype=bool)
        return X, y, list(data.columns)

    def load_model(self, num_trees, depth):
        print("\n=== NEW YOUTUBE MODEL ===")
        model_name = f"youtube-{num_trees}-{depth}.xgb"
        meta_name = f"youtube-{num_trees}-{depth}.meta"
        at_name = f"youtube-{num_trees}-{depth}.at"

        if not os.path.isfile(os.path.join(self.result_dir, model_name)):
            print(f"training model: {model_name}")
            X, y, columns = self.load_data()

            #def optimize_learning_rate(X, y, params, num_trees, metric):
            params = {
                "max_depth": depth,
                "objective": "reg:squarederror",
                #"base_score": base_score,
                "eval_metric": "mae",
                "tree_method": "hist",
                "seed": 1,
                "nthread": 4,
            }

            def metric(y, raw_yhat): #maximized
                return -metrics.mean_squared_error(y, raw_yhat)

            self.model, lr, metric_value = util.optimize_learning_rate(X, y,
                    params, num_trees, metric)

            self.meta = {"lr": lr, "metric": metric_value, "columns": columns}

            with open(os.path.join(self.result_dir, model_name), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.result_dir, meta_name), "w") as f:
                json.dump(self.meta, f)

        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, model_name), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.result_dir, meta_name), "r") as f:
                self.meta = json.load(f)

        self.at = addtree_from_xgb_model(self.model)
        self.at.base_score = 0
        feat2id_dict = {v: i for i, v in enumerate(self.meta["columns"])}
        self.feat2id = lambda x: feat2id_dict[x]
        self.id2feat = lambda i: self.meta["columns"][i]

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory, use_dyn_prog_heuristic=False)
        used_feat_ids = set(opt.feat_info.feat_ids1())
        self.fixed_word_feat_ids = set([self.feat2id(x) for x in self.fixed_words])
        #print("used", [self.meta["columns"][i] for i in used_feat_ids])
        #print("fixed", self.fixed_word_feat_ids.intersection(used_feat_ids))
        box = [RealDomain() for i in range(len(self.meta["columns"]))]
        for w, feat_id in zip(self.fixed_words, self.fixed_word_feat_ids):
            id = opt.feat_info.get_id(1, feat_id)
            print(w, ":", feat_id, "->", id)
            box[feat_id] = RealDomain(1.0, np.inf)
        for w in ["txt_la", "txt_de", "txt_mv", "txt_2017", "txt_2018", "txt_no", "txt_12"]:
            box[self.feat2id(w)] = RealDomain(-np.inf, 1.0)
        print("before prune", opt.g1.num_vertices())
        opt.prune_box(box, 1)
        print("after prune", opt.g1.num_vertices())
        other_feat_ids = used_feat_ids.difference(self.fixed_word_feat_ids)
        other_ids = [opt.feat_info.get_id(1, fid) for fid in other_feat_ids]
        #opt.adjuster.add_one_out_of_k(other_ids, True)
        opt.adjuster.add_at_most_k(other_ids, self.at_most_k)
        return opt

def calhouse(outfile, max_memory):
    exp = CalhouseScaleExperiment(max_memory=max_memory, max_time=120, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    for depth in [4]:#, 6, 8]:
        for num_trees in [25]:#, 50, 75, 100, 150, 200, 300, 400, 500]:
            exp.load_model(num_trees, depth)
            exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr":
                exp.meta["lr"]}, start_eps=0.01)
            #break
        #break
    exp.write_results()

def calhouse_random(outfile, max_memory, N, seed):
    exp = CalhouseRandomScaleExperiment(max_memory=max_memory, max_time=30, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    rng = np.random.RandomState(seed)
    for depth in [3, 4, 5, 6, 7, 8]:
        for num_trees in [25, 50, 75, 100, 150, 200]:
            for i in range(N):
                exp.constraint_seed = rng.randint(2**31)
                exp.load_model(num_trees, depth)
                exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr":
                    exp.meta["lr"], "constraint_seed": exp.constraint_seed},
                    start_eps=0.01)
    exp.write_results()

def covtype(outfile, max_memory):
    exp = CovtypeScaleExperiment(max_memory=max_memory, max_time=120, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    for depth in [4, 6, 8]:
        for num_trees in [25, 50, 75, 100, 150, 200, 300, 400, 500]:
            exp.load_model(num_trees, depth)
            exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr":
                exp.meta["lr"]}, start_eps=0.01)
            #break
        #break
    exp.write_results()

def covtype_random(outfile, max_memory, N, seed):
    exp = CovtypeRandomScaleExperiment(max_memory=max_memory, max_time=120, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    rng = np.random.RandomState(seed)
    for depth in [3, 4, 5, 6, 7, 8]:
        for num_trees in [25, 50, 75, 100, 150, 200]:
            for i in range(N):
                exp.constraint_seed = rng.randint(2**31)
                exp.load_model(num_trees, depth)
                exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr":
                    exp.meta["lr"], "constraint_seed": exp.constraint_seed},
                    start_eps=0.01)
    exp.write_results()

def mnistXvall(outfile, max_memory, label):
    exp = MnistXvallScaleExperiment(max_memory=max_memory, max_time=120, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    for depth in [4, 6, 8]:
        for num_trees in [25, 50, 75, 100, 150, 200, 250, 500]:
            exp.load_model(num_trees, depth, label)
            exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr": exp.meta["lr"]})
            #break
        #break
    exp.write_results()

def mnist_robust(outfile, max_memory, N, seed):
    exp = MnistXvallPrunedScaleExperiment(max_memory=max_memory, max_time=10, num_threads=1)
    exp.steps_kwargs["min_output_difference"] = 0.0
    exp.confirm_write_results(outfile)
    exp.do_merge = True

    rng = np.random.RandomState(seed)
    start_delta = 20.1 # avoid floating point issues (robustness for mnist is an integer anyway!)

    for example_i in rng.randint(exp.X.shape[0], size=N):
        exp.load_example(example_i, start_delta)
        for target_label in range(10):
            if target_label == exp.example_label: continue
            print(f"source {exp.example_label} vs target {target_label} FOLLOW ASTAR")
            mnist_robust_search(outfile, exp, example_i, target_label, True, start_delta, seed)
            print(f"source {exp.example_label} vs target {target_label} FOLLOW MERGE")
            mnist_robust_search(outfile, exp, example_i, target_label, False, start_delta, seed)

    exp.write_results()

def mnist_robust_search(outfile, exp, example_i, target_label, follow_astar,
        start_delta, seed):
    num_trees = exp.num_trees
    depth = exp.tree_depth

    upper = start_delta
    exp.delta = start_delta
    lower = 0.0

    def calc_example_delta(exp, box):
        feat_info = FeatInfo(exp.at, exp.target_at, set(), False)
        xvar_id_map = get_xvar_id_map(feat_info)
        x = get_closest_example(xvar_id_map, exp.example, box)
        pred_target = exp.target_at.predict_single(x)
        pred_source = exp.at.predict_single(x)
        print(f"Adv.example target {pred_target:.6f}, ({up:.6f}, {pred_target-up:.3g})")
        print(f"Adv.example source {pred_source:.6f}, ({lo:.6f}, {pred_source-lo:.3g})")
        example_delta = max(abs(x - exp.example))
        print(f"Adv.example delta", example_delta)
        return example_delta

    for step in range(10):
        exp.load_model(num_trees, depth, target_label)
        exp.target_at = exp.at
        exp.load_model(num_trees, depth, exp.example_label)
        exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr":
            exp.meta["lr"], "example_i": int(exp.example_i), "example_label":
            int(exp.example_label), "example": list(exp.example),
            "target_label": target_label, "delta": exp.delta,
            "binsearch_step": step, "binsearch_upper": upper,
            "binsearch_lower": lower, "example_seed": seed,
            "follow_astar": follow_astar},
            start_eps=0.01)

        data = exp.results[outfile][-1]
        if follow_astar:
            lo, up = util.get_best_astar(data["a*"], task="both")

            if len(data["ara*"]["solutions"]) > 0:
                aralo, araup = max(data["ara*"]["solutions"], key=lambda p: p[1]-p[0])
                print("ara* best bounds", aralo, araup)
                if up-lo < araup-aralo:
                    lo, up = aralo, araup
                    print("using ARA* bounds")
            max_output_diff = up-lo
        else: # follow merge
            min_output_diff = data["merge"]["bounds"][-1][0] # min of at0
            max_output_diff = data["merge"]["bounds"][-1][1] # max of at1

        # if follow_astar, check if we generated examples, if so, update delta
        delta_update = exp.delta
        example_delta = delta_update
        solution_box = None

        if "best_solution_box" in data["a*"] and follow_astar:
            solution_box = dict(data["a*"]["best_solution_box"])
            example_delta = calc_example_delta(exp, solution_box) + 0.1 # !! avoid floating point issues
            print("A* example delta", example_delta)

        if "best_solution_box" in data["ara*"] and follow_astar:
            solution_box = dict(data["ara*"]["best_solution_box"])
            example_delta = calc_example_delta(exp, solution_box) + 0.1 # !! avoid floating point issues
            print("ARA* example delta", example_delta)

        delta_update = min(delta_update, example_delta) # !! avoid floating point rounding issues

        print("STEP:", "max_output_diff", max_output_diff, "delta", exp.delta)
        old_delta = exp.delta

        # either we found an adversarial example, or we could not prove that one does not exist
        if max_output_diff >= 0.0:
            upper = delta_update
            exp.delta = delta_update - 0.5 * (upper - lower)
            print(f"maybe SAT delta update: {old_delta:.3f}/{delta_update:.3f}",
                  f"-> {exp.delta:.3f} [{lower:.3f}, {upper:.3f}]")
        # we showed that no adversarial example exists for this delta
        else:
            if exp.delta == upper:
                lower = exp.delta
                exp.delta = 2.0 * exp.delta
                upper = exp.delta
            else:
                lower = exp.delta
                exp.delta = exp.delta + 0.5 * (upper - lower)
            print(f"UNSAT delta update: {old_delta:.3f}/{delta_update:.3f}"
                  f"-> {exp.delta:.3f} [{lower:.3f}, {upper:.3f}]")

        # there's no difference between delta=1.5 and delta=1.75, pixels are int values
        if np.floor(upper) == np.floor(lower):
            print("we're done early")
            data["done_early"] = True
            break

def allstate(outfile, max_memory):
    exp = AllstateScaleExperiment(max_memory=max_memory, max_time=120, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    for depth in [4, 6, 8]:
        for num_trees in [25, 50, 75, 100, 150, 200, 300, 400, 500]:
            exp.load_model(num_trees, depth)
            exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr": exp.meta["lr"]},
                    start_eps=0.01)
            #break
        #break
    exp.write_results()

def allstate_random(outfile, max_memory, N, seed):
    exp = AllstateRandomScaleExperiment(max_memory=max_memory, max_time=30, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    rng = np.random.RandomState(seed)
    for depth in [3, 4, 5, 6, 7, 8]:
        for num_trees in [25, 50, 75, 100, 150, 200]:
            for i in range(N):
                exp.constraint_seed = rng.randint(2**31)
                exp.load_model(num_trees, depth)
                exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr":
                    exp.meta["lr"], "constraint_seed": exp.constraint_seed},
                    start_eps=0.01)
    exp.write_results()

def soccer(outfile, max_memory, exp_type):
    exp = exp_type(max_memory=max_memory, max_time=10, num_threads=1)
    exp.confirm_write_results(outfile)
    num_trees, depth, lr = 200, 10, 0.25
    exp.load_model(num_trees, depth, lr)
    #exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr": lr}, start_eps=0.01)
    def incr_eps_fun(eps):
        return min(1.0, eps+0.05)
    exp.results[outfile] = [{"num_trees": num_trees, "depth": depth, "lr": lr}]
    info = exp.arastar_single_multiple_solutions(0.05, incr_eps_fun, 10)
    exp.results[outfile][0]["ara*"] = info
    exp.write_results()

def higgs(outfile, max_memory):
    exp = HiggsScaleExperiment(max_memory=max_memory, max_time=120, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    for depth in [4, 6, 8]:
        for num_trees in [25, 50, 75, 100, 150, 200, 300, 400, 500]:
            exp.load_model(num_trees, depth)
            exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr": exp.meta["lr"]},
                    start_eps=0.01)
    exp.write_results()

def higgs_random(outfile, max_memory, N, seed):
    exp = HiggsRandomScaleExperiment(max_memory=max_memory, max_time=30, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    rng = np.random.RandomState(seed)
    for depth in [3, 4, 5, 6, 7, 8]:
        for num_trees in [25, 50, 75, 100, 150, 200]:
            for i in range(N):
                exp.constraint_seed = rng.randint(2**31)
                exp.load_model(num_trees, depth)
                exp.run(outfile, {"num_trees": num_trees, "depth": depth, "lr":
                    exp.meta["lr"], "constraint_seed": exp.constraint_seed},
                    start_eps=0.01)
    exp.write_results()

def youtube(outfile, max_memory):
    exp = YouTubeScaleExperiment(max_memory=max_memory, max_time=30, num_threads=4)
    exp.confirm_write_results(outfile)
    exp.do_merge = False
    exp.at_most_k = 3
    num_trees = 100
    depth = 10
    for fixed_words in [
                #[],
                #["txt_music", "txt_pop", "txt_video", "txt_album"],
                ["txt_epic", "txt_challenge"],
                #["txt_breaking", "txt_news", "txt_live", "txt_war"],
            ]:
        exp.fixed_words = fixed_words
        exp.load_model(num_trees, depth)
        exp.run(outfile, {"num_trees": num_trees, "depth": depth,
            "lr": exp.meta["lr"], "fixed_words": fixed_words},
            start_eps=0.01)
    exp.write_results()

if __name__ == "__main__":
    output_file = sys.argv[2]
    max_memory_gb = int(sys.argv[3])
    max_memory = 1024*1024*1024*max_memory_gb

    exp = sys.argv[1]
    if exp == "calhouse":
        calhouse(output_file, max_memory)
    if exp == "calhouse_random":
        calhouse_random(output_file, max_memory, N=int(sys.argv[4]), seed=int(sys.argv[5]))
    if exp == "covtype":
        covtype(output_file, max_memory)
    if exp == "covtype_random":
        covtype_random(output_file, max_memory, N=int(sys.argv[4]), seed=int(sys.argv[5]))
    if exp == "mnist":
        mnistXvall(output_file, max_memory, sys.argv[4])
    if exp == "mnist_robust":
        N = int(sys.argv[4])
        seed = int(sys.argv[5])
        output_file = f"{output_file}{max_memory_gb}g10s{N}N_{seed}"
        mnist_robust(output_file, max_memory, N=N, seed=seed)
    if exp == "allstate":
        allstate(output_file, max_memory)
    if exp == "allstate_random":
        allstate_random(output_file, max_memory, N=int(sys.argv[4]), seed=int(sys.argv[5]))
    if exp == "soccer_backward":
        soccer(output_file, max_memory, SoccerScaleExperiment_Backwards)
    if exp == "soccer_optimize":
        soccer(output_file, max_memory, SoccerScaleExperiment_Optimize)
    if exp == "higgs":
        higgs(output_file, max_memory)
    if exp == "higgs_random":
        higgs_random(output_file, max_memory, N=int(sys.argv[4]), seed=int(sys.argv[5]))
    if exp == "youtube":
        youtube(output_file, max_memory)
