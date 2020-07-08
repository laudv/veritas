import os, sys, json, io
import time, timeit
import pickle
import multiprocessing as mp

import xgboost as xgb

from treeck import *
from treeck.xgb import addtree_from_xgb_model

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
        self.min_time_per_step = 10
        self.max_time_per_step = 1000
        self.do_merge = True

        self.results = {}

    def load_model(self):
        raise RuntimeError("override")

    def get_opt(self):
        raise RuntimeError("override")

    def astar_single(self):
        opt = self.get_opt()
        done = False
        num_steps = self.min_num_steps
        start = timeit.default_timer()
        stop = start + self.max_time
        while not done and opt.num_solutions() == 0 \
                and timeit.default_timer() < stop:
            try:
                done = not opt.steps(num_steps)
                num_steps = min(self.max_num_steps, num_steps * 2)
            except:
                print("A* OUT OF MEMORY")
                done = True
        dur = timeit.default_timer() - start
        return opt, dur

    def astar_parallel(self):
        opt = self.get_opt()

        time_per_step = self.min_time_per_step
        start = timeit.default_timer()
        stop = start + self.max_time

        # just a few steps so we have some work to share between workers
        opt.steps(self.min_num_steps)

        try:
            paropt = opt.parallel(self.num_threads)
            while paropt.num_solutions() == 0 \
                    and not paropt.num_candidate_cliques() == 0 \
                    and timeit.default_timer() < stop:
                #try:
                paropt.steps_for(time_per_step)
                time_per_step = min(self.max_time_per_step, int(time_per_step * 1.5))
                #except:
                #    print("A* OUT OF MEMORY")
                #    break
        finally:
            paropt.join_all()

        dur = timeit.default_timer() - start
        return paropt, opt, dur

    def arastar_single(self, start_eps, incr_eps_fun):
        opt = self.get_opt()
        opt.set_eps(start_eps)
        done = False
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
            #try:
            done = not opt.steps(num_steps)
            num_steps = min(self.max_num_steps, num_steps * 2)
            #except:
            #    print("ARA* OUT OF MEMORY")
            #    done = True
        dur = timeit.default_timer() - start
        return opt, dur
    
    def arastar_parallel(self, start_eps, incr_eps_fun):
        opt = self.get_opt()
        opt.set_eps(start_eps)

        time_per_step = self.min_time_per_step
        start = timeit.default_timer()
        stop = start + self.max_time

        # just a few steps so we have some work to share between workers
        opt.steps(self.min_num_steps)
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
                #try:
                paropt.steps_for(time_per_step)
                time_per_step = min(self.max_time_per_step, int(time_per_step * 1.5))
                #except:
                #    print("ARA* OUT OF MEMORY")
                #    break
        finally:
            paropt.join_all()

        dur = timeit.default_timer() - start
        return paropt, opt, dur

    def merge_worker_fun(self, conn):
        opt = self.get_opt()
        t = 0.0
        b = opt.current_bounds()
        m = opt.get_mem_size()
        v = opt.g0.num_vertices() + opt.g1.num_vertices()
        conn.send(("point", t, b, m, v))
        start = timeit.default_timer()
        try:
            # dynamic programming gives optimal solution at 2
            while True:
                try:
                    print("MERGE worker: num_independent_sets:", opt.g1.num_independent_sets())
                    opt.merge(2)
                except:
                    print("MERGE worker: OUT OF MEMORY")
                    conn.send(("oom",))
                    break

                t = timeit.default_timer() - start
                b = opt.current_bounds()
                m = opt.get_mem_size()
                v = opt.g0.num_vertices() + opt.g1.num_vertices()
                conn.send(("point", t, b, m, v))

                if opt.g0.num_independent_sets() <= 2 and opt.g1.num_independent_sets() <= 2:
                    conn.send(("optimal",))
                    break
        finally:
            print("MERGE worker: closing")
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
                    print("MERGE host: data", t, b, m, v)
                    times.append(t)
                    bounds.append(b)
                    memory.append(m)
                    vertices.append(v)
                elif msg[0] == "optimal":
                    print("MERGE host: optimal found")
                    data["optimal"] = True
                elif msg[0] == "oom":
                    print("MERGE host: oom")
                    data["oom"] = True
            elif p.exitcode is not None:
                data["oot"] = False
                break

        if data["oot"]:
            print("MERGE host: timeout")

        print("MERGE host: terminating")
        p.terminate()
        cparent.close()

        data["times"] = times
        data["bounds"] = bounds
        data["memory"] = memory
        data["vertices"] = vertices

        return data

    def _extract_info(self, opt, dur):
        sols = opt.solutions()
        data = {
            "solutions": [(s.output0, s.output1) for s in sols],
            "bounds": opt.bounds,
            "bounds_times": opt.times,
            "memory": opt.memory,
            "sol_times": [s.time for s in sols],
            "epses": [s.eps for s in sols],
            "total_time": dur,
        }
        if isinstance(opt, Optimizer):
            data["num_vertices0"] = opt.g0.num_vertices()
            data["num_vertices1"] = opt.g1.num_vertices()
        if isinstance(opt, ParallelOptimizer):
            data["num_vertices0"] = opt.opt.g0.num_vertices()
            data["num_vertices1"] = opt.opt.g1.num_vertices()
        return data

    def astar(self):
        if self.num_threads == 1:
            opt, dur = self.astar_single()
            return self._extract_info(opt, dur)
        else:
            paropt, opt, dur = self.astar_parallel()
            return self._extract_info(paropt, dur)

    def arastar(self, start_eps, eps_incr):
        if self.num_threads == 1:
            opt, dur = self.arastar_single(start_eps, eps_incr)
            return self._extract_info(opt, dur)
        else:
            paropt, opt, dur = self.arastar_parallel(start_eps, eps_incr)
            return self._extract_info(paropt, dur)

    def run(self, name, data={}, start_eps=0.5):
        def eps_incr(eps):
            return eps + (1.0-eps)/10.0 if eps < 0.99 else 1.0

        if name not in self.results:
            self.results[name] = []
        self.results[name].append(data)

        data["num_threads"] = self.num_threads
        data["max_memory"] = self.max_memory
        data["max_time"] = self.max_time

        print(f"\n -- A* {time.ctime()} --")
        data["a*"] = self.astar()
        print(f"\n -- ARA* {time.ctime()} --")
        data["ara*"] = self.arastar(start_eps, eps_incr)
        if self.do_merge:
            print(f"\n -- MERGE {time.ctime()} --")
            data["merge"] = self.merge()

    def confirm_write_results(self, name):
        filename = os.path.join(self.result_dir, name)
        print("writing output to", filename)
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

    def load_model(self, num_trees, depth, lr):
        print("\n=== NEW MODEL ===")
        model_name = f"calhouse-{num_trees}-{depth}-{lr}.xgb"
        if not os.path.isfile(os.path.join(self.result_dir, model_name)):
            X, y = util.load_openml("calhouse", data_id=537)
            num_examples, num_features = X.shape
            Itrain, Itest = util.train_test_indices(num_examples)
            print(f"training model learning_rate={lr}, depth={depth}, num_trees={num_trees}")
            params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "max_depth": depth,
                "learning_rate": lr,
                "seed": 14,
            }

            dtrain = xgb.DMatrix(X[Itrain], y[Itrain], missing=None)
            dtest = xgb.DMatrix(X[Itest], y[Itest], missing=None)
            model = xgb.train(params, dtrain, num_boost_round=num_trees,
                              #early_stopping_rounds=5,
                              evals=[(dtrain, "train"), (dtest, "test")])
            with open(os.path.join(self.result_dir, model_name), "wb") as f:
                pickle.dump(model, f)
            #with open(os.path.join(RESULT_DIR, "model.json"), "w") as f:
            #    model.dump_model(f, dump_format="json")
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, model_name), "rb") as f:
                model = pickle.load(f)

        self.model = model
        self.at = addtree_from_xgb_model(model)
        self.at.base_score = 0

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        return opt

class CovtypScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/covtype"

    def load_model(self, num_trees, depth, lr):
        print("\n=== NEW MODEL ===")
        model_name = f"covtype-{num_trees}-{depth}-{lr}.xgb"
        if not os.path.isfile(os.path.join(self.result_dir, model_name)):
            X, y = util.load_openml("covtype", data_id=1596)
            y = (y==2)
            num_examples, num_features = X.shape
            Itrain, Itest = util.train_test_indices(num_examples)
            print(f"training model learning_rate={lr}, depth={depth}, num_trees={num_trees}")
            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "max_depth": depth,
                "learning_rate": lr,
                "eval_metric": "error",
                "seed": 235,
            }

            dtrain = xgb.DMatrix(X[Itrain], y[Itrain], missing=None)
            dtest = xgb.DMatrix(X[Itest], y[Itest], missing=None)
            model = xgb.train(params, dtrain, num_boost_round=num_trees,
                              #early_stopping_rounds=5,
                              evals=[(dtrain, "train"), (dtest, "test")])
            with open(os.path.join(self.result_dir, model_name), "wb") as f:
                pickle.dump(model, f)
            #with open(os.path.join(RESULT_DIR, "model.json"), "w") as f:
            #    model.dump_model(f, dump_format="json")
        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, model_name), "rb") as f:
                model = pickle.load(f)

        self.model = model
        self.at = addtree_from_xgb_model(model)
        self.at.base_score = 0

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

def calhouse(outfile, max_memory):
    exp = CalhouseScaleExperiment(max_memory=max_memory, max_time=60)
    exp.confirm_write_results(outfile)
    for num_trees, depth, lr in [
            (20, 5, 1.0),
            (50, 5, 0.5),
            (100, 5, 0.25),
            (100, 6, 0.25),
            ]:
        exp.load_model(num_trees, depth, lr)
        exp.run(output_file, {"num_trees": num_trees, "depth": depth, "lr": lr})
    exp.write_results()

def covtype(outfile, max_memory):
    exp = CovtypScaleExperiment(max_memory=max_memory, max_time=60, num_threads=4)
    exp.confirm_write_results(outfile)
    exp.do_merge = False
    for num_trees, depth, lr in [
            (20, 5, 1.0),
            (50, 5, 0.5),
            (100, 5, 0.25),
            (100, 6, 0.25),
            ]:
        exp.load_model(num_trees, depth, lr)
        exp.run(output_file, {"num_trees": num_trees, "depth": depth, "lr": lr})
    exp.write_results()

if __name__ == "__main__":
    output_file = sys.argv[1]
    max_memory = 1024*1024*1024*int(sys.argv[2])

    #calhouse(output_file, max_memory)
    covtype(output_file, max_memory)

