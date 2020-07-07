import os, sys, json
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
        opt.step(self.min_num_steps)

        try:
            paropt = opt.parallel(self.num_threads)
            while paropt.num_solutions() == 0 \
                    and not paropt.num_candidate_cliques() == 0 \
                    and timeit.default_timer() < stop:
                try:
                    paropt.step_for(time_per_step)
                    time_per_step = min(self.max_time_per_step, time_per_step * 1.5)
                except:
                    print("A* OUT OF MEMORY")
                    break
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
        opt.step(self.min_num_steps)
        try:
            paropt = opt.parallel(self.num_threads)
            while not paropt.num_candidate_cliques() == 0 \
                    and timeit.default_timer() < stop \
                    and not (paropt.get_ara_eps() == 1.0 \
                    and paropt.num_new_valid_solutions() > 0):
                if paropt.num_new_valid_solutions() > 0:
                    eps = paropt.get_eps()
                    paropt_ara.set_eps(incr_eps_fun(eps))
                    print(f"ARA* eps: {eps} -> {opt.get_eps()}")
                    time_per_step = self.min_time_per_step
                try:
                    paropt.steps_for(time_per_step)
                    time_per_step = min(self.max_time_per_step, time_per_step * 1.5)
                except:
                    print("ARA* OUT OF MEMORY")
                    break
        finally:
            paropt.join_all()

        dur = timeit.default_timer() - start
        return paropt, opt, dur

    def merge_worker_fun(self, conn):
        opt = self.get_opt()
        t = 0.0
        b = opt.current_bounds()[1]
        m = opt.get_mem_size()
        v = opt.g0.num_vertices() + opt.g1.num_vertices()
        conn.send((t, b, m, v))
        start = timeit.default_timer()
        try:
            while True:
                try:
                    opt.merge(2)
                except:
                    print("MERGE worker: OUT OF MEMORY")
                    break

                if b == opt.current_bounds()[1]:
                    break

                t = timeit.default_timer() - start
                b = opt.current_bounds()[1]
                m = opt.get_mem_size()
                v = opt.g0.num_vertices() + opt.g1.num_vertices()
                conn.send((t, b, m, v))
        finally:
            print("MERGE worker: closing")
            conn.close()

    def merge(self):
        cparent, cchild = mp.Pipe()
        p = mp.Process(target=ScaleExperiment.merge_worker_fun, name="Merger", args=(self, cchild,))
        p.start()
        start = timeit.default_timer()
        timings, bounds, memory, vertices = [], [], [], []
        #print("MERGE host: runtime", self.max_time)
        while timeit.default_timer() - start < self.max_time:
            has_data = cparent.poll(1)
            if has_data:
                t, b, m, v = cparent.recv()
                print("MERGE host: data", t, b, m, v)
                timings.append(t)
                bounds.append(b)
                memory.append(m)
                vertices.append(v)
            elif p.exitcode is not None:
                break
        print("MERGE host: terminating")
        p.terminate()
        cparent.close()

        return {
            "timings": timings,
            "bounds": bounds,
            "memory": memory,
            "vertices": vertices
        }

    def _extract_info(self, opt, dur):
        sols = opt.solutions()
        data = {
            "solutions": [(s.output0, s.output1) for s in sols],
            "bounds": opt.bounds,
            "bounds_timings": opt.times,
            "memory": opt.memory,
            "timings": [s.time for s in sols],
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

        print(f"\n -- A* {time.ctime()} --")
        data["a*"] = self.astar()
        print(f"\n -- ARA* {time.ctime()} --")
        data["ara*"] = self.arastar(start_eps, eps_incr)
        print(f"\n -- MERGE {time.ctime()} --")
        data["merge"] = self.merge()

    def write_results(self):
        for key, data in self.results.items():
            filename = os.path.join(self.result_dir, key)
            print("writing output to", filename)
            if input("OK? ") != "y":
                break
            with open(filename, "w") as f:
                json.dump(data, f)

class CalhouseScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/calhouse"

    def load_model(self, num_trees, depth, lr):
        print("=== NEW MODEL ===")
        model_name = f"calhouse-{num_trees}-{depth}-{lr}.xgb"
        if not os.path.isfile(os.path.join(self.result_dir, model_name)):
            X, y = util.load_openml("calhouse", data_id=537)
            num_examples, num_features = X.shape
            Itrain, Itest = util.train_test_indices(num_examples)
            print(f"training model learning_rate={lr}, num_trees={num_trees}")
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


if __name__ == "__main__":
    output_file = sys.argv[1]
    max_memory = 1024*1024*1024*int(sys.argv[2])

    exp = CalhouseScaleExperiment(max_memory=max_memory, max_time=10)
    exp.load_model(10, 5, 1.0)
    exp.run(output_file, {"num_trees": 10, "depth": 5, "lr": 1.0})
    exp.load_model(20, 5, 0.8)
    exp.run(output_file, {"num_trees": 20, "depth": 5, "lr": 0.8})

    exp.write_results()
