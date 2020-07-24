import os, sys, json, io
import time, timeit
import pickle
import multiprocessing as mp

import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.special import logit, expit as logistic

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
        self.min_time_per_step = 100
        self.max_time_per_step = 2000
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
            #try:
            done = not opt.steps(num_steps)
            num_steps = min(self.max_num_steps, num_steps * 2)
            #except:
            #    print("A* OUT OF MEMORY")
            #    done = True
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
        b = opt.current_basic_bounds()
        m = opt.get_mem_size()
        v = opt.g0.num_vertices() + opt.g1.num_vertices()
        conn.send(("point", t, b, m, v))
        start = timeit.default_timer()
        try:
            # dynamic programming gives optimal solution at 2
            while True:
                try:
                    print("MERGE worker: num_independent_sets:", opt.g1.num_independent_sets())
                    opt.merge(2, reset_optimizer=False) # dyn prog. algorithm is quadratic, basic bound is linear
                except:
                    print("MERGE worker: OUT OF MEMORY")
                    conn.send(("oom",))
                    break

                t = timeit.default_timer() - start
                b = opt.current_basic_bounds()
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

        dur = timeit.default_timer() - start

        if data["oot"]:
            print("MERGE host: timeout")

        print("MERGE host: terminating")
        p.terminate()
        cparent.close()

        data["times"] = times
        data["bounds"] = bounds
        data["memory"] = memory
        data["vertices"] = vertices
        data["total_time"] = dur

        return data

    def _extract_info(self, opt, dur):
        sols = util.filter_solutions(opt)
        data = {
            "solutions": [(s.output0, s.output1) for s in sols],
            "bounds": opt.bounds,
            "bounds_times": opt.times,
            "memory": opt.memory,
            "sol_times": [s.time for s in sols],
            "epses": [s.eps for s in sols],
            "total_time": dur,
        }
        if len(sols) > 0:
            best_sol = max(sols, key=lambda s: s.output_difference())
            data["best_solution_box"] = {i: (d.lo, d.hi) for i, d in best_sol.box().items()},
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

class CovtypeScaleExperiment(ScaleExperiment):
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

class Mnist2vallScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/mnist"

    def load_model(self, num_trees, depth, lr, delta):
        print("\n=== NEW MODEL ===")
        model_name = f"mnist2vall-{num_trees}-{depth}-{lr}.xgb"
        X, y = util.load_openml("mnist", data_id=554)
        old_y = y
        y = (y==2)
        num_examples, num_features = X.shape
        Itrain, Itest = util.train_test_indices(num_examples)
        if not os.path.isfile(os.path.join(self.result_dir, model_name)):
            print(f"training model learning_rate={lr}, depth={depth}, num_trees={num_trees}")
            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "max_depth": depth,
                "learning_rate": lr,
                "eval_metric": "error",
                "seed": 53589,
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
        self.example = X[Itest[15]]
        self.label = old_y[Itest[15]]
        self.delta = delta

        print(list(self.example))
        print(self.label)

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory)
        print("before num_vertices", opt.g1.num_vertices())
        opt.prune_example(list(self.example), self.delta)
        print("after num_vertices", opt.g1.num_vertices())
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

    def load_data(self):
        soccer_data_path = os.path.join(os.environ["TREECK_DATA_DIR"], "soccerdata.h5")
        X = pd.read_hdf(soccer_data_path, "features")
        X["goalscore_team"] = X["goalscore_team"].astype(np.float32)
        X["goalscore_opponent"] = X["goalscore_opponent"].astype(np.float32)
        X["goalscore_diff"] = X["goalscore_diff"].astype(np.float32)
        y = pd.read_hdf(soccer_data_path, "labels").astype(np.float32)["scores"]

        X = X.drop(columns=["dist_to_goal_a0", "dist_to_goal_a1",
            "angle_to_goal_a0", "angle_to_goal_a1"])
        X = X.drop(columns=["time_seconds_a0", "time_seconds_a1"])
        X = X.drop(columns=["dx_a0", "dy_a0", "dx_a1", "dy_a1"])
        X = X.drop(columns=["period_id_a0", "period_id_a1", "time_delta_1"])
        X = X.drop(columns=["goalscore_opponent"])
        X = X.drop(columns=["time_seconds_overall_a1"])

        return X, y

    def load_model(self, num_trees, depth, lr):
        print("\n=== NEW MODEL ===")
        model_name = f"soccer-{num_trees}-{depth}-{lr}.xgb"
        if not os.path.isfile(os.path.join(self.result_dir, model_name)):
            print(f"training model learning_rate={lr}, depth={depth}, num_trees={num_trees}")
            X, y = self.load_data()

            num_examples, num_features = X.shape
            Itrain, Itest = util.train_test_indices(num_examples, seed=1)
            Xtrain = X.iloc[Itrain]
            ytrain = y.to_numpy()[Itrain]
            Xtest = X.iloc[Itest]
            ytest = y.to_numpy()[Itest]
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
                "seed": 1
            }
            self.model = xgb.train(params, dtrain, num_trees, [(dtrain, "train"), (dtest, "test")])
                    #early_stopping_rounds=10)
            ytest_hat = self.model.predict(dtest)

            self.feat2id_dict = dict([(v, i) for i, v in enumerate(X.columns)])
            self.feat2id = lambda x: self.feat2id_dict[x]
            self.id2feat = lambda i: X.columns[i]

            self.at = addtree_from_xgb_model(self.model, self.feat2id)
            self.at.base_score = 0.0

            pred = self.model.predict(dtest, output_margin=True)
            pred_at = np.array(self.at.predict(Xtest[:1000]))

            brier_score = sum((logistic(pred) - ytest)**2) / len(pred)
            print("brier score", brier_score)

            mae = mean_absolute_error(pred[:1000], pred_at)
            print(f"mae model difference {mae} before base_score")

            self.at.base_score = -mae
            pred_at = np.array(self.at.predict(Xtest[:1000].to_numpy()))
            mae = mean_absolute_error(pred[:1000], pred_at)
            print(f"mae model difference {mae}")

            with open(os.path.join(self.result_dir, model_name), "wb") as f:
                pickle.dump(self.model, f)
            self.at.write(os.path.join(self.result_dir, f"{model_name}.at"))
            self.meta = {"feat2id": self.feat2id_dict}
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "wb") as f:
                pickle.dump(self.meta, f)

        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, model_name), "rb") as f:
                self.model = pickle.load(f)
            self.at = AddTree.read(os.path.join(self.result_dir, f"{model_name}.at"))
            with open(os.path.join(self.result_dir, f"{model_name}.meta"), "rb") as f:
                self.meta = pickle.load(f)

        #print(num_examples, num_features)

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory, use_dyn_prog_heuristic=False)

        u = lambda n: self.meta["feat2id"][n]
        used_ids1 = set(opt.feat_info.feat_ids1())
        action0_ids = [opt.feat_info.get_id(1, u(name)) for name in SoccerScaleExperiment.action0]
        action0_ids = list(list(used_ids1.intersection(action0_ids)))
        action1_ids = [opt.feat_info.get_id(1, u(name)) for name in SoccerScaleExperiment.action1]
        action1_ids = list(list(used_ids1.intersection(action1_ids)))
        bodypart0_ids = [opt.feat_info.get_id(1, u(name)) for name in SoccerScaleExperiment.bodypart0]
        bodypart0_ids = list(list(used_ids1.intersection(bodypart0_ids)))
        bodypart1_ids = [opt.feat_info.get_id(1, u(name)) for name in SoccerScaleExperiment.bodypart1]
        bodypart1_ids = list(list(used_ids1.intersection(bodypart1_ids)))

        opt.adjuster.add_one_out_of_k(action0_ids, len(action0_ids) == len(SoccerScaleExperiment.action0))
        opt.adjuster.add_one_out_of_k(action1_ids, len(action1_ids) == len(SoccerScaleExperiment.action1))
        opt.adjuster.add_one_out_of_k(bodypart0_ids, len(bodypart0_ids) == len(SoccerScaleExperiment.bodypart0))
        opt.adjuster.add_one_out_of_k(bodypart1_ids, len(bodypart1_ids) == len(SoccerScaleExperiment.bodypart1))

        print("num_vertices", opt.g1.num_vertices())
        return opt

class HiggsScaleExperiment(ScaleExperiment):
    result_dir = "tests/experiments/scale/higgs"

    def load_data(self):
        higs_data_path = os.path.join(os.environ["TREECK_DATA_DIR"], "higgs.h5")
        X = pd.read_hdf(higs_data_path, "X")
        y = pd.read_hdf(higs_data_path, "y")
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
                "seed": 1
            }

            def metric(y, raw_yhat):
                return accuracy_score(y, raw_yhat > 0)

            self.model, lr, metric_value = util.optimize_learning_rate(X, y,
                    params, num_trees, metric)

            self.meta = {}
            self.meta["feat2id"] = dict([(v, i) for i, v in enumerate(X.columns)])
            self.meta["lr"] = lr
            self.at = addtree_from_xgb_model(self.model, lambda x: self.meta["feat2id"][x])

            with open(os.path.join(self.result_dir, model_name), "wb") as f:
                pickle.dump(self.model, f)
            with open(os.path.join(self.result_dir, meta_name), "w") as f:
                json.dump(self.meta, f)
            self.at.write(os.path.join(self.result_dir, at_name))

        else:
            print(f"loading model from file: {model_name}")
            with open(os.path.join(self.result_dir, model_name), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.result_dir, meta_name), "r") as f:
                self.meta = json.load(f)
            self.at = AddTree.read(os.path.join(self.result_dir, at_name))

    def get_opt(self):
        opt = Optimizer(maximize=self.at, max_memory=self.max_memory, use_dyn_prog_heuristic=False)
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
    exp = CovtypeScaleExperiment(max_memory=max_memory, max_time=60, num_threads=4)
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

def mnist2vall(outfile, max_memory):
    exp = Mnist2vallScaleExperiment(max_memory=max_memory, max_time=10, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    for num_trees, depth, lr in [
            #(20, 5, 1.0),
            #(50, 5, 0.5),
            #(100, 5, 0.25),
            #(100, 6, 0.25),
            #(150, 5, 0.20),
            (150, 6, 0.20),
            #(200, 6, 0.10)
            ]:
        exp.load_model(num_trees, depth, lr, 10)
        exp.run(output_file, {"num_trees": num_trees, "depth": depth, "lr": lr})
    exp.write_results()

def soccer(outfile, max_memory):
    exp = SoccerScaleExperiment(max_memory=max_memory, max_time=10, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = True
    for num_trees, depth, lr in [
            #(10, 5, 1.0),
            #(20, 5, 0.8),
            #(30, 5, 0.5),
            (50, 5, 0.35)
            ]:
        exp.load_model(num_trees, depth, lr)
        exp.run(output_file, {"num_trees": num_trees, "depth": depth, "lr": lr}, start_eps=0.01)
    exp.write_results()

def higgs(outfile, max_memory):
    exp = HiggsScaleExperiment(max_memory=max_memory, max_time=10, num_threads=1)
    exp.confirm_write_results(outfile)
    exp.do_merge = False
    num_trees = 100
    depth = 5
    exp.load_model(num_trees, depth)
    exp.run(output_file, {"num_trees": num_trees, "depth": depth, "lr": exp.meta["lr"]})
    exp.write_results()

if __name__ == "__main__":
    output_file = sys.argv[1]
    max_memory = 1024*1024*1024*int(sys.argv[2])

    #calhouse(output_file, max_memory)
    #covtype(output_file, max_memory)
    #mnist2vall(output_file, max_memory)
    #soccer(output_file, max_memory)
    higgs(output_file, max_memory)



