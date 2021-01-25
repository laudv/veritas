import time, timeit
from veritas import *

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

    def filter_solutions(self, num=None):
        sols = self.solutions()
        sols.sort(key=lambda s: s.output_difference(), reverse=True)
        sols.sort(key=lambda s: s.eps) # stable sort

        fsols = [] # filtered solutions
        prev_eps = -1
        for s in sols:
            if s.eps != prev_eps:
                fsols.append(s)
            prev_eps = s.eps

        if num is not None:
            fsols = fsols[-num:]

        return fsols

    def snapshot(self, num_solutions=None):
        sols = self.filter_solutions(num_solutions)
        snap = {
            "bounds": self.bounds.copy(),
            "times": self.times.copy(),
            "memory": self.memory.copy(),
            "clique_count": self.clique_count.copy(),
            "solutions": [(s.output0, s.output1) for s in sols],
            "sol_times": [s.time for s in sols],
            "sol_epses": [s.eps for s in sols],

            "start_eps": self.ara_eps,
            "eps": self.opt.get_eps(),
            "max_memory": self.max_memory,

            "total_time": timeit.default_timer() - self.start_time,

            "num_vertices0": self.g0.num_vertices(),
            "num_vertices1": self.g1.num_vertices(),
        }
        if len(sols) > 0:
            best_sol = max(sols, key=lambda s: s.output_difference())
            snap["best_solution_box"] = {i: (d.lo, d.hi) for i, d in best_sol.box().items()}
        return snap

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
                raise e
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

    # Chen, H., Zhang, H., Si, S., Li, Y., Boning, D., & Hsieh, C. J. (2019).
    # Robustness verification of tree-based models. In Advances in Neural
    # Information Processing Systems (pp. 12317-12328).
    def merge(self, max_time=10, max_merge_depth=9999):
        def merge_worker_fun(self, conn, max_merge_depth):
            g = self.g1
            g.add_with_negated_leaf_values(self.g0)

            t = 0.0
            b = g.basic_bound()
            m = g.get_used_mem_size()
            v = g.num_vertices()
            conn.send(("point", t, b, m, v))
            start = timeit.default_timer()
            try:
                for merge_step in range(max_merge_depth):
                    try:
                        print("MERGE worker: num_independent_sets:",
                                g.num_independent_sets(), g.num_vertices())
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
                print("MERGE worker: closing")
                conn.close()

        times, bounds, memory, vertices = [], [], [], []
        start = timeit.default_timer()
        start_p = time.process_time()
        data = {"oot": True, "oom": False, "optimal": False}

        if max_merge_depth > 3:
            cparent, cchild = mp.Pipe()
            p = mp.Process(target=merge_worker_fun, name="Merger", args=(self, cchild, max_merge_depth))
            p.start()
            while timeit.default_timer() - start < max_time:
                has_data = cparent.poll(0.1)
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
                        m = msg[1]
                        print(f"MERGE host: oom ({m/(1024*1024):.2f} MiB)")
                        data["oom"] = True
                        data["oom_value"] = m
                elif p.exitcode is not None:
                    data["oot"] = False
                    break

            if data["oot"]:
                print("MERGE host: timeout")

            print("MERGE host: terminating")
            p.terminate()
            cparent.close()
        else:
            g = self.g1
            g.add_with_negated_leaf_values(self.g0)
            for merge_step in range(max_merge_depth):
                try:
                    print("MERGE worker: num_independent_sets:",
                            g.num_independent_sets(), g.num_vertices())
                    g.merge(2)
                except Exception as e:
                    m = g.get_used_mem_size()
                    print(f"MERGE worker: OUT OF MEMORY: {m/(1024*1024):.2f} MiB", type(e))
                    data["oom"] = True
                    data["oom_value"] = m
                    break

                times.append(timeit.default_timer() - start)
                bounds.append(g.basic_bound())
                memory.append(g.get_used_mem_size())
                vertices.append(g.num_vertices())

                if g.num_independent_sets() <= 1:
                    data["optimal"] = True
                    break

        dur = timeit.default_timer() - start
        dur_p = time.process_time() - start_p

        data["times"] = times
        data["bounds"] = bounds
        data["memory"] = memory
        data["vertices"] = vertices
        data["total_time"] = dur
        data["total_time_p"] = dur_p

        return data

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

    def get_closest_example(self, solution, example, instance=0, guard=1e-4):
        num_attributes = len(example)
        intervals = self.solution_to_intervals(solution, num_attributes)[instance]

        closest = example.copy()

        for feat_id, interval in enumerate(intervals):
            x = example[feat_id]
            if isinstance(x, bool):
                raise RuntimeError("not supported yet")
            if interval is None:
                continue
            lo, hi = interval
            if lo <= x and x < hi:
                continue # keep the value x

            dist_lo = abs(lo - x)
            dist_hi = abs(x - hi)
            if dist_lo > dist_hi:
                closest[feat_id] = hi - guard
            else:
                closest[feat_id] = lo

            #print(f"interval {feat_id}:", interval, x, "->", closest[feat_id])

        return closest
