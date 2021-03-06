import timeit, time
import numpy as np

from veritas import Optimizer, AddTree

DUMMY_AT = AddTree()

class RobustnessSearch:

    NO_STOP_COND = lambda lo, up: False
    INT_STOP_COND = lambda lo, up: np.floor(up) == np.ceil(lo)

    def __init__(self, example, start_delta, num_steps=10, guard=1e-4,
            max_time = 10,
            stop_condition=NO_STOP_COND):
        self.example = example
        self.start_delta = start_delta+guard
        self.num_steps = num_steps
        self.guard = guard # avoid numerical issues when boundaries are ints (e.g. mnist)
        self.max_time = max_time
        self.stop_condition = stop_condition

        self.generated_examples = []
        self.delta_log = []
        self.total_time = None
        self.total_time_p = None

    def search(self):
        self.start_time = timeit.default_timer()
        self.start_time_p = time.process_time()
        upper = self.start_delta
        lower = 0.0
        delta = self.start_delta

        for i in range(self.num_steps):
            self.delta_log.append((delta, lower, upper, timeit.default_timer()-self.start_time))
            res = self.get_max_output_difference(delta)
            if res is None: break
            max_output_diff, generated_examples = res
            best_example_delta = delta
            if len(generated_examples) > 0:
                for example in generated_examples:
                    example_delta = self._calc_example_delta(example) + self.guard
                    best_example_delta = min(best_example_delta, example_delta)
                    self.generated_examples.append(example)

            old_delta = delta

            # (max. target output)-(min. source output)>=0 -> an adv. can exist
            if max_output_diff >= 0.0:
                upper = min(delta, best_example_delta)
                delta = upper - 0.5 * (upper - lower)
                maybe_sat_str = "maybe SAT" if len(generated_examples) == 0 else "SAT"
                print(f"[{i}]: {maybe_sat_str} delta update: {old_delta:.5f}/{best_example_delta:.5f}"
                      f" -> {delta:.5f} [{lower:.5f}, {upper:.5f}]")
            else: # no adv. can exist
                if delta == upper:
                    lower = delta
                    delta = 2.0 * delta
                    upper = delta
                else:
                    lower = delta
                    delta = lower + 0.5 * (upper - lower)
                print(f"[{i}]: UNSAT delta update: {old_delta:.3f}"
                      f" -> {delta:.5f} [{lower:.5f}, {upper:.5f}]")

            if self.stop_condition(lower, upper):
                print(f"done early {lower} <= {delta} <= {upper}")
                break

        self.total_time = timeit.default_timer() - self.start_time
        self.total_time_p = time.process_time() - self.start_time_p
        self.delta_log.append((delta, lower, upper, timeit.default_timer()-self.start_time))

        return delta, lower, upper

    def get_max_output_difference(self, delta): # returns (max_output_diff, [list of generated_example])
        """
        Return tuple
         - [0] max_output_diff: (max. possible output of target_at)
                                  - (min. possible output of source_at)
           If this value is <= 0, then it is not possible for target_at to be
           more confident than source_at of its prediction -> proof that no
           adv. exists for this delta

           else, if this value is > 0, then it may be possible (it is a bound
           after all) that a counter `adversarial` example exists

         - [1] list of generated (counter) examples for this delta, empty list
           if none found. If available, this is used to update the current
           value of `upper`, the upper bound on the minimal distance to the
           closest adversarial example
         """
        raise RuntimeError("not implemented")

    def _calc_example_delta(self, generated_example):
        #pred_target = self.target_at.predict_single(generated_example)
        #pred_source = self.source_at.predict_single(generated_example)
        #print(f"Adv.example target {pred_target:.6f}")#, ({up:.6f}, {pred_target-up:.3g})")
        #print(f"Adv.example source {pred_source:.6f}")#, ({lo:.6f}, {pred_source-lo:.3g})")
        example_delta = max(abs(x-y) for x, y in zip(generated_example, self.example))
        print(f"Adv.example delta", example_delta)
        return example_delta




class OptimizerRobustnessSearch(RobustnessSearch):
    def __init__(self, source_at, target_at, example,
            optimizer_kwargs={}, **kwargs):
        super().__init__(example, **kwargs)

        self.source_at = source_at if source_at is not None else DUMMY_AT
        self.target_at = target_at if target_at is not None else DUMMY_AT
        self.optimizer_kwargs = optimizer_kwargs

        self.opt = Optimizer(minimize=self.source_at, maximize=self.target_at,
                matches=set(), match_is_reuse=False)
        self.prev_delta = None

        self.log = []

    def get_opt(self, delta):
        # Share all variables between source and target model
        if self.prev_delta is None or self.prev_delta > delta:
            pass  # reuse self.opt, and prune further
        else:
            self.opt.reset_graphs_and_optimizer()
        self.opt.prune_example(list(self.example), delta)
        self.prev_delta = delta

    def _log_opt(self, delta):
        if self.opt is not None: 
            stats = self.opt.stats()
            self.log.append(stats)




class VeritasRobustnessSearch(OptimizerRobustnessSearch):

    def __init__(self, source_at, target_at, example,
            optimizer_kwargs={}, eps_start=1.0, eps_incr=0.05, **kwargs):
        super().__init__(source_at, target_at, example, optimizer_kwargs,
                **kwargs)

        self.eps_start = eps_start
        self.eps_incr = eps_incr

        self.steps_kwargs = { "min_output_difference": 0.0 }
        #self.steps_kwargs = { "max_output": 0.0, "min_output": 0.0 }

    def get_max_output_difference(self, delta):
        self.get_opt(delta)

        rem_time = self.max_time - timeit.default_timer() + self.start_time
        #rem_time = min(rem_time, 2.0 * self.max_time / self.num_steps)
        rem_time /= (self.num_steps - len(self.log))
        print("step time", rem_time, self.max_time, self.num_steps)

        if rem_time < 0.0:
            return None

        if self.eps_start == 1.0:
            self.opt.astar(rem_time, steps_kwargs=self.steps_kwargs, max_num_steps=100)
        else:
            self.opt.arastar(rem_time, self.eps_start, self.eps_incr,
                    steps_kwargs=self.steps_kwargs, max_num_steps=100)

        if self.opt.num_solutions() > 0:
            sol = max(self.opt.solutions(), key=lambda s: s.output_difference())
            print(f"Veritas generated example {sol.output0} {sol.output1}")
            max_output_diff = sol.output_difference()
            closest = self.opt.get_closest_example(sol, self.example, instance=0)
            closest = self.opt.get_closest_example(sol, closest, instance=1)
            generated_examples = [closest]
        else:
            lo, up = self.opt.bounds[-1]
            max_output_diff = (up - lo) / self.opt.get_eps()
            generated_examples = []

        super()._log_opt(delta)

        return max_output_diff, generated_examples




class MergeRobustnessSearch(OptimizerRobustnessSearch):
    def __init__(self, source_at, target_at, example, max_merge_depth=9999,
            optimizer_kwargs={}, **kwargs):
        super().__init__(source_at, target_at, example, optimizer_kwargs,
                **kwargs)
        self.merge_kwargs = { "max_merge_depth": max_merge_depth }

    def get_max_output_difference(self, delta):
        self.get_opt(delta)

        rem_time = self.max_time - timeit.default_timer() + self.start_time
        #rem_time = min(rem_time / 2, 2.0 * self.max_time / self.num_steps)
        rem_time /= (self.num_steps - len(self.log))
        print("step time", rem_time, self.max_time, self.num_steps)

        if rem_time < 0.0:
            return None

        result = self.opt.merge(rem_time, **self.merge_kwargs)

        try: max_output_diff = result["bounds"][-1][1]
        except: max_output_diff = np.inf

        super()._log_opt(delta)

        return max_output_diff, [] # merge cannot generate examples
