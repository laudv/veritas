## \file robustness.py
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import timeit, time, os, contextlib
import numpy as np

from . import AddTree, Search, get_closest_example, Domain

try:
    from .kantchelian import KantchelianOutputOpt
except:
    pass

DUMMY_AT = AddTree()

## \ingroup python
# \brief Base class binary robustness search
class RobustnessSearch:

    NO_STOP_COND = lambda lo, up: False
    INT_STOP_COND = lambda lo, up: np.floor(up) == np.ceil(lo)

    def __init__(self, example, start_delta, num_steps=10, guard=0.0,
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

    ## Do the search
    #
    # For each binary search step, a tuple `(delta, delta_lower_bound,
    # delta_upper_bound)` is added to self.delta_log.
    #
    # \return Tuple `(delta, delta_lower_bound, delta_upper_bound)`
    def search(self):
        self.start_time = timeit.default_timer()
        self.start_time_p = time.process_time()
        upper = self.start_delta
        lower = 0.0
        delta = self.start_delta

        for self.i in range(self.num_steps):
            self.delta_log.append((delta, lower, upper, timeit.default_timer()-self.start_time))
            step_time = self.get_step_time()
            if step_time < 0.0: break
            res = self.get_max_output_difference(delta, step_time)
            if res is None: break
            max_output_diff, generated_examples = res
            best_example_delta = delta
            if len(generated_examples) > 0:
                for example in generated_examples:
                    example_delta = self._calc_example_delta(example) + self.guard
                    #print("example_delta", example_delta, self.guard, best_example_delta)
                    #print("max_output_diff", max_output_diff,
                    #        self.at.eval(example)[0], self.at.eval(self.example)[0])
                    best_example_delta = min(best_example_delta, example_delta)
                    self.generated_examples.append(example)

            old_delta = delta

            # (max. target output)-(min. source output)>=0 -> an adv. can exist
            if max_output_diff >= 0.0:
                upper = min(delta, best_example_delta)
                delta = upper - 0.5 * (upper - lower)
                maybe_sat_str = "? SAT" if len(generated_examples) == 0 else "  SAT"
                t = timeit.default_timer() - self.start_time
                print(f"[{self.i} {t:3.1f}s]:"
                      f" {maybe_sat_str} for delta {old_delta:.5f}" #/{best_example_delta:.5f}"
                      f" -> {delta:.5f} [{lower:.5f}, {upper:.5f}]", end="")
                if len(generated_examples):
                    print(" (!) ex.w/ delta", best_example_delta-self.guard)
                else: print()
            else: # no adv. can exist
                if delta == upper and lower == 0.0:
                    lower = delta
                    delta = 2.0 * delta
                    upper = delta
                elif upper != 0.0 and (upper - lower) / upper < 1e-5:
                    self.early_stop = True
                    print("STOPPING EARLY")
                    break
                else:
                    lower = delta
                    delta = lower + 0.5 * (upper - lower)
                t = timeit.default_timer() - self.start_time
                print(f"[{self.i} {t:3.1f}s]:"
                      f" UNSAT for delta {old_delta:.5f}"
                      f" -> {delta:.5f} [{lower:.5f}, {upper:.5f}]")

            if self.stop_condition(lower, upper):
                print(f"done early {lower} <= {delta} <= {upper}")
                break

        self.total_time = timeit.default_timer() - self.start_time
        self.total_time_p = time.process_time() - self.start_time_p
        self.delta_log.append((delta, lower, upper, timeit.default_timer()-self.start_time))

        return delta, lower, upper

    def get_step_time(self):
        rem_time = self.max_time - timeit.default_timer() + self.start_time
        #rem_time = min(rem_time, 2.0 * self.max_time / self.num_steps)
        rem_time /= (self.num_steps - self.i)
        #print("step time", rem_time, self.max_time, self.num_steps)
        return rem_time

    # returns (max_output_diff, [list of generated_example])
    def get_max_output_difference(self, delta, max_time):
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
        return example_delta


## \ingroup python
# \brief Robustness search using Veritas for the output estimate
class VeritasRobustnessSearch(RobustnessSearch):
    def __init__(self, source_at, target_at, example, mem_capacity=1024*1024*1024,
            **kwargs):
        super().__init__(example, **kwargs)
        self.mem_capacity = mem_capacity

        if source_at is not None and target_at is not None:
            self.at = target_at.concat_negated(source_at) # minimize source_at
        elif source_at is None:
            self.at = target_at
        elif target_at is None:
            self.at = AddTree().concat_negated(source_at)
        else:
            raise RuntimeError("source_at and target_at None")

    def get_search(self, delta):
        s = Search.max_output(self.at)
        #s.set_example(self.example)
        s.stop_when_optimal = True
        s.stop_when_upper_less_than = 0.0
        s.stop_when_num_solutions_exceeds = 1
        s.reject_solution_when_output_less_than = 0.0
        s.max_focal_size = 10000
        s.debug = False;
        s.auto_eps = True;

        s.set_mem_capacity(self.mem_capacity)
        box = [Domain(x-delta, x+delta) for x in self.example]
        s.prune(box)
        return s

    def get_max_output_difference(self, delta, max_time):
        s = self.get_search(delta)

        #milp = KantchelianOutputOpt(self.at, silent=True)
        #milp.constrain_to_box([Domain(x-delta, x+delta) for x in self.example])
        #milp.model.update()
        #m = milp.model.relax()
        #print()
        #print("====================================")
        ##m.setParam("OutputFlag", 0)
        #m.setParam("Threads", 1)
        #t=timeit.default_timer()
        #m.optimize()
        #t = timeit.default_timer() - t
        #print("SECONDS", t)
        #import gurobipy as gu
        #kanup = m.getAttr(gu.GRB.Attr.ObjBound)
        #kanlo = m.getAttr(gu.GRB.Attr.ObjVal)
        #print("mip relax", kanup, kanlo)
        #print("====================================")
        #print()

        stop_reason = s.step_for(max_time, 100)
        upper_bound = s.current_bounds()[1]
        #print("stop reason", stop_reason, upper_bound)
        max_output_diff = upper_bound
        generated_examples = []
        if s.num_solutions() > 0:
            best_sol = s.get_solution(0)
            if best_sol.output > 0.0:
                #max_output_diff = upper_bound if best_sol.eps != 1.0 else best_sol.output
                max_output_diff = best_sol.output
                closest = get_closest_example(best_sol, self.example)
                generated_examples = [closest]

        #print("VERITAS numsol", s.num_solutions())
        #print("VERITAS num rej sol", s.num_rejected_solutions)
        #print("VERITAS num steps", s.num_steps, "{:.2f}k/sec".format(s.num_steps / 1000 / s.time_since_start()))
        #print("VERITAS focal_size", np.mean([sn.avg_focal_size for sn in s.snapshots]))
        del s

        return max_output_diff, generated_examples

class MilpRobustnessSearch(RobustnessSearch):
    def __init__(self, source_at, target_at, example, silent=True, **kwargs):
        super().__init__(example, **kwargs)

        if source_at is not None and target_at is not None:
            self.at = target_at.concat_negated(source_at) # minimize source_at
        elif source_at is None:
            self.at = target_at
        elif target_at is None:
            self.at = AddTree().concat_negated(source_at)
        else:
            raise RuntimeError("source_at and target_at None")

        self.silent = silent

    def get_milp(self, delta, rem_time):
        milp = KantchelianOutputOpt(self.at, silent=self.silent, max_time=rem_time)
        box = [Domain(x-delta, x+delta) for x in self.example]
        milp.constrain_to_box(box)
        return milp

    def get_max_output_difference(self, delta, max_time):
        if self.silent:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                milp = self.get_milp(delta, max_time)
                milp.optimize()
        else:
            milp = self.get_milp(delta, max_time)
            milp.optimize()

        generated_examples = []
        max_output_diff = milp.objective_bound()
        if milp.has_solution():
            output_diff, intervals = milp.solution()
            if max_output_diff > 0.0:
                generated_examples = [get_closest_example(intervals, self.example)]
        return max_output_diff, generated_examples
