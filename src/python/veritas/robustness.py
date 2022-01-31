## \file robustness.py
#
# Copyright 2022 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Author: Laurens Devos

import timeit, time
import numpy as np

from . import AddTree, GraphOutputSearch, get_closest_example, Domain

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
                print(f"[{self.i}]: {maybe_sat_str} delta update: {old_delta:.5f}/{best_example_delta:.5f}"
                      f" -> {delta:.5f} [{lower:.5f}, {upper:.5f}]")
                if len(generated_examples):
                    print(f"   -> generated adv.example w/ delta",
                            best_example_delta-self.guard)
            else: # no adv. can exist
                if delta == upper:
                    lower = delta
                    delta = 2.0 * delta
                    upper = delta
                else:
                    lower = delta
                    delta = lower + 0.5 * (upper - lower)
                print(f"[{self.i}]: UNSAT delta update: {old_delta:.3f}"
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

        self.log = []

    def get_search(self, delta):
        s = GraphOutputSearch(self.at)
        #s.stop_when_num_solutions_equals = 1 # !! could be solution with bad eps not good enough yet
        #s.stop_when_solution_output_greater_than = 0.0
        s.stop_when_solution_eps_equals = 1.0
        s.stop_when_up_bound_less_than = 0.0
        s.set_mem_capacity(self.mem_capacity)
        box = [Domain(x-delta, x+delta) for x in self.example]
        s.prune(box)
        return s

    def get_max_output_difference(self, delta):
        s = self.get_search(delta)

        rem_time = self.max_time - timeit.default_timer() + self.start_time
        #rem_time = min(rem_time, 2.0 * self.max_time / self.num_steps)
        rem_time /= (self.num_steps - self.i)
        #print("step time", rem_time, self.max_time, self.num_steps)

        if rem_time < 0.0:
            return None

        s.step_for(rem_time, 50)

        upper_bound = min(s.current_bounds()[1:])
        max_output_diff = upper_bound
        generated_examples = []
        if s.num_solutions() > 0:
            best_sol = s.get_solution(0)
            if best_sol.output > 0.0:
                #print(f"Veritas generated example", best_sol)
                max_output_diff = upper_bound if best_sol.eps != 1.0 else best_sol.output
                closest = get_closest_example(best_sol, self.example)
                generated_examples = [closest]

        return max_output_diff, generated_examples

class MilpRobustnessSearch(RobustnessSearch):
    def __init__(self, source_at, target_at, example, **kwargs):
        super().__init__(example, **kwargs)

        if source_at is not None and target_at is not None:
            self.at = target_at.concat_negated(source_at) # minimize source_at
        elif source_at is None:
            self.at = target_at
        elif target_at is None:
            self.at = AddTree().concat_negated(source_at)
        else:
            raise RuntimeError("source_at and target_at None")

    def get_milp(self, delta, rem_time):
        milp = KantchelianOutputOpt(self.at, max_time=rem_time)
        box = [Domain(x-delta, x+delta) for x in self.example]
        milp.constrain_to_box(box)
        return milp

    def get_max_output_difference(self, delta):
        rem_time = self.max_time - timeit.default_timer() + self.start_time
        #rem_time = min(rem_time, 2.0 * self.max_time / self.num_steps)
        rem_time /= (self.num_steps - self.i)
        print("step time", rem_time, self.max_time, self.num_steps)

        if rem_time < 0.0:
            return None

        milp = self.get_milp(delta, rem_time)
        milp.optimize()

        generated_examples = []
        max_output_diff = milp.objective_bound()
        if milp.has_solution():
            output_diff, intervals = milp.solution()
            if max_output_diff > 0.0:
                generated_examples = [get_closest_example(intervals, self.example)]
        return max_output_diff, generated_examples
