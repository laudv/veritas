import timeit, math

from dask.distributed import wait

from . import DomTree, DomTreeLeaf
from .verifier import Verifier, VerifierTimeout, VerifierNotExpr
from .verifier import in_domain_constraint


class VerifierFactory:
    """ Must be pickleable """

    def __call__(self, domtree_leaf):
        """ Override this method for your verifier factory.  """
        raise RuntimeError("Override this method in your own verifier "
            + "factory defining your problem's constraints.")

    def inv_logit(self, prob):
        """ Convert probability to raw output values for binary classification. """
        return -math.log(1.0 / x - 1)



class _VerifierFactoryWrap(VerifierFactory):
    def __init__(self, vfactory, add_domain_constraints_opt):
        self._vfactory = vfactory
        self.add_domain_constraints_opt = add_domain_constraints_opt

    def __call__(self, lk):
        v = self._vfactory(lk)
        if self.add_domain_constraints_opt:
            for instance_index in range(lk.num_instances()):
                v.add_constraint(in_domain_constraint(v,
                    lk.get_domains(instance_index),
                    instance=instance_index))
        return v





class DistributedVerifier:

    def __init__(self,
            client,
            domtree,
            verifier_factory,
            check_paths = True,
            num_initial_tasks = 1,
            stop_when_sat = False,
            add_domain_constraints = True,
            timeout_start = 30,
            timeout_max = 600,
            timeout_grow_rate = 1.5):

        assert isinstance(verifier_factory, VerifierFactory), "invalid verifier factory"

        self._timeout_start = float(timeout_start)
        self._timeout_max = float(timeout_max)
        self._timeout_rate = float(timeout_grow_rate)

        self._client = client # dask client
        self._domtree = domtree

        self._verifier_factory = _VerifierFactoryWrap(verifier_factory,
                add_domain_constraints)

        self._check_paths_opt = check_paths
        self._num_initial_tasks_opt = num_initial_tasks
        self._stop_when_sat_opt = stop_when_sat

        self._stop_flag = False
        self._print_queue = []

    def check(self):
        self.done_count = 0
        self.start_time = timeit.default_timer()

        self._fs = []
        self.results = {}

        # 1: loop over trees, check reachability of each path from root in
        # addtrees of all instances
        l0 = self._domtree.get_leaf(self._domtree.tree().root())
        if self._check_paths_opt:
            t0 = timeit.default_timer()
            l0 = self._check_paths(l0)
            t1 = timeit.default_timer()
            self.results["check_paths_time"] = t1 - t0

        # domtree_node_id => result info per instance + additional info
        self.results["num_leafs"] = [l0.addtree(i).num_leafs()
                for i in range(l0.num_instances())]
        self.results[0] = self._init_results(l0)

        # 2: splits until we have a piece of work for each worker
        if self._num_initial_tasks_opt > 1:
            t0 = timeit.default_timer()
            lks = self._generate_splits(l0, self._num_initial_tasks_opt)
            t1 = timeit.default_timer()
            self.results["generate_splits_time"] = t1 - t0
        else:
            lks = [l0]

        # 3: submit verifier 'check' tasks for each item in `ls`
        for lk in lks:
            f = self._make_verify_future(lk, self._timeout_start)
            self._fs.append(f)

        # 4: wait for future to complete, act on result
        # - if sat/unsat -> done (finish if sat if opt set)
        # - if new split -> schedule two new tasks
        while len(self._fs) > 0: # while task are running...
            if self._stop_flag:
                self._print("Stop flag: cancelling remaining tasks")
                for f in self._fs:
                    f.cancel()
                    self._stop_flag = False
                break

            wait(self._fs, return_when="FIRST_COMPLETED")
            next_fs = []
            for f in self._fs:
                if f.done(): next_fs += self._handle_done_future(f)
                else:        next_fs.append(f)
            self._fs = next_fs
            self._print_flush()

        self.results["check_time"] = timeit.default_timer() - self.start_time

    def _check_paths(self, l0):
        # update reachabilities in domtree root leaf 0 in parallel
        fs = []
        for instance_index in range(l0.num_instances()):
            for tree_index in range(len(l0.addtree(instance_index))):
                f = self._client.submit(DistributedVerifier._check_tree_paths,
                        l0, instance_index, tree_index, self._verifier_factory)
                fs.append(f)
        wait(fs)
        for f in fs:
            if not f.done():
                raise RuntimeError("future not done?")
            if f.exception():
                raise f.exception() from RuntimeError("exception on worker")
        return DomTreeLeaf.merge([f.result() for f in fs])

    def _generate_splits(self, l0, ntasks):
        # split domtrees until we have ntask `Subspace`s; this runs locally
        l0.find_best_split()
        lks = [l0]

        while len(lks) < ntasks:
            max_score = 0
            max_lk = None

            for lk in lks:
                if lk.score > max_score:
                    max_score = lk.score
                    max_lk = lk

            if max_lk is None:
                raise RuntimeError("no more splits!")

            lks.remove(max_lk)
            lks += self._split_domtree(max_lk, True)

            self._print_flush()
        return lks

    def _split_domtree(self, lk, find_best_split):
        nid = lk.domtree_leaf_id()
        split = lk.get_best_split()
        score, balance = lk.score, lk.balance

        self._domtree.apply_leaf(lk) # lk's fields are invalid after .split(lk)

        l, r = self._domtree.tree().left(nid), self._domtree.tree().right(nid)
        lk_l = self._domtree.get_leaf(l)
        lk_r = self._domtree.get_leaf(r)

        if find_best_split:
            lk_l.find_best_split()
            lk_r.find_best_split()

        self.results[nid]["split"] = split
        self.results[nid]["score"] = score
        self.results[nid]["balance"] = balance
        self.results[nid]["children"] = [l, r]

        self.results[l] = self._init_results(lk_l)
        self.results[r] = self._init_results(lk_r)
        self.results[l]["parent"] = nid
        self.results[r]["parent"] = nid

        self._print("SPLIT l{}: {} into {}, {}, score {} ".format(
            nid, split, l, r, score))

        return [lk_l, lk_r]

    def _handle_done_future(self, f):
        t = f.result()
        status, check_time = t[0], t[1]

        self._print("{} for domtree leaf {} in {:.2f}s (timeout={:.1f}s)".format(status,
            f.domtree_leaf_id, check_time, f.timeout))

        self.results[f.domtree_leaf_id]["status"] = status
        self.results[f.domtree_leaf_id]["check_time"] = check_time

        # We're finished with this branch!
        if status != Verifier.Result.UNKNOWN:
            self.done_count += 1
            model = t[2]
            self.results[f.domtree_leaf_id]["model"] = model
            if status.is_sat() and self._stop_when_sat_opt:
                self._stop_flag = True
            return []

        else: # We timed out, split and try again
            lk = t[2]
            next_timeout = min(self._timeout_max, self._timeout_rate * f.timeout)

            new_lks = self._split_domtree(lk, False)
            new_fs = [self._make_verify_future(lk, next_timeout) for lk in new_lks]

            return new_fs




    def _make_verify_future(self, lk, timeout):
        f = self._client.submit(DistributedVerifier._verify_fun,
                lk, timeout, self._verifier_factory)

        f.timeout = timeout
        f.domtree_leaf_id = lk.domtree_leaf_id()
        return f

    def _init_results(self, lk):
        return {
            "num_unreachable": self._num_unreachable(lk),
            "bounds": self._tree_bounds(lk)
        }

    def _num_unreachable(self, lk):
        return sum(map(lambda i: lk.num_unreachable(i), range(lk.num_instances())))

    def _tree_bounds(self, lk):
        bounds = []
        for i in range(lk.num_instances()):
            lo, hi = 0.0, 0.0
            for tree_index in range(len(lk.addtree(i))):
                bnds = lk.get_tree_bounds(i, tree_index)
                lo += bnds[0]
                hi += bnds[1]
            bounds.append((lo, hi))
        return bounds

    def _print(self, msg):
        self._print_queue.append(msg)

    def _print_flush(self):
        for msg in self._print_queue:
            self._print_msg(msg)
        self._print_queue = []

    def _print_msg(self, msg):
        t = int(timeit.default_timer() - self.start_time)
        m, s = t // 60, t % 60
        h, m = m // 60, m % 60
        done = self.done_count
        rem = len(self._fs) if hasattr(self, "_fs") else -1
        print(f"[{h}h{m:02d}m{s:02d}s {done:>4} {rem:<4}]", msg)





    # - WORKERS ------------------------------------------------------------- #

    @staticmethod
    def _check_tree_paths(lk, instance_index, tree_index,
            v_or_vfactory, only_feat_id = -1):
        addtree = lk.addtree(instance_index)
        tree = addtree[tree_index]
        stack = [(tree.root(), True)]

        v = v_or_vfactory
        if not isinstance(v, Verifier):
            v = v_or_vfactory(lk)

        v.instance(instance_index).mark_unreachable_paths(tree_index, only_feat_id)

        return lk

    @staticmethod
    def _verify_fun(lk, timeout, vfactory):
        v = vfactory(lk)

        # TODO re-check reachabilities in other trees due to new constraint

        ## this `ls` is a result of splitting on (instance_index, feat_id)
        ## check the other trees' reachabilities again!
        #if split_instance_index != -1 and split_feat_id != -1 and len(ls) > 1:
        #    for instance_index, lk in enumerate(ls):
        #        if instance_index == split_instance_index: continue # already done by subspaces
        #        for tree_index in range(len(addtrees[instance_index])):
        #            DistributedVerifier._check_tree_paths(addtrees, ls,
        #                    instance_index, tree_index, v,
        #                    only_feat_id=split_feat_id)
        #    # TODO this work is lost if lk does not go back to its subspaces -> done when UNKNOWN

        v.set_timeout(timeout)
        v.add_all_trees()
        try:
            status = v.check()
            model = {}
            if status.is_sat():
                model = v.model()
                model["family"] = v.model_family(model)

            return status, v.check_time, model

        except VerifierTimeout as e:
            lk.find_best_split()

            print(f"timeout after {e.unk_after} (timeout = {timeout}) best split = {lk.get_best_split()}")

            return Verifier.Result.UNKNOWN, v.check_time, lk
