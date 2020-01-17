import timeit, math, time

from dask.distributed import wait

from . import DomTree, DomTreeLeaf
from .verifier import Verifier, VerifierTimeout, VerifierNotExpr
from .verifier import in_domain_constraint


class VerifierFactory:
    """ Must be pickleable """

    def __call__(self, domtree_leaf, path_checking):
        """
        Override this method for your verifier factory.

        You can return a less constrained verifier for the path_checking part.
        """
        raise RuntimeError("Override this method in your own verifier "
            + "factory defining your problem's constraints.")

    def inv_logit(self, prob):
        """ Convert probability to raw output values for binary classification. """
        return -math.log(1.0 / x - 1)



class _VerifierFactoryWrap(VerifierFactory):
    def __init__(self, vfactory, add_domain_constraints_opt):
        self._vfactory = vfactory
        self.add_domain_constraints_opt = add_domain_constraints_opt

    def __call__(self, lk, path_checking):
        v = self._vfactory(lk, path_checking)
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
            stop_when_num_sats = 1,
            add_domain_constraints = True,
            global_timeout = 0,
            timeout_start = 30,
            timeout_max = 600,
            timeout_grow_rate = 1.5):

        assert isinstance(verifier_factory, VerifierFactory), "invalid verifier factory"

        self._timeout_start = float(timeout_start)
        self._timeout_max = float(timeout_max)
        self._timeout_rate = float(timeout_grow_rate)
        self._nworkers = sum(client.nthreads().values())

        self._client = client # dask client
        self._domtree = domtree

        self._verifier_factory = _VerifierFactoryWrap(verifier_factory,
                add_domain_constraints)

        self._check_paths_opt = check_paths
        self._num_initial_tasks_opt = num_initial_tasks
        self._stop_when_num_sats_opt = stop_when_num_sats
        self._global_timeout_opt = global_timeout

        self._stop_flag = False
        self._print_queue = []

    def check(self):
        self.done_count = 0
        self.start_time = timeit.default_timer()
        self.sat_count = 0

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

        self._print("num_leafs {}".format(self.results["num_leafs"]))
        self._print_flush()

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
            if self._global_timeout_opt > 0:
                t = timeit.default_timer() - self.start_time
                if t > self._global_timeout_opt:
                    self._print("Global timeout: cancelling remaining tasks")
                    for f in self._fs:
                        f.cancel()
                    break

            wait(self._fs, return_when="FIRST_COMPLETED")

            next_fs = []
            for f in self._fs:
                if f.done(): next_fs += self._handle_done_future(f)
                else:        next_fs.append(f)
            self._fs = next_fs
            self._print_flush()

        self.results["check_time"] = timeit.default_timer() - self.start_time
        self._print_flush()

    def _check_paths(self, l0):
        num_unreachable_before = [l0.num_unreachable(i) for i in range(l0.num_instances())]

        self._print("checking paths")
        self._print_flush()

        # update reachabilities in domtree root leaf 0 in parallel
        fs = []
        for instance_index in range(l0.num_instances()):
            for tree_index in range(len(l0.addtree(instance_index))):
                f = self._client.submit(DistributedVerifier._check_tree_paths,
                        l0, instance_index, tree_index, self._verifier_factory)
                fs.append(f)

        wait(fs)

        l0 = DomTreeLeaf.merge([f.result() for f in fs])
        num_unreachable_after = [l0.num_unreachable(i) for i in range(l0.num_instances())]

        self._print("check_paths({}): {} -> {}".format(l0.domtree_leaf_id(),
            num_unreachable_before, num_unreachable_after))
        self._print_flush()

        return l0

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
        status, check_time, num_leafs = t[0], t[1], t[2]

        self._print("{} for l{} in {:.2f}s (timeout={:.1f}s, #leafs={})".format(status,
            f.domtree_leaf_id, check_time, f.timeout, num_leafs))

        self.results[f.domtree_leaf_id]["status"] = status
        self.results[f.domtree_leaf_id]["check_time"] = check_time
        self.results[f.domtree_leaf_id]["num_leafs"] = num_leafs

        # We're finished with this branch!
        if status != Verifier.Result.UNKNOWN:
            self.done_count += 1
            model = t[3]
            self.results[f.domtree_leaf_id]["model"] = model
            if status.is_sat(): self.sat_count += 1
            if self.sat_count >= self._stop_when_num_sats_opt:
                self._stop_flag = True
            return []

        else: # We timed out, split and try again
            lk = t[3]
            self.results[f.domtree_leaf_id]["num_unreachable_after"] = self._num_unreachable(lk)
            next_timeout = min(self._timeout_max, self._timeout_rate * f.timeout)

            new_lks = self._split_domtree(lk, False)
            new_fs = [self._make_verify_future(lk, next_timeout) for lk in new_lks]

            return new_fs




    def _make_verify_future(self, lk, timeout):
        nid = lk.domtree_leaf_id()
        tree = self._domtree.tree()
        parent_split = None

        if not tree.is_root(nid) and self._check_paths_opt:
            pid = tree.parent(nid)
            parent_split = tree.get_split(pid)

        f = self._client.submit(DistributedVerifier._verify_fun,
                lk, timeout, self._verifier_factory, parent_split)

        f.timeout = timeout
        f.domtree_leaf_id = nid
        return f

    def _init_results(self, lk):
        return {
            "num_unreachable_before": self._num_unreachable(lk),
            "bounds": self._tree_bounds(lk),
            "domains": [lk.get_domains(i) for i in range(lk.num_instances())]
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
            v = v_or_vfactory(lk, True)

        v.instance(instance_index).mark_unreachable_paths(tree_index, only_feat_id)

        return lk

    @staticmethod
    def _recheck_tree_paths(lk, v, feat_id):
        for i in range(lk.num_instances()):
            num_reach_before = lk.num_unreachable(i)
            for tree_index in range(len(lk.addtree(i))):
                lk = DistributedVerifier._check_tree_paths(lk, i, tree_index, v, feat_id)
            print("_recheck_tree_paths({}): num_unreachable({}): {} -> {}".format(
                lk.domtree_leaf_id(), i, num_reach_before, lk.num_unreachable(i)))
        return lk

    @staticmethod
    def _verify_fun(lk, timeout, vfactory, parent_split = None):
        v = vfactory(lk, False)

        # Re-checking reachabilities after split, only for splits involving feat_id
        if parent_split is not None and lk.num_instances() > 1:
            feat_id = parent_split[1].feat_id
            lk = DistributedVerifier._recheck_tree_paths(lk, v, feat_id)

        v.set_timeout(timeout)
        v.add_all_trees()
        num_leafs = [v.instance(i).leaf_count for i in range(lk.num_instances())]
        try:
            status = v.check()
            model = {}
            if status.is_sat():
                model = v.model()
                model["family"] = v.model_family(model)

            return status, v.check_time, num_leafs, model

        except VerifierTimeout as e:
            lk.find_best_split()

            print(f"timeout after {e.unk_after} (timeout = {timeout}) best split = {lk.get_best_split()}")

            return Verifier.Result.UNKNOWN, v.check_time, num_leafs, lk
