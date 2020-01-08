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
            print("ls?", ls, "waar is ls??")
            l0 = self._check_paths(ls)
            t1 = timeit.default_timer()
            self.results["check_paths_time"] = t1 - t0

        print("#TODO 3")
        return;

        # split_id => result info per instance + additional info
        self.results["num_leafs"] = [inst.addtree.num_leafs() for inst in self._instances]
        self.results[0] = self._init_results(ls)

        # 2: splits until we have a piece of work for each worker
        if self._num_initial_tasks_opt > 1:
            t0 = timeit.default_timer()
            lss = self._generate_splits(ls, self._num_initial_tasks_opt)
            t1 = timeit.default_timer()
            self.results["generate_splits_time"] = t1 - t0
        else:
            lss = [(self._new_split_id(), ls)]


        # 3: submit verifier 'check' tasks for each item in `ls`
        for split_id, ls in lss:
            f = self._make_verify_future(split_id, ls, self._timeout_start)
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

    def _generate_splits(self, ls, ntasks):
        # split domtrees until we have ntask `Subspace`s; this runs locally
        lss = [(self._new_split_id(), ls)]
        for instance_index, lk in enumerate(ls):
            lk.find_best_domtree_split(self._instances[instance_index].addtree)

        while len(lss) < ntasks:
            print(list(map(lambda ls: list(map(lambda lk: lk.domtree_node_id(), ls[1])), lss)))
            max_score = 0
            max_instance_index = -1
            max_ls = None
            max_split_id = -1

            for split_id, ls in lss:
                for instance_index, lk in enumerate(ls):
                    if lk.split_score > max_score:
                        max_score = lk.split_score
                        max_instance_index = instance_index
                        max_ls = ls
                        max_split_id = split_id

            if max_ls is None:
                raise RuntimeError("no more splits!")

            print("max: ", max_instance_index, max_ls[max_instance_index].domtree_node_id(),
                    list(map(lambda lk: lk.domtree_node_id(), max_ls)))

            lss.remove((max_split_id, max_ls))
            lss += self._split_domtree(max_split_id, max_ls, max_instance_index, True)
        return lss

    def _split_domtree(self, split_id, ls, max_instance_index, find_best_domtree_split):
        lk = ls[max_instance_index]
        inst = self._instances[max_instance_index]
        nid = lk.domtree_node_id()
        split = lk.get_best_split()
        split_score = lk.split_score
        split_balance = lk.split_balance

        for instk in self._instances:
            print(instk.subspaces.domtree())

        print("splitting", max_instance_index, lk.domtree_node_id())
        inst.subspaces.split(lk) # lk's fields are invalid after .split(lk)

        domtree = inst.subspaces.domtree()
        l, r = domtree.left(nid), domtree.right(nid)
        lk_l = inst.subspaces.get_subspace(l)
        lk_r = inst.subspaces.get_subspace(r)

        if find_best_domtree_split:
            lk_l.find_best_domtree_split(inst.addtree)
            lk_r.find_best_domtree_split(inst.addtree)

        split_id_l = self._new_split_id()
        split_id_r = self._new_split_id()
        ls_l = ls.copy(); ls_l[max_instance_index] = lk_l
        ls_r = ls.copy(); ls_r[max_instance_index] = lk_r

        self.results[split_id]["split"] = split
        self.results[split_id]["split_score"] = split_score
        self.results[split_id]["split_balance"] = split_balance
        self.results[split_id]["instance_index"] = inst.index
        self.results[split_id]["domtree_node_id"] = nid
        self.results[split_id]["next_split_ids"] = [split_id_l, split_id_r]

        self.results[split_id_l] = self._init_results(ls_l)
        self.results[split_id_r] = self._init_results(ls_r)
        self.results[split_id_l]["prev_split_id"] = split_id
        self.results[split_id_r]["prev_split_id"] = split_id

        self._print("SPLIT {}:{} {} into {}, {}, score {} ".format(
            inst.index, nid, lk.get_best_split(), l, r, split_score))

        return [(split_id_l, ls_l), (split_id_r, ls_r)]

    def _handle_done_future(self, f):
        t = f.result()
        status, check_time = t[0], t[1]

        self._print("{} for task {} in {:.2f}s (timeout={:.1f}s)".format(status,
            f.split_id, check_time, f.timeout))

        self.results[f.split_id]["status"] = status
        self.results[f.split_id]["check_time"] = check_time
        self.results[f.split_id]["split_id"] = f.split_id

        # We're finished with this branch!
        if status != Verifier.Result.UNKNOWN:
            self.done_count += 1
            model = t[2]
            self.results[f.split_id]["model"] = model
            if status.is_sat() and self._stop_when_sat_opt:
                self._stop_flag = True
            return []

        else: # We timed out, split and try again
            ls = t[2]
            next_timeout = min(self._timeout_max, self._timeout_rate * f.timeout)

            max_score = 0
            max_instance_index = -1
            for instance_index, lk in enumerate(ls):
                if lk.split_score > max_score:
                    max_score = lk.split_score
                    max_instance_index = instance_index

            new_ls = self._split_domtree(f.split_id, ls, max_instance_index, False)
            new_fs = [self._make_verify_future(sid, ls, next_timeout) for sid, ls in new_ls]

            return new_fs




    def _new_split_id(self):
        split_id = self._split_id
        self._split_id += 1
        return split_id

    def _make_verify_future(self, split_id, ls, timeout):
        split_instance_index, split_feat_id = -1, 1
        if "prev_split_id" in self.results[split_id] \
                and self._verifier_factory.add_domain_constraints_opt \
                and self._check_paths_opt:
            prev_split_id = self.results[split_id]["prev_split_id"]
            split = self.results[prev_split_id]["split"]
            split_instance_index = self.results[prev_split_id]["instance_index"]

        f = self._client.submit(DistributedVerifier._verify_fun,
                self._addtrees_fut, ls, timeout,
                self._verifier_factory,
                split_instance_index, split_feat_id)

        f.timeout = timeout
        f.split_id = split_id
        self._split_id += 1
        return f

    def _init_results(self, ls):
        return {
            "num_unreachable": self._num_unreachable(ls),
            "bounds": self._tree_bounds(ls)
        }

    def _num_unreachable(self, ls):
        return sum(map(lambda lk: lk.num_unreachable(), ls))

    def _tree_bounds(self, ls):
        bounds = []
        for at, lk in zip(self._addtrees, ls):
            lo, hi = 0.0, 0.0
            for tree_index in range(len(at)):
                bnds = lk.get_tree_bounds(at, tree_index)
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
    def _verify_fun(addtrees, ls, timeout, vfactory,
            split_instance_index = -1, split_feat_id = -1):
        v = vfactory(addtrees, ls)

        # TODO re-check reachabilities in other trees due to new constraint

        # this `ls` is a result of splitting on (instance_index, feat_id)
        # check the other trees' reachabilities again!
        if split_instance_index != -1 and split_feat_id != -1 and len(ls) > 1:
            for instance_index, lk in enumerate(ls):
                if instance_index == split_instance_index: continue # already done by subspaces
                for tree_index in range(len(addtrees[instance_index])):
                    DistributedVerifier._check_tree_paths(addtrees, ls,
                            instance_index, tree_index, v,
                            only_feat_id=split_feat_id)
            # TODO this work is lost if lk does not go back to its subspaces

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
            print(f"timeout after {e.unk_after} (timeout = {timeout}) finding best split...")
            for instance_index, lk in enumerate(ls):
                if not lk.has_best_split():
                    lk.find_best_domtree_split(addtrees[instance_index])

            return Verifier.Result.UNKNOWN, v.check_time, ls
