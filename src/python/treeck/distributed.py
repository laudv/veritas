import codecs, time, io, json

from enum import Enum
from dask.distributed import wait, get_worker

from .pytreeck import AddTree, RealDomain, SplitTreeLeaf
from .verifier import Verifier, VerifierTimeout


class DistributedVerifier:

    def __init__(self,
            client,
            splittree,
            verifier_factory,
            check_paths = True,
            saturate_workers_from_start = True,
            saturate_workers_factor = 1.0,
            stop_when_sat = False):

        self._timeout_start = 60
        self._timeout_max = 3600
        self._timeout_rate = 1.5

        self._client = client # dask client
        self._st = splittree
        self._at = splittree.addtree()
        self._at_fut = client.scatter(self._at, broadcast=True) # distribute addtree to all workers
        self._verifier_factory = verifier_factory # (addtree, splittree_leaf) -> Verifier

        self._check_paths_opt = check_paths
        self._saturate_workers_opt = saturate_workers_from_start
        self._saturate_workers_factor_opt = saturate_workers_factor
        self._stop_when_sat_opt = stop_when_sat

    def check(self):
        l0 = self._st.get_leaf(0)

        # TODO add domain constraints to verifier

        # 1: loop over trees, check reachability of each path from root
        if self._check_paths_opt:
            l0 = self._check_paths(l0)

        # 2: splits until we have a piece of work for each worker
        if self._saturate_workers_opt:
            nworkers = sum(self._client.nthreads().values())
            ntasks = int(round(self._saturate_workers_factor_opt * nworkers))
            ls = self._generate_splits(l0, ntasks)
        else:
            ls = [l0]

        # 3: submit verifier 'check' tasks for each
        # TODO fs = submit(ls)

        # 4: wait for future to complete, act on result
        # - if sat/unsat -> done (finish if sat if opt set)
        # - if new split -> schedule two new tasks
        # TODO check loop wait(fs, return_when="FIRST_COMPLETED")

    def _check_paths(self, l0):
        # update reachabilities in splittree_leaf 0 in parallel
        fs = []
        for tree_index in range(len(self._at)):
            f = self._client.submit(DistributedVerifier._check_tree_paths,
                self._at_fut,
                tree_index,
                l0,
                self._verifier_factory)
            fs.append(f)
            pass
        wait(fs)
        return SplitTreeLeaf.merge(list(map(lambda f: f.result(), fs)))

    def _generate_splits(self, l0, ntasks):
        # split and collects splittree_leafs
        l0.find_best_domtree_split(self._at)
        ls = [l0]
        while len(ls) < ntasks:
            max_score = 0
            max_lx = None
            for lx in ls:
                print("lx", lx, "with score", lx.split_score)
                if lx.split_score > max_score:
                    max_score = lx.split_score
                    max_lx = lx

            ls.remove(max_lx)
            nid = max_lx.domtree_node_id()

            print("splitting domtree_node_id", nid, max_lx.get_best_split())
            self._st.split(max_lx)

            domtree = self._st.domtree()
            l, r = domtree.left(nid), domtree.right(nid)
            ll, lr = self._st.get_leaf(l), self._st.get_leaf(r)
            ll.find_best_domtree_split(self._at)
            lr.find_best_domtree_split(self._at)
            ls += [ll, lr]

        return ls


    # - WORKERS ------------------------------------------------------------- #

    @staticmethod
    def _check_tree_paths(at, tree_index, l0, vfactory):
        tree = at[tree_index]
        stack = [(tree.root(), True)]
        v = vfactory(at, l0)

        while len(stack) > 0:
            node, path_constraint = stack.pop()

            l, r = tree.left(node), tree.right(node)
            feat_id, split_value = tree.get_split(node)
            xvar = v.xvar(feat_id)

            if tree.is_internal(l) and l0.is_reachable(tree_index, l):
                path_constraint_l = (xvar < split_value) & path_constraint;
                if v.check(path_constraint_l).is_sat():
                    stack.append((l, path_constraint_l))
                else:
                    print(f"unreachable  left: {tree_index} {l}")
                    l0.mark_unreachable(tree_index, l)

            if tree.is_internal(r) and l0.is_reachable(tree_index, r):
                path_constraint_r = (xvar >= split_value) & path_constraint;
                if v.check(path_constraint_r).is_sat():
                    stack.append((r, path_constraint_r))
                else:
                    print(f"unreachable right: {tree_index} {r}")
                    l0.mark_unreachable(tree_index, r)

        return l0


    #def run(self):
    #    at_enc = DistributedVerifier._enc_at(self._addtree)
    #    splittree = SplitTree([], at_enc)

    #    f0 = self._client.submit(DistributedVerifier._split_fun, None, at_enc,
    #            self._branch_factor)
    #    f0.meta = Meta()
    #    f0.meta.splittree = splittree
    #    f0.meta.timeout = self._timeout_start
    #    fs = [f0]

    #    # Compute reachabilities while trees are being split
    #    f1 = self._client.submit(DistributedVerifier._reachability_fun,
    #            self._verifier_factory, at_enc)
    #    self._reachability = f1.result() # blocks

    #    while len(fs) > 0: # while task are running...
    #        wait(fs, return_when="FIRST_COMPLETED")
    #        next_fs = []
    #        for f in fs:
    #            if f.done(): next_fs += self._handle_done_future(f)
    #            else:        next_fs.append(f)
    #        fs = next_fs

    #    print("done!")
    #    print(splittree)
    #    print(splittree.to_json())

    #def _handle_done_future(self, future):
    #    assert future.done()
    #    result = future.result()
    #    if result[0] == DistributedVerifier.TaskType.SPLIT:
    #        subprobs = result[1]
    #        return self._handle_split_result(future.meta, subprobs)
    #    if result[0] == DistributedVerifier.TaskType.VERIFY:
    #        status, doms, at_enc, verify_time = result[1:]
    #        return self._handle_verify_result(future.meta, status, doms, at_enc, verify_time)
    #    raise RuntimeError("unhandled future")

    #def _handle_split_result(self, meta, subprobs):
    #    print("SPLIT HANDLER with", len(subprobs), "subprobs")
    #    timeout = meta.timeout
    #    fs = []
    #    children = meta.splittree.split(subprobs)
    #    for subsplittree, (doms, at_enc) in zip(children, subprobs):
    #        f = self._client.submit(DistributedVerifier._verify_fun,
    #                self._verifier_factory, doms, at_enc,
    #                self._reachability, timeout)
    #        f.meta = Meta()
    #        f.meta.timeout = meta.timeout
    #        f.meta.splittree = subsplittree
    #        fs.append(f)
    #    return fs

    #def _handle_verify_result(self, meta, status, doms, at_enc, verify_time):
    #    print("VERIFY HANDLER", status)
    #    meta.splittree.status = status
    #    meta.splittree.verify_time = verify_time
    #    meta.splittree.timeout = meta.timeout
    #    if status == Verifier.Result.UNKNOWN:
    #        assert doms is not None
    #        assert at_enc is not None
    #        f = self._client.submit(DistributedVerifier._split_fun,
    #                doms, at_enc, self._branch_factor)
    #        f.timeout = min(self._timeout_max, meta.timeout * self._timeout_rate)
    #        f.splittree = meta.splittree
    #        return [f]
    #    return []

    #@staticmethod
    #def _reachability_fun(vfactory, at_enc):
    #    at = DistributedVerifier._dec_at(at_enc)
    #    num_features = at.num_features()
    #    domains = [RealDomain() for i in range(num_features)]
    #    v = vfactory(domains, at)
    #    assert isinstance(v._strategy, SplitCheckStrategy) # only support this for now
    #    v._strategy.verify_setup()
    #    return v._strategy._reachability

    #@staticmethod
    #def _split_fun(domains, at_enc, branch_factor):
    #    at = DistributedVerifier._dec_at(at_enc)
    #    if domains is None: sp = SearchSpace(at)
    #    else:               sp = SearchSpace(at, domains)

    #    sp.split(branch_factor)

    #    subprobs = []
    #    for leaf_id in sp.leafs():
    #        sub_doms = sp.get_domains(leaf_id)
    #        sub_at = sp.get_pruned_addtree(leaf_id)
    #        sub_at_enc = DistributedVerifier._enc_at(sub_at)
    #        subprobs.append((sub_doms, sub_at_enc))

    #    return DistributedVerifier.TaskType.SPLIT, subprobs

    #@staticmethod
    #def _verify_fun(vfactory, domains, at_enc, reachability, timeout):
    #    at = DistributedVerifier._dec_at(at_enc)
    #    v = vfactory(domains, at)
    #    assert isinstance(v._strategy, SplitCheckStrategy) # only support this for now
    #    v._strategy.set_reachability(reachability)
    #    v.set_timeout(timeout)
    #    try:
    #        status = v.verify() # maybe also return logs, stats
    #    except VerifierTimeout as e:
    #        status = Verifier.Result.UNKNOWN
    #        print(f"timeout after ", e.unk_after, " (timeout =", timeout, ")")
    #    return DistributedVerifier.TaskType.VERIFY, status, domains, at_enc, v.verify_time

    #@staticmethod
    #def _enc_at(at):
    #    b = bytes(at.to_json(), encoding="ascii")
    #    return codecs.encode(b, encoding="zlib")

    #@staticmethod
    #def _dec_at(b):
    #    at_json = codecs.decode(b, encoding="zlib").decode("ascii")
    #    return AddTree.from_json(at_json)
