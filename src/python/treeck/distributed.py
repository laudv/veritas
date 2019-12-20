import codecs, time, io, json
from enum import Enum
from dask.distributed import wait

from .pytreeck import AddTree, RealDomain
from .verifier import Verifier, VerifierTimeout


class DistributedVerifier:

    def __init__(self,
            client,
            splittree,
            verifier_factory,
            check_paths = True,
            saturate_workers_from_start = True,
            stop_when_sat = False):

        self._timeout_start = 60
        self._timeout_max = 3600
        self._timeout_rate = 1.5

        self._client = client # dask client
        self._st = splittree # _st.addtree() has addtree
        self._verifier_factory = verifier_factory # (addtree, splittree_leaf) -> Verifier

        self._check_paths_opt = check_paths
        self._saturate_workers_opt = saturate_workers_from_start
        self._stop_when_sat_opt = stop_when_sat

    def check(self):
        l0 = self._st.get_leaf(0)

        # 0: distribute addtree to all workers
        # TODO

        # 1: loop over trees, check reachability of each path from root
        if self._check_paths_opt:
            self._check_paths(l0)

        # 2: splits until we have a piece of work for each worker
        if self._saturate_workers_opt:
            nworkers = sum(self._client.nthreads().values())
            ls = self._generate_tasks(l0, nworkers)
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
        pass

    def _generate_tasks(self, l0, ntasks):
        # split and collects splittree_leafs
        pass


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
