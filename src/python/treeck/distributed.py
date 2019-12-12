import codecs, time, io, json
from enum import Enum
from dask.distributed import wait

from .pytreeck import SearchSpace, AddTree
from .verifier import Verifier, VerifierTimeout, SplitCheckStrategy, RealDomain


class Meta:
    pass


class SplitTree:
    id = 0

    def __init__(self, domains, at_enc, root=None, parent=None):
        self.id = SplitTree.id
        SplitTree.id += 1

        self.domains = [(i, d.lo, d.hi) for i, d in enumerate(domains) if not d.is_everything()]
        self.at_enc = at_enc
        self.root = self if root is None else root
        self.parent = self if parent is None else parent
        self.children = []

        # additional info
        self.status = None
        self.verify_time = 0.0

    def split(self, subprobs): # subprobs = [(doms, at_enc) ... ]
        self.children = [SplitTree(*prob, root=self.root, parent=self)
                for prob in subprobs]
        return self.children

    def __getitem__(self, index):
        return self.children[index]

    def __str__(self, depth=0):
        s = io.StringIO()
        print("  "*depth, "-", self.id, self.domains, self.status, self.verify_time, file=s)
        for c in self.children:
            print(c.__str__(depth=depth+1), file=s, end="")
        return s.getvalue()

    def to_obj(self):
        at = DistributedVerifier._dec_at(self.at_enc)
        num_nodes = at.num_nodes()
        return {
            "id":          self.id,
            "parent":      self.parent.id,
            "domains":     self.domains,
            #"at_enc":      codecs.encode(self.at_enc, encoding="base64").decode("ascii"),
            "num_nodes":   num_nodes,
            "status":      str(self.status),
            "verify_time": self.verify_time,
            "children":    [c.to_obj() for c in self.children]
        }

    def to_json(self):
        return json.dumps(self.to_obj())


class DistributedVerifier:

    class TaskType(Enum):
        SPLIT = 1
        VERIFY = 2

    def __init__(self, client, addtree, verifier_factory):
        self._timeout_start = 60
        self._timeout_max = 3600
        self._timeout_rate = 1.5

        self._client = client # dask client
        self._addtree = addtree # SearchSpace
        self._verifier_factory = verifier_factory # (domains, addtree) -> Verifier
        self._branch_factor = sum(client.nthreads().values()) * 4
        self._reachability = None

    def run(self):
        at_enc = DistributedVerifier._enc_at(self._addtree)
        splittree = SplitTree([], at_enc)

        f0 = self._client.submit(DistributedVerifier._split_fun, None, at_enc,
                self._branch_factor)
        f0.meta = Meta()
        f0.meta.splittree = splittree
        f0.meta.timeout = self._timeout_start
        fs = [f0]

        # Compute reachabilities while trees are being split
        f1 = self._client.submit(DistributedVerifier._reachability_fun,
                self._verifier_factory, at_enc)
        self._reachability = f1.result() # blocks

        while len(fs) > 0: # while task are running...
            wait(fs, return_when="FIRST_COMPLETED")
            next_fs = []
            for f in fs:
                if f.done(): next_fs += self._handle_done_future(f)
                else:        next_fs.append(f)
            fs = next_fs

        print("done!")
        print(splittree)
        print(splittree.to_json())

    def _handle_done_future(self, future):
        assert future.done()
        result = future.result()
        if result[0] == DistributedVerifier.TaskType.SPLIT:
            subprobs = result[1]
            return self._handle_split_result(future.meta, subprobs)
        if result[0] == DistributedVerifier.TaskType.VERIFY:
            status, doms, at_enc, verify_time = result[1:]
            return self._handle_verify_result(future.meta, status, doms, at_enc, verify_time)
        raise RuntimeError("unhandled future")

    def _handle_split_result(self, meta, subprobs):
        print("SPLIT HANDLER with", len(subprobs), "subprobs")
        timeout = meta.timeout
        fs = []
        children = meta.splittree.split(subprobs)
        for subsplittree, (doms, at_enc) in zip(children, subprobs):
            f = self._client.submit(DistributedVerifier._verify_fun,
                    self._verifier_factory, doms, at_enc,
                    self._reachability, timeout)
            f.meta = Meta()
            f.meta.timeout = meta.timeout
            f.meta.splittree = subsplittree
            fs.append(f)
        return fs

    def _handle_verify_result(self, meta, status, doms, at_enc, verify_time):
        print("VERIFY HANDLER", status)
        meta.splittree.status = status
        meta.splittree.verify_time = verify_time
        meta.splittree.timeout = meta.timeout
        if status == Verifier.Result.UNKNOWN:
            assert doms is not None
            assert at_enc is not None
            f = self._client.submit(DistributedVerifier._split_fun,
                    doms, at_enc, self._branch_factor)
            f.timeout = min(self._timeout_max, meta.timeout * self._timeout_rate)
            f.splittree = meta.splittree
            return [f]
        return []

    @staticmethod
    def _reachability_fun(vfactory, at_enc):
        at = DistributedVerifier._dec_at(at_enc)
        num_features = at.num_features()
        domains = [RealDomain() for i in range(num_features)]
        v = vfactory(domains, at)
        assert isinstance(v._strategy, SplitCheckStrategy) # only support this for now
        v._strategy.verify_setup()
        return v._strategy._reachability

    @staticmethod
    def _split_fun(domains, at_enc, branch_factor):
        at = DistributedVerifier._dec_at(at_enc)
        if domains is None: sp = SearchSpace(at)
        else:               sp = SearchSpace(at, domains)

        sp.split(branch_factor)

        subprobs = []
        for leaf_id in sp.leafs():
            sub_doms = sp.get_domains(leaf_id)
            sub_at = sp.get_pruned_addtree(leaf_id)
            sub_at_enc = DistributedVerifier._enc_at(sub_at)
            subprobs.append((sub_doms, sub_at_enc))

        return DistributedVerifier.TaskType.SPLIT, subprobs

    @staticmethod
    def _verify_fun(vfactory, domains, at_enc, reachability, timeout):
        at = DistributedVerifier._dec_at(at_enc)
        v = vfactory(domains, at)
        assert isinstance(v._strategy, SplitCheckStrategy) # only support this for now
        v._strategy.set_reachability(reachability)
        v.set_timeout(timeout)
        try:
            status = v.verify() # maybe also return logs, stats
        except VerifierTimeout as e:
            status = Verifier.Result.UNKNOWN
            print(f"timeout after ", e.unk_after, " (timeout =", timeout, ")")
        return DistributedVerifier.TaskType.VERIFY, status, domains, at_enc, v.verify_time

    @staticmethod
    def _enc_at(at):
        b = bytes(at.to_json(), encoding="ascii")
        return codecs.encode(b, encoding="zlib")

    @staticmethod
    def _dec_at(b):
        at_json = codecs.decode(b, encoding="zlib").decode("ascii")
        return AddTree.from_json(at_json)
