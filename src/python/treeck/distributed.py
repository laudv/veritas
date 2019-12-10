import os, socket, codecs, random, time
from enum import Enum
from dask.distributed import wait, get_client

from .pytreeck import SearchSpace, AddTree
from .verifier import Verifier

class DistributedVerifier:

    class TaskType(Enum):
        SPLIT = 1
        VERIFY = 2

    def __init__(self, client, addtree, verifier_factory):
        self._client = client # dask client
        self._addtree = addtree # SearchSpace
        self._verifier_factory = verifier_factory # (domains, addtree) -> Verifier
        self._branch_factor = sum(client.nthreads().values()) * 4

    def run(self):
        at_enc = DistributedVerifier._enc_at(self._addtree)
        f = self._client.submit(DistributedVerifier._split_fun,
                None, at_enc, self._branch_factor)
        fs = [f]

        while len(fs) > 0: # while task are running...
            wait(fs, return_when="FIRST_COMPLETED")

            next_fs = []
            for f in fs:
                if f.done():
                    next_fs += self._handle_done_future(f)
                else:
                    next_fs.append(f)
            fs = next_fs

        print("done!")

    def _handle_done_future(self, future):
        assert future.done()
        result = future.result()
        if result[0] == DistributedVerifier.TaskType.SPLIT:
            subprobs = result[1]
            return self._handle_split_result(subprobs)
        if result[0] == DistributedVerifier.TaskType.VERIFY:
            status, doms, at_enc = result[1:]
            return self._handle_verify_result(status, doms, at_enc)

    def _handle_split_result(self, subprobs):
        print("SPLIT HANDLER with", len(subprobs), "subprobs")
        timeout = 10
        fs = []
        for doms, at_enc in subprobs:
            f = self._client.submit(DistributedVerifier._verify_fun,
                    self._verifier_factory, doms, at_enc, timeout)
            fs.append(f)
        return fs

    def _handle_verify_result(self, status, doms, at_enc):
        print("VERIFY HANDLER", status)
        if status == Verifier.Result.UNKNOWN:
            assert doms is not None
            assert at_enc is not None
            f = self._client.submit(DistributedVerifier._split_fun,
                    doms, at_enc, self._branch_factor)
            return [f]
        return []

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

    def _verify_fun(vfactory, domains, at_enc, timeout):
        at = DistributedVerifier._dec_at(at_enc)
        v = vfactory(domains, at)
        v.set_timeout(timeout)
        status = v.verify() # maybe also return logs, stats
        return DistributedVerifier.TaskType.VERIFY, status, domains, at_enc

    def _enc_at(at):
        b = bytes(at.to_json(), encoding="ascii")
        return codecs.encode(b, encoding="zlib")

    def _dec_at(b):
        at_json = codecs.decode(b, encoding="zlib").decode("ascii")
        return AddTree.from_json(at_json)
