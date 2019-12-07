import os, socket
import dask
from dask.distributed import wait, as_completed

from .pytreeck import SearchSpace, AddTree

class DistributedVerifier:

    def __init__(self, client, addtree, verifier_factory):
        self.client = client # dask client
        self.addtree = addtree # SearchSpace
        self._verifier_factory = verifier_factory # (domains, addtree) -> Verifier

    def run(self):
        self._step1()

    def _step1(self):
        """
        Construct a few complex trees for each worker.
        """
        nleafs = sum(self.client.nthreads().values()) * 4
        print("NLEAFS=", nleafs)

        sp = SearchSpace(self.addtree)
        sp.split(nleafs)

        futures = []
        for leaf_id in sp.leafs():
            addtree = sp.get_pruned_addtree(leaf_id).to_json()
            domains = sp.get_domains(leaf_id)
            future = self.client.submit(DistributedVerifier._solve,
                    self._verifier_factory,
                    leaf_id, domains, addtree, 60)
            futures.append(future)

        for future in as_completed(futures):
            y = future.result()
            print("RESULT as_completed", y, socket.gethostname(), os.getpid(), "done")

    def _solve(vfactory, leaf_id, domains, addtree_json, timeout):
        addtree = AddTree.from_json(addtree_json)
        v = vfactory(domains, addtree)
        v.set_timeout(timeout)
        print("STARTING solving", leaf_id, socket.gethostname(), os.getpid())
        status = v.verify()
        return leaf_id, status
