import os, socket
import dask
from dask.distributed import wait, as_completed

from .pytreeck import SearchSpace, AddTree
from .z3solver import Z3Solver

class ParallelSolver:

    def __init__(self, client, addtree):
        self.client = client # dask client
        self.addtree = addtree # SearchSpace
        self._constraintsf = lambda solver: []

    def verify(self, constraintsf, threshold, op):
        self._constraintsf = constraintsf # (Z3Solver -> [Z3Constraints])
        self._threshold = threshold
        self._threshold_op = op
        self._step1()

    def _step1(self):
        """
        Construct a few complex trees for each worker.
        """
        nleafs = sum(self.client.nthreads().values())
        print("NLEAFS=", nleafs)

        sp = SearchSpace(self.addtree)
        sp.split(nleafs)

        futures = []
        for leaf_id in sp.leafs():
            addtree = sp.get_pruned_addtree(leaf_id).to_json()
            domains = sp.get_domains(leaf_id)
            future = self.client.submit(ParallelSolver._solve,
                    leaf_id,
                    domains, addtree,
                    self._constraintsf,
                    self._threshold,
                    self._threshold_op,
                    100)
            futures.append(future)

        for future in as_completed(futures):
            y = future.result()
            print("RESULT as_completed", y, socket.gethostname(), os.getpid(), "done")

    @staticmethod
    def _solve(leaf_id, domains, addtree_json, constraintsf, threshold, op, timeout):
        addtree = AddTree.from_json(addtree_json)
        solver = Z3Solver(domains, addtree)
        solver.set_timeout(timeout)
        constraints = constraintsf(solver)
        print("STARTING solving", socket.gethostname(), os.getpid())
        return leaf_id, solver.verify(constraints, threshold=threshold, op=op)
