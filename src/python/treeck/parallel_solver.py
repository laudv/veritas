import dask

class ParallelSolver:

    def __init__(self, client, solver):
        self._client = client # dask client
        self._trees = solver.trees() # pytrees
        self._solver = solver # z3solver

    def verify(self):
        tasks = [dask.delayed(self._client.test_tree_reachability(i)) \
                    for i in range(len(self._trees))]
        results = dask.compute(tasks)
        print(results)
        print(self._trees._reachable)


        


