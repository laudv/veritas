import dask

class ParallelSolver:

    def __init__(self, client, sp):
        self._client = client # dask client
        self._sp = sp # SearchSpace

    def verify(self):
        pass
        #tasks = [dask.delayed(self._solver.test_tree_reachability(i)) \
        #            for i in range(len(self._trees))]
        #results = dask.compute(tasks)
        #print(results)
        #print(self._trees._reachable)


        


