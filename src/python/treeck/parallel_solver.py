import dask

class ParallelSolver:

    def __init__(self, client, addtree):
        self.client = client # dask client
        self.addtree = addtree # SearchSpace

    def verify(self):
        pass
        #tasks = [dask.delayed(self._solver.test_tree_reachability(i)) \
        #            for i in range(len(self._trees))]
        #results = dask.compute(tasks)
        #print(results)
        #print(self._trees._reachable)

    def _step1(self):
        """
        Construct a few complex trees for each worker.
        """
        nthreads = client.nthreads()
        nleafs = nthreads

        sp = SearchSpace(self.addtree)
        sp.split(nleafs)
        num_features = sp.num_features()

        tasks = []
        for leaf in sp.leafs():
            addtree = sp.get_pruned_addtree(leaf).to_json()
            domains = sp.get_domains(leaf)
            tasks.append((domains, addtree))




        


