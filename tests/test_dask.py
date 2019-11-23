import unittest, z3, dask
from dask.distributed import Client
from treeck import *

def get_client():
    client = Client(threads_per_worker=1,
                    n_workers=4,
                    memory_limit='2GB')
    return client

def get_sp():
    at = AddTree.read("tests/models/xgb-calhouse-intermediate.json")
    #at = AddTree.read("tests/models/xgb-calhouse-easy.json")
    sp = SearchSpace(at)

    return sp

@dask.delayed
def solve(num_feat, domains, addtree_json):
    addtree = AddTree.from_json(addtree_json)
    solver = Z3Solver(num_feat, domains, addtree)
    return solver.verify(threshold=-3, op=LESS_THAN)


def tdask(client, num_leafs):
    sp = get_sp()
    sp.split(num_leafs)
    leafs = sp.leafs()

    num_feats = sp.num_features()

    tasks = []
    for leaf in leafs:
        addtree = sp.get_pruned_addtree(leaf).to_json()
        domains = sp.get_domains(leaf)

        print(f"Domains for leaf {leaf}:", list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

        tasks.append(solve(num_feats, domains, addtree))

    print(tasks)

    return dask.compute(tasks)



















#class TestDask(unittest.TestCase):
#    def test_dask1(self):
#        pass

if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)

    #tdask(client)
