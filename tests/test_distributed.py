#import matplotlib.pyplot as plt
import unittest, json
import numpy as np
import z3
import importlib

import treeck
from treeck import *
from treeck.verifier import Verifier
from treeck.z3backend import Z3Backend as Backend
from treeck.distributed import DistributedVerifier, VerifierFactory

from dask.distributed import Client
from start_dask import start_local

dask_scheduler = "localhost:8786"

class TestDistributedVerifier(unittest.TestCase):
    def test_img_check_paths(self):
        class VFactory(VerifierFactory):
            def __call__(self, lk):
                v = Verifier(lk, Backend())
                v.add_constraint(v.fvar() < 0.0)
                v.add_constraint(v.xvar(0) > 50)
                v.add_constraint(v.xvar(0) <= 80)
                return v

        print("hey")

        with Client(dask_scheduler) as client:
            client.restart()
            at = AddTree.read("tests/models/xgb-img-easy.json")
            dt = DomTree(at, {})
            dv = DistributedVerifier(client, dt, VFactory())

            def test_reachable(m, l0):
                m(l0.is_reachable(0, 9, 109))
                m(l0.is_reachable(0, 9, 117))
                m(l0.is_reachable(0, 9, 95))
                m(l0.is_reachable(0, 9, 104))
                m(l0.is_reachable(0, 9, 83))
                m(l0.is_reachable(0, 9, 70))
                m(l0.is_reachable(0, 9, 52))
                m(l0.is_reachable(0, 9, 37))
                m(l0.is_reachable(0, 9, 5))
                m(l0.is_reachable(0, 9, 29))
                m(l0.is_reachable(0, 8, 96))
                m(l0.is_reachable(0, 8, 3))
                m(l0.is_reachable(0, 7, 23))
                m(l0.is_reachable(0, 7, 15))
                m(l0.is_reachable(0, 6, 118))
                m(l0.is_reachable(0, 6, 98))
                m(l0.is_reachable(0, 6, 37))
                m(l0.is_reachable(0, 6, 5))
                m(l0.is_reachable(0, 6, 29))
                m(l0.is_reachable(0, 5, 112))
                m(l0.is_reachable(0, 5, 90))
                m(l0.is_reachable(0, 5, 66))
                m(l0.is_reachable(0, 5, 3))
                m(l0.is_reachable(0, 5, 37))
                m(l0.is_reachable(0, 4, 84))
                m(l0.is_reachable(0, 4, 70))
                m(l0.is_reachable(0, 4, 53))
                m(l0.is_reachable(0, 4, 37))
                m(l0.is_reachable(0, 4, 5))
                m(l0.is_reachable(0, 4, 29))
                m(l0.is_reachable(0, 4, 23))
                m(l0.is_reachable(0, 3, 108))
                m(l0.is_reachable(0, 3, 92))
                m(l0.is_reachable(0, 3, 84))
                m(l0.is_reachable(0, 3, 78))
                m(l0.is_reachable(0, 3, 3))
                m(l0.is_reachable(0, 3, 49))
                m(l0.is_reachable(0, 3, 43))
                m(l0.is_reachable(0, 3, 37))
                m(l0.is_reachable(0, 2, 96))
                m(l0.is_reachable(0, 2, 80))
                m(l0.is_reachable(0, 2, 66))
                m(l0.is_reachable(0, 2, 3))
                m(l0.is_reachable(0, 2, 49))
                m(l0.is_reachable(0, 2, 37))
                m(l0.is_reachable(0, 1, 94))
                m(l0.is_reachable(0, 1, 82))
                m(l0.is_reachable(0, 1, 66))
                m(l0.is_reachable(0, 1, 3))
                m(l0.is_reachable(0, 1, 35))
                m(l0.is_reachable(0, 1, 23))
                m(l0.is_reachable(0, 0, 98))
                m(l0.is_reachable(0, 0, 86))
                m(l0.is_reachable(0, 0, 29))
                m(l0.is_reachable(0, 0, 5))

            def test(addtree):
                return str(type(addtree))

            l0 = dt.get_leaf(0)
            test_reachable(self.assertTrue, l0)
            l0 = dv._check_paths(l0)
            test_reachable(self.assertFalse, l0)

    def test_img_generate_splits(self):
        class VFactory(VerifierFactory):
            def __call__(self, addtrees, ls):
                v = Verifier(addtrees, ls, Backend())
                v.add_constraint(v.fvar() < 0.0)
                v.add_constraint(v.xvar(0) > 50)
                v.add_constraint(v.xvar(1) < 50)
                return v

        with Client(dask_scheduler) as client:
            client.restart()
            nworkers = sum(client.nthreads().values())
            N = 10
            at = AddTree.read("tests/models/xgb-img-easy.json")
            sb = Subspaces(at, {})
            dv = DistributedVerifier(client, sb, VFactory(),
                    check_paths = True,
                    num_initial_tasks = N,
                    stop_when_sat = False)

            dv.check()
            #print(json.dumps(dv.results, indent=2, default=str))
            count_with_status = 0
            count_with_sat = 0
            for split_id, d in dv.results.items():
                if isinstance(split_id, int) and "status" in d:
                    count_with_status += 1
                    if d["status"].is_sat():
                        count_with_sat += 1

            self.assertEqual(count_with_status, N)
            self.assertGreater(count_with_sat, 0)

    def test_bin_mnist(self):
        class VFactory(VerifierFactory):
            def __call__(self, addtrees, ls):
                v = Verifier(addtrees, ls, Backend())
                ftypes = v.instance(0)._feat_types
                v.add_constraint(v.fvar() > 5.0)
                v.add_constraint(z3.PbLe(
                    [(v.xvar(fid).get(), 1) for fid in ftypes.feat_ids()], 25))
                return v

        with Client(dask_scheduler) as client:
            client.restart()
            nworkers = sum(client.nthreads().values())
            N = 10
            at = AddTree.read("tests/models/xgb-mnist-bin-yis1-intermediate.json")
            sb = Subspaces(at, {})
            dv = DistributedVerifier(client, sb, VFactory(),
                    check_paths = False,
                    num_initial_tasks = N,
                    timeout_start = 5.0,
                    stop_when_sat = False)

            dv.check()
            #print(json.dumps(dv.results, indent=2, default=repr))
            print(sb.domtree())
            count_with_status = 0
            count_with_sat = 0
            for split_id, d in dv.results.items():
                if isinstance(split_id, int) and "status" in d:
                    count_with_status += 1
                    if d["status"] == Verifier.Result.SAT:
                        count_with_sat += 1

            self.assertGreaterEqual(count_with_status, N)
            self.assertGreater(count_with_sat, 0)

    def test_img_multi_instance(self):
        class VFactory(VerifierFactory):
            def __call__(self, addtrees, ls):
                v = Verifier(addtrees, ls, Backend())
                v.add_constraint(v.fvar(instance=0) < 0.0)
                v.add_constraint(v.xvar(0, instance=0) > 50)
                v.add_constraint(v.xvar(1, instance=0) < 50)

                # instances are exactly the same!
                for fid1, fid2 in zip(v.instance(0).feat_ids(), v.instance(1).feat_ids()):
                    v.add_constraint(v.xvar(fid1, instance=0) == v.xvar(fid2, instance=1))

                v.add_constraint(v.fvar(instance=1).get() - v.fvar(instance=0).get() < 99999)

                return v

        with Client(dask_scheduler) as client:
            client.restart()
            nworkers = sum(client.nthreads().values())
            N = 10
            at0 = AddTree.read("tests/models/xgb-img-easy.json")
            at1 = AddTree.read("tests/models/xgb-img-easy.json")
            at1.base_score = 100000
            sbs = [Subspaces(at0, {}), Subspaces(at1, {})]
            dv = DistributedVerifier(client, sbs, VFactory(),
                    check_paths = False,
                    num_initial_tasks = N,
                    stop_when_sat = False)

            dv.check()
            print(json.dumps(dv.results, indent=2, default=str))
            #count_with_status = 0
            #count_with_sat = 0
            #for split_id, d in dv.results.items():
            #    if isinstance(split_id, int) and "status" in d:
            #        count_with_status += 1
            #        if d["status"].is_sat():
            #            count_with_sat += 1

            #self.assertEqual(count_with_status, N)
            #self.assertEqual(count_with_sat, 0)


    #def test_adv(self):
    #    instance_key = 0
    #    model = "xgb-mnist-yis0-easy"
    #    offset = 10
    #    max_sum_offset = 50

    #    def vfactory_aux(instance, offset, max_sum_offset, doms, at):
    #        v = Verifier(doms, at, Backend(), Strategy())
    #        sum_constraint = 0
    #        for j, pixel in zip(range(v.num_features), instance):
    #            x = v.xvar(j)
    #            v.add_constraint((x > max(0, pixel-offset)) & (x < min(255, pixel+offset)))
    #            sum_constraint += z3.If(x.get()-pixel <= 0, pixel-x.get(), x.get()-pixel)
    #        v.add_constraint(sum_constraint < max_sum_offset)
    #        return v

    #    with open("tests/models/mnist-instances.json") as f:
    #        instance_key = 0
    #        instance = np.array(json.load(f)[str(instance_key)])

    #    vfactory = lambda doms, at: vfactory_aux(instance, offset, max_sum_offset, doms, at)
    #    with Client("tcp://localhost:30333") as client:
    #        at = AddTree.read(f"tests/models/{model}.json")
    #        dv = DistributedVerifier(client, at, vfactory)
    #        dv.run()


if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()
