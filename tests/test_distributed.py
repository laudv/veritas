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

    def test_img_generate_splits(self):
        class VFactory(VerifierFactory):
            def __call__(self, lk):
                v = Verifier(lk, Backend())
                v.add_constraint(v.fvar() < 0.0)
                v.add_constraint(v.xvar(0) > 50)
                v.add_constraint(v.xvar(1) < 50)
                return v

        with Client(dask_scheduler) as client:
            client.restart()
            N = 10
            at = AddTree.read("tests/models/xgb-img-easy.json")
            dt = DomTree(at, {})
            dv = DistributedVerifier(client, dt, VFactory(),
                    check_paths = True,
                    num_initial_tasks = N,
                    stop_when_num_sats = N)

            dv.check()
            #print(json.dumps(dv.results, indent=2, default=str))
            count_with_status = 0
            count_with_sat = 0
            for k, d in dv.results.items():
                if isinstance(k, int) and "status" in d:
                    count_with_status += 1
                    if d["status"].is_sat():
                        count_with_sat += 1

            self.assertEqual(count_with_status, N)
            self.assertGreater(count_with_sat, 0)

    def test_bin_mnist(self):
        class VFactory(VerifierFactory):
            def __call__(self, lk):
                v = Verifier(lk, Backend())
                v.add_constraint(v.fvar() > 5.0)
                v.add_constraint(z3.PbLe([(v.xvar(fid).get(), 1)
                    for fid in v.instance(0).feat_ids()], 50))
                return v

        with Client(dask_scheduler) as client:
            client.restart()
            N = 10
            at = AddTree.read("tests/models/xgb-mnist-bin-yis1-intermediate.json")
            dt = DomTree(at, {})
            dv = DistributedVerifier(client, dt, VFactory(),
                    check_paths = False,
                    num_initial_tasks = N,
                    timeout_start = 5.0,
                    stop_when_num_sats = N)

            dv.check()
            #print(json.dumps(dv.results, indent=2, default=repr))
            #print(dt.tree())
            count_with_status = 0
            count_with_sat = 0
            for k, d in dv.results.items():
                if isinstance(k, int) and "status" in d:
                    count_with_status += 1
                    if d["status"] == Verifier.Result.SAT:
                        count_with_sat += 1
                        self.assertGreater(d["model"]["f"], 5.0)
                        inst = [True] * (28*28)
                        for i, x in d["model"]["xs"].items():
                            inst[i] = x
                        self.assertEqual(d["model"]["f"], at.predict_single(inst))
                        #print(d["model"]["f"], at.predict_single(inst))

            self.assertGreaterEqual(count_with_status, N)
            self.assertGreater(count_with_sat, 0)

    def test_img_multi_instance(self):
        class VFactory(VerifierFactory):
            def __call__(self, lk):
                v = Verifier(lk, Backend())
                v.add_constraint(v.fvar(instance=0) < 0.0)
                v.add_constraint(v.xvar(0, instance=0) > 50)
                v.add_constraint(v.xvar(1, instance=0) < 50)

                # instances are exactly the same!
                for fid1, fid2 in zip(v.instance(0).feat_ids(), v.instance(1).feat_ids()):
                    v.add_constraint(v.xvar(fid1, instance=0) == v.xvar(fid2, instance=1))

                v.add_constraint(v.fvar(instance=1).get() - v.fvar(instance=0).get() < 9999)

                return v

        with Client(dask_scheduler) as client:
            client.restart()
            N = 10
            at0 = AddTree.read("tests/models/xgb-img-easy.json")
            at1 = AddTree.read("tests/models/xgb-img-easy.json")
            at1.base_score = 10000
            dt = DomTree([(at0, {}), (at1, {})])
            dv = DistributedVerifier(client, dt, VFactory(),
                    check_paths = True,
                    num_initial_tasks = N,
                    stop_when_num_sats = N)

            dv.check()
            #print(json.dumps(dv.results, indent=2, default=str))
            #print(dt.tree())
            count_with_status = 0
            count_with_sat = 0
            for k, d in dv.results.items():
                if isinstance(k, int) and "status" in d:
                    count_with_status += 1
                    if d["status"].is_sat():
                        count_with_sat += 1

            self.assertEqual(count_with_status, N)
            self.assertEqual(count_with_sat, 0)


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
