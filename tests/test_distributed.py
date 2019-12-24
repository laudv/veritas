#import matplotlib.pyplot as plt
import unittest, json
import numpy as np
import z3
import importlib

import treeck
from treeck import *
from treeck.verifier import Verifier
from treeck.z3backend import Z3Backend as Backend
from treeck.distributed import DistributedVerifier

from dask.distributed import Client
from start_dask import start_local

class TestDistributedVerifier(unittest.TestCase):
    def test_img1(self):
        def vfactory(at, leaf):
            v = Verifier(at, leaf, Backend())
            v.add_constraint(v.fvar() < 0.0)
            v.add_constraint(v.xvar(0) > 50)
            v.add_constraint(v.xvar(0) <= 80)
            return v

        with Client("tcp://localhost:30333") as client:
        #with start_local() as client:
            client.run(importlib.reload, treeck)
            at = AddTree.read("tests/models/xgb-img-easy.json")
            st = SplitTree(at, {})
            dv = DistributedVerifier(client, st, vfactory)

            l0 = dv._st.get_leaf(0)

            def test_reachable(m, l0):
                m(l0.is_reachable(9, 109))
                m(l0.is_reachable(9, 117))
                m(l0.is_reachable(9, 95))
                m(l0.is_reachable(9, 104))
                m(l0.is_reachable(9, 83))
                m(l0.is_reachable(9, 70))
                m(l0.is_reachable(9, 52))
                m(l0.is_reachable(9, 37))
                m(l0.is_reachable(9, 5))
                m(l0.is_reachable(9, 29))
                m(l0.is_reachable(8, 96))
                m(l0.is_reachable(8, 3))
                m(l0.is_reachable(7, 23))
                m(l0.is_reachable(7, 15))
                m(l0.is_reachable(6, 118))
                m(l0.is_reachable(6, 98))
                m(l0.is_reachable(6, 37))
                m(l0.is_reachable(6, 5))
                m(l0.is_reachable(6, 29))
                m(l0.is_reachable(5, 112))
                m(l0.is_reachable(5, 90))
                m(l0.is_reachable(5, 66))
                m(l0.is_reachable(5, 3))
                m(l0.is_reachable(5, 37))
                m(l0.is_reachable(4, 84))
                m(l0.is_reachable(4, 70))
                m(l0.is_reachable(4, 53))
                m(l0.is_reachable(4, 37))
                m(l0.is_reachable(4, 5))
                m(l0.is_reachable(4, 29))
                m(l0.is_reachable(4, 23))
                m(l0.is_reachable(3, 108))
                m(l0.is_reachable(3, 92))
                m(l0.is_reachable(3, 84))
                m(l0.is_reachable(3, 78))
                m(l0.is_reachable(3, 3))
                m(l0.is_reachable(3, 49))
                m(l0.is_reachable(3, 43))
                m(l0.is_reachable(3, 37))
                m(l0.is_reachable(2, 96))
                m(l0.is_reachable(2, 80))
                m(l0.is_reachable(2, 66))
                m(l0.is_reachable(2, 3))
                m(l0.is_reachable(2, 49))
                m(l0.is_reachable(2, 37))
                m(l0.is_reachable(1, 94))
                m(l0.is_reachable(1, 82))
                m(l0.is_reachable(1, 66))
                m(l0.is_reachable(1, 3))
                m(l0.is_reachable(1, 35))
                m(l0.is_reachable(1, 23))
                m(l0.is_reachable(0, 98))
                m(l0.is_reachable(0, 86))
                m(l0.is_reachable(0, 29))
                m(l0.is_reachable(0, 5))

            test_reachable(self.assertTrue, l0)
            l0 = dv._check_paths(l0)
            test_reachable(self.assertFalse, l0)


            #client.shutdown()

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
