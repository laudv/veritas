#import matplotlib.pyplot as plt
import unittest, json
import numpy as np
import z3
import importlib

import treeck
from treeck import *
from treeck.verifier import Verifier, SplitCheckStrategy as Strategy
from treeck.z3backend import Z3Backend as Backend
from treeck.distributed import DistributedVerifier

from dask.distributed import Client
from start_dask import start_local

class TestDistributedVerifier(unittest.TestCase):
    def test_img1(self):
        def vfactory(doms, at):
            v = Verifier(doms, at, Backend(), Strategy())
            v.add_constraint(v.fvar() < 0.0)
            v.add_constraint(v.xvar(0) > 50)
            return v

        with Client("tcp://localhost:30333") as client:
        #with start_local() as client:
            client.run(importlib.reload, treeck)
            at = AddTree.read("tests/models/xgb-img-easy.json")
            dv = DistributedVerifier(client, at, vfactory)
            dv.run()

            #client.shutdown()

    def test_adv(self):
        instance_key = 0
        model = "xgb-mnist-yis0-easy"
        offset = 10
        max_sum_offset = 50

        def vfactory_aux(instance, offset, max_sum_offset, doms, at):
            v = Verifier(doms, at, Backend(), Strategy())
            sum_constraint = 0
            for j, pixel in zip(range(v.num_features), instance):
                x = v.xvar(j)
                v.add_constraint((x > max(0, pixel-offset)) & (x < min(255, pixel+offset)))
                sum_constraint += z3.If(x.get()-pixel <= 0, pixel-x.get(), x.get()-pixel)
            v.add_constraint(sum_constraint < max_sum_offset)
            return v

        with open("tests/models/mnist-instances.json") as f:
            instance_key = 0
            instance = np.array(json.load(f)[str(instance_key)])

        vfactory = lambda doms, at: vfactory_aux(instance, offset, max_sum_offset, doms, at)
        with Client("tcp://localhost:30333") as client:
            at = AddTree.read(f"tests/models/{model}.json")
            dv = DistributedVerifier(client, at, vfactory)
            dv.run()


if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()
