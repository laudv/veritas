import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()
