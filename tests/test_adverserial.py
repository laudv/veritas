import matplotlib.pyplot as plt
import numpy as np
import unittest
import z3
import json

from treeck import *

def test_adverserial_mnist(testcase, model, instance_key, offset=10, nleafs=10,
        max_sum_offset = None,
        threshold_op=LESS_THAN, show_plot=False):

    with open("tests/models/mnist-instances.json") as f:
        instance = np.array(json.load(f)[str(instance_key)])

    at = AddTree.read(f"tests/models/{model}.json")
    sp = SearchSpace(at)
    sp.split(nleafs)
    results = []

    for i, leaf in enumerate(sp.leafs()):
        addtree = sp.get_pruned_addtree(leaf)
        domains = sp.get_domains(leaf)

        solver = Z3Solver(domains, addtree)
        constraints = []
        sum_constraint = 0
        for i, pixel in enumerate(instance):
            try:
                x = solver.xvar(i)
            except:
                break
            constraints.append(x > max(0, pixel-offset))
            constraints.append(x < min(255, pixel+offset))
            sum_constraint += z3.If(x-pixel <= 0, pixel-x, x-pixel)
        if max_sum_offset is not None:
            constraints.append(sum_constraint < max_sum_offset)
        check = solver.verify(constraints, threshold=0.0, op=threshold_op)
        results.append(check)

        #print(addtree)
        print(f"Domains for {check} leaf {i}({leaf}):",
                list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

        if check == z3.sat:
            m = solver.model()
            xs, ws = m["xs"], m["ws"]
            adverserial = instance.copy()
            for k, x in enumerate(xs):
                if x is not None:
                    #print(f"leaf {i}: pixel {k}: {adverserial[k]} -> {x} ({adverserial[k]-x})")
                    adverserial[k] = x

            print(f"real prediction (={instance_key}):", addtree.predict_single(instance))
            print(f"adverserial prediction (!={instance_key}):", addtree.predict_single(adverserial))
            print("ws:", sum(ws))
            print("absolute distance:", sum(abs(adverserial-instance)))

            if show_plot:
                fig, ax = plt.subplots(1, 3)
                im0 = ax[0].imshow(instance.reshape((28, 28)), vmin=0, vmax=255)
                ax[0].set_title("real")
                im1 = ax[1].imshow(adverserial.reshape((28, 28)), vmin=0, vmax=255)
                ax[1].set_title("adverserial")
                im2 = ax[2].imshow((instance-adverserial).reshape((28, 28)))
                ax[2].set_title("difference")
                fig.colorbar(im0, ax=ax[0])
                fig.colorbar(im1, ax=ax[1])
                fig.colorbar(im2, ax=ax[2])
                plt.show()

    print(results)
    testcase.assertTrue(sum(map(lambda x: x==z3.sat, results)) > 0)


class TestSearchSpace(unittest.TestCase):

    def test_mnist17(self):
        # GREATER_THAN ==> classify as 1, LESS_THAN ==> classify as not 1
        test_adverserial_mnist(self, "xgb-mnist-yis1-easy", 1,
                offset=5, nleafs=20, max_sum_offset=500,
                threshold_op=LESS_THAN, show_plot=True)

    def test_mnist08(self):
        # GREATER_THAN ==> classify as 0, LESS_THAN ==> classify as not 0
        test_adverserial_mnist(self, "xgb-mnist-yis0-easy", 0,
                offset=10, nleafs=20, max_sum_offset=500,
                threshold_op=LESS_THAN, show_plot=True)

if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)

    unittest.main()