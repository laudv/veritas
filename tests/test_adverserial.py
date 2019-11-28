import matplotlib.pyplot as plt
import numpy as np
import unittest
import z3
import json

from treeck import *

class TestSearchSpace(unittest.TestCase):

    def test_mnist17(self):
        with open("tests/models/mnist-instances.json") as f:
            instances = json.load(f)

        at = AddTree.read("tests/models/xgb-mnist17-easy.json")
        sp = SearchSpace(at)
        sp.split(10)
        instance = np.array(instances["7"])
        offset = 10
        results = []

        for i, leaf in enumerate(sp.leafs()):
            addtree = sp.get_pruned_addtree(leaf)
            domains = sp.get_domains(leaf)

            solver = Z3Solver(domains, addtree)
            constraints = []
            for i, pixel in enumerate(instance):
                try:
                    x = solver.xvar(i)
                except:
                    break
                constraints.append(x > max(0, pixel-offset))
                constraints.append(x < min(255, pixel+offset))
            check = solver.verify(constraints, threshold=0.0, op=GREATER_THAN)
            results.append(check)

            if check == z3.sat:
                m = solver.model()
                xs, ws = m["xs"], m["ws"]
                adverserial = instance.copy()
                for k, x in enumerate(xs):
                    if x is not None:
                        #print(f"leaf {i}: pixel {k}: {adverserial[k]} -> {x} ({adverserial[k]-x})")
                        adverserial[k] = x
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

                print("real prediction:", addtree.predict_single(instance))
                print("adverserial prediction:", addtree.predict_single(adverserial))
                print("ws:", sum(ws))

                plt.show()

            #print(addtree)
            print(f"Domains for {check} leaf {i}({leaf}):",
                    list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

        print(results)

if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)

    #plot_pruned_trees()
    #test_z3()
    unittest.main()
