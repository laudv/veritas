import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import unittest, time, json
import z3

from treeck import *
from treeck.plot import TreePlot
from treeck.verifier import Verifier, DefaultVerifier
from treeck.z3backend import Z3Backend as Backend

def plot_pruned_trees():
    at = AddTree.read("tests/models/xgb-covtype-easy.json")
    sp = SearchSpace(at)

    sp.split(1000)

    leafs = sp.leafs()
    print(sp.scores())

    leaf_index_to_plot = 330

    for k, leaf in enumerate(leafs):
        at_pruned = sp.get_pruned_addtree(leaf)
        domains = sp.get_domains(leaf)

        print("{:4}: num_nodes for leaf {:<5}: {} vs {}".format(k, leaf, at.num_nodes(), at_pruned.num_nodes()))

        if k == leaf_index_to_plot:
            for i in range(len(at)):
                p = TreePlot()
                p.add_domains(domains)
                p.add_tree_cmp(at[i], at_pruned[i])
                #p.add_tree(at[i])
                p.add_tree(at_pruned[i])
                p.render("plots/plot-leaf{}-{}".format(k, i))



class TestSearchSpace(unittest.TestCase):
    #def test_single_tree(self):
    #    at = AddTree()
    #    at.base_score = 10
    #    t = at.add_tree();
    #    t.split(t.root(), 0, 2)
    #    t.split( t.left(t.root()), 0, 1)
    #    t.split(t.right(t.root()), 0, 3)
    #    t.set_leaf_value( t.left( t.left(t.root())), 0.1)
    #    t.set_leaf_value(t.right( t.left(t.root())), 0.2)
    #    t.set_leaf_value( t.left(t.right(t.root())), 0.3)
    #    t.set_leaf_value(t.right(t.right(t.root())), 0.4)

    #    print(at)
    #    sp = SearchSpace(at)
    #    sp.split(2)
    #    leafs = sp.leafs()

    #    self.assertEqual(len(leafs), 2)

    #    atp = [sp.get_pruned_addtree(l) for l in leafs]
    #    dom = [sp.get_domains(l) for l in leafs]

    #    self.assertEqual(dom[0][0].hi, 2)
    #    self.assertEqual(dom[1][0].lo, 2)
    #    self.assertEqual(atp[0].base_score, at.base_score)
    #    self.assertEqual(atp[1].base_score, at.base_score)
    #    self.assertEqual(sp.scores(), [6])

    #    self.assertEqual(atp[0][0].get_split(0), (0, 1))
    #    self.assertEqual(atp[0][0].get_leaf_value(1), 0.1)
    #    self.assertEqual(atp[0][0].get_leaf_value(2), 0.2)
    #    self.assertEqual(atp[1][0].get_split(0), (0, 3))
    #    self.assertEqual(atp[1][0].get_leaf_value(1), 0.3)
    #    self.assertEqual(atp[1][0].get_leaf_value(2), 0.4)

    #    sp = SearchSpace(at)
    #    sp.split(100) # more than possible!
    #    leafs = sp.leafs()
    #    leaf_values = []

    #    for leaf in sp.leafs():
    #        at = sp.get_pruned_addtree(leaf)
    #        domains = sp.get_domains(leaf)

    #        #print(f"Domains for leaf {leaf}:",
    #        #        list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))
    #        #print(at)

    #        self.assertTrue(at[0].is_leaf(at[0].root()))
    #        leaf_values.append(at[0].get_leaf_value(at[0].root()))

    #    leaf_values.sort()
    #    self.assertEqual(leaf_values, [0.1, 0.2, 0.3, 0.4])

    #    self.assertEqual(len(leafs), 4)
    #    self.assertEqual(sp.scores(), [6, 2, 2])

    #def test_covtype(self):
    #    at = AddTree.read("tests/models/xgb-covtype-easy.json")
    #    sp = SearchSpace(at)

    #    sp.split(10)
    #    threshold = -7.3

    #    results = []

    #    for i, leaf in enumerate(sp.leafs()):
    #        addtree = sp.get_pruned_addtree(leaf)
    #        domains = sp.get_domains(leaf)

    #        #print(addtree)
    #        print(f"Domains for leaf {i}({leaf}):",
    #                list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

    #        dv = DefaultVerifier(domains, addtree, Backend())
    #        dv.add_constraint(dv.xvar(35) > 0.5)
    #        results.append(dv.verify(dv.fvar() < threshold))

    #    unsat, sat = Verifier.Result.UNSAT, Verifier.Result.SAT
    #    self.assertEqual(results, [unsat, unsat, unsat, unsat, sat, unsat, unsat, unsat, unsat, unsat])

    #def test_img1(self):
    #    with open("tests/models/xgb-img-easy-values.json") as f:
    #        ys = json.load(f)
    #    at = AddTree.read("tests/models/xgb-img-easy.json")
    #    img = np.array(ys).reshape((100, 100))
    #    m, M = min(ys), max(ys)
    #    argm = np.unravel_index(img.argmin(), img.shape)

    #    sp = SearchSpace(at)
    #    sp.split(20)
    #    results = []

    #    for i, leaf in enumerate(sp.leafs()):
    #        addtree = sp.get_pruned_addtree(leaf)
    #        domains = sp.get_domains(leaf)

    #        dv = DefaultVerifier(domains, addtree, Backend())
    #        check = dv.verify(dv.fvar() < m+1e-4)
    #        results.append(check)

    #        if check == Verifier.Result.SAT:
    #            sat_leaf = leaf
    #            sat_model = dv.model()

    #        #print(addtree)
    #        print(f"Domains for {check} leaf {i}({leaf}):",
    #                list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

    #    #fig, ax = plt.subplots(1)
    #    #ax.set_title("min")
    #    #im = ax.imshow(img)

    #    #for leaf in sp.leafs():
    #    #    doms = sp.get_domains(leaf)
    #    #    y0, y1 = max(0, doms[0].lo), min(99, doms[0].hi)
    #    #    x0, x1 = max(0, doms[1].lo), min(99, doms[1].hi)
    #    #    print(x0, y0, x1, y1, argm)
    #    #    color = "r" if leaf == sat_leaf else "b"
    #    #    zorder = 5 if leaf == sat_leaf else 1

    #    #    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1, edgecolor=color, facecolor='none', zorder=zorder)
    #    #    ax.add_patch(rect)

    #    #print("w=", sum(sat_model["ws"])+at.base_score, "m=", m, "img[*,*]=",
    #    #        img[int(np.floor(sat_model["xs"][0])), int(np.floor(sat_model["xs"][1]))])
    #    #ax.scatter([sat_model["xs"][1]], [sat_model["xs"][0]], marker="o", c="b")
    #    #ax.scatter([argm[1]], [argm[0]], marker="x", c="r")
    #    #ax.set_xlim([0, 99])
    #    #ax.set_ylim([99, 0])
    #    #fig.colorbar(im, ax=ax)
    #    #plt.show()

    #    self.assertEqual(sum(map(lambda x: x==Verifier.Result.SAT, results)), 1)

    #def test_img2(self):
    #    with open("tests/models/xgb-img-easy-values.json") as f:
    #        ys = json.load(f)
    #    at = AddTree.read("tests/models/xgb-img-easy.json")
    #    img = np.array(ys).reshape((100, 100))
    #    m, M = min(ys), max(ys)
    #    argM = np.unravel_index(img.argmax(), img.shape)

    #    sp = SearchSpace(at)
    #    sp.split(20)
    #    results = []

    #    for i, leaf in enumerate(sp.leafs()):
    #        addtree = sp.get_pruned_addtree(leaf)
    #        domains = sp.get_domains(leaf)

    #        dv = DefaultVerifier(domains, addtree, Backend())
    #        check = dv.verify(dv.fvar() > M-1e-4)
    #        results.append(check)

    #        if check == Verifier.Result.SAT:
    #            sat_leaf = leaf
    #            sat_model = dv.model()

    #        #print(addtree)
    #        print(f"Domains for {check} leaf {i}({leaf}):",
    #                list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

    #    #fig, ax = plt.subplots(1)
    #    #ax.set_title("max")
    #    #im = ax.imshow(img)

    #    #for leaf in sp.leafs():
    #    #    doms = sp.get_domains(leaf)
    #    #    y0, y1 = max(0, doms[0].lo), min(99, doms[0].hi)
    #    #    x0, x1 = max(0, doms[1].lo), min(99, doms[1].hi)
    #    #    print(x0, y0, x1, y1, argM)
    #    #    color = "r" if leaf == sat_leaf else "b"
    #    #    zorder = 5 if leaf == sat_leaf else 1

    #    #    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1, edgecolor=color, facecolor='none', zorder=zorder)
    #    #    ax.add_patch(rect)

    #    #print("w=", sum(sat_model["ws"])+at.base_score, "M=", M, "img[*,*]=",
    #    #        img[int(np.floor(sat_model["xs"][0])), int(np.floor(sat_model["xs"][1]))])
    #    #ax.scatter([sat_model["xs"][1]], [sat_model["xs"][0]], marker="o", c="b")
    #    #ax.scatter([argM[1]], [argM[0]], marker="x", c="r")
    #    #ax.set_xlim([0, 99])
    #    #ax.set_ylim([99, 0])
    #    #fig.colorbar(im, ax=ax)
    #    #plt.show()

    #    self.assertEqual(sum(map(lambda x: x==Verifier.Result.SAT, results)), 1)

    def test_img3(self):
        at = AddTree.read("tests/models/xgb-img-easy.json")
        sp = SearchSpace(at)
        sp.split(4)
        domains = [sp.get_domains(l) for l in sp.leafs()]
        scores = sp.scores()
        self.assertEqual(len(domains), 4)

        print(len(domains))

        sp0 = SearchSpace(at)
        sp0.split(2)
        domains0 = [sp0.get_domains(l) for l in sp0.leafs()]
        ats0 = [sp0.get_pruned_addtree(l) for l in sp0.leafs()]
        self.assertEqual(len(ats0), 2)

        sp1 = SearchSpace(ats0[0], domains[0])
        sp1.split(2)
        domains1 = [sp1.get_domains(l) for l in sp1.leafs()]
        self.assertEqual(len(domains1), 2)

        sp2 = SearchSpace(ats0[1], domains[1])
        sp2.split(2)
        domains2 = [sp2.get_domains(l) for l in sp2.leafs()]
        self.assertEqual(len(domains2), 2)

        domains12 = domains1 + domains2
        scores12 = sorted(sp0.scores() + sp1.scores() + sp2.scores(), reverse=True)

        print(scores)
        print(scores12)

        self.assertEqual(scores, scores12)

        self.assertEqual(len(domains), len(domains12))
        for d1 in domains:
            print(d1)
            domains12.remove(d1)
        self.assertEqual(len(domains12), 0)

if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)

    #plot_pruned_trees()
    unittest.main()
