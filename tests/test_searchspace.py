import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import unittest, time, json
import z3

from treeck import *
from treeck.plot import TreePlot
from treeck.verifier import Verifier, SplitCheckStrategy as Strategy
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
    def test_single_tree(self):
        at = AddTree(1)
        at.base_score = 10
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        print(at)
        sp = SearchSpace(at)
        sp.split(2)
        leafs = sp.leafs()

        self.assertEqual(len(leafs), 2)

        atp = [sp.get_pruned_addtree(l) for l in leafs]
        dom = [sp.get_domains(l) for l in leafs]

        self.assertEqual(dom[0][0].hi, 2)
        self.assertEqual(dom[1][0].lo, 2)
        self.assertEqual(atp[0].base_score, at.base_score)
        self.assertEqual(atp[1].base_score, at.base_score)
        self.assertEqual(sp.scores(), [6])

        self.assertEqual(atp[0][0].get_split(0), (0, 1))
        self.assertEqual(atp[0][0].get_leaf_value(1), 0.1)
        self.assertEqual(atp[0][0].get_leaf_value(2), 0.2)
        self.assertEqual(atp[1][0].get_split(0), (0, 3))
        self.assertEqual(atp[1][0].get_leaf_value(1), 0.3)
        self.assertEqual(atp[1][0].get_leaf_value(2), 0.4)

        sp = SearchSpace(at)
        sp.split(100) # more than possible!
        leafs = sp.leafs()
        leaf_values = []

        for leaf in sp.leafs():
            pat = sp.get_pruned_addtree(leaf)
            domains = sp.get_domains(leaf)

            self.assertTrue(pat[0].is_leaf(pat[0].root()))
            leaf_values.append(pat[0].get_leaf_value(pat[0].root()))

        leaf_values.sort()
        self.assertEqual(leaf_values, [0.1, 0.2, 0.3, 0.4])

        self.assertEqual(len(leafs), 4)
        self.assertEqual(sp.scores(), [6, 2, 2])

    def test_covtype(self):
        at = AddTree.read("tests/models/xgb-covtype-easy.json")
        sp = SearchSpace(at)

        sp.split(10)
        threshold = -7.3

        results = []

        for i, leaf in enumerate(sp.leafs()):
            addtree = sp.get_pruned_addtree(leaf)
            domains = sp.get_domains(leaf)

            #print(addtree)
            print(f"Domains for leaf {i}({leaf}):",
                    list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

            v = Verifier(domains, addtree, Backend(), Strategy())
            v.add_constraint(v.xvar(35) > 0.5)
            results.append(v.verify(v.fvar() < threshold))

        unsat, sat = Verifier.Result.UNSAT, Verifier.Result.SAT
        self.assertEqual(results, [unsat, unsat, unsat, unsat, sat, unsat, unsat, unsat, unsat, unsat])

    def test_img1(self):
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        at = AddTree.read("tests/models/xgb-img-easy.json")
        img = np.array(ys).reshape((100, 100))
        m, M = min(ys), max(ys)
        argm = np.unravel_index(img.argmin(), img.shape)

        sp = SearchSpace(at)
        sp.split(20)
        results = []

        for i, leaf in enumerate(sp.leafs()):
            addtree = sp.get_pruned_addtree(leaf)
            domains = sp.get_domains(leaf)

            v = Verifier(domains, addtree, Backend(), Strategy())
            check = v.verify(v.fvar() < m+1e-4)
            results.append(check)

            if check == Verifier.Result.SAT:
                sat_leaf = leaf
                sat_model = v.model()

            #print(addtree)
            print(f"Domains for {check} leaf {i}({leaf}):",
                    list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

        #fig, ax = plt.subplots(1)
        #ax.set_title("min")
        #im = ax.imshow(img)

        #for leaf in sp.leafs():
        #    doms = sp.get_domains(leaf)
        #    y0, y1 = max(0, doms[0].lo), min(99, doms[0].hi)
        #    x0, x1 = max(0, doms[1].lo), min(99, doms[1].hi)
        #    print(x0, y0, x1, y1, argm)
        #    color = "r" if leaf == sat_leaf else "b"
        #    zorder = 5 if leaf == sat_leaf else 1

        #    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1, edgecolor=color, facecolor='none', zorder=zorder)
        #    ax.add_patch(rect)

        #print("w=", sum(sat_model["ws"])+at.base_score, "m=", m, "img[*,*]=",
        #        img[int(np.floor(sat_model["xs"][0])), int(np.floor(sat_model["xs"][1]))])
        #ax.scatter([sat_model["xs"][1]], [sat_model["xs"][0]], marker="o", c="b")
        #ax.scatter([argm[1]], [argm[0]], marker="x", c="r")
        #ax.set_xlim([0, 99])
        #ax.set_ylim([99, 0])
        #fig.colorbar(im, ax=ax)
        #plt.show()

        self.assertEqual(sum(map(lambda x: x==Verifier.Result.SAT, results)), 1)

    def test_img2(self):
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        at = AddTree.read("tests/models/xgb-img-easy.json")
        img = np.array(ys).reshape((100, 100))
        m, M = min(ys), max(ys)
        argM = np.unravel_index(img.argmax(), img.shape)

        sp = SearchSpace(at)
        sp.split(20)
        results = []

        for i, leaf in enumerate(sp.leafs()):
            addtree = sp.get_pruned_addtree(leaf)
            domains = sp.get_domains(leaf)

            v = Verifier(domains, addtree, Backend(), Strategy())
            check = v.verify(v.fvar() > M-1e-4)
            results.append(check)

            if check == Verifier.Result.SAT:
                sat_leaf = leaf
                sat_model = v.model()

            #print(addtree)
            print(f"Domains for {check} leaf {i}({leaf}):",
                    list(filter(lambda d: not d[1].is_everything(), enumerate(domains))))

        #fig, ax = plt.subplots(1)
        #ax.set_title("max")
        #im = ax.imshow(img)

        #for leaf in sp.leafs():
        #    doms = sp.get_domains(leaf)
        #    y0, y1 = max(0, doms[0].lo), min(99, doms[0].hi)
        #    x0, x1 = max(0, doms[1].lo), min(99, doms[1].hi)
        #    print(x0, y0, x1, y1, argM)
        #    color = "r" if leaf == sat_leaf else "b"
        #    zorder = 5 if leaf == sat_leaf else 1

        #    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1, edgecolor=color, facecolor='none', zorder=zorder)
        #    ax.add_patch(rect)

        #print("w=", sum(sat_model["ws"])+at.base_score, "M=", M, "img[*,*]=",
        #        img[int(np.floor(sat_model["xs"][0])), int(np.floor(sat_model["xs"][1]))])
        #ax.scatter([sat_model["xs"][1]], [sat_model["xs"][0]], marker="o", c="b")
        #ax.scatter([argM[1]], [argM[0]], marker="x", c="r")
        #ax.set_xlim([0, 99])
        #ax.set_ylim([99, 0])
        #fig.colorbar(im, ax=ax)
        #plt.show()

        self.assertEqual(sum(map(lambda x: x==Verifier.Result.SAT, results)), 1)

    def test_img3(self):
        at = AddTree.read("tests/models/xgb-img-easy.json")
        sp = SearchSpace(at)
        sp.split(4)
        domains = [sp.get_domains(l) for l in sp.leafs()]
        ats = [sp.get_pruned_addtree(l) for l in sp.leafs()]
        scores = sp.scores()
        self.assertEqual(len(domains), 4)

        for x in ats:
            print(at.num_nodes(), "->", x.num_nodes())

        print("SP0")
        sp0 = SearchSpace(at)
        sp0.split(2)
        doms1, doms2 = [sp0.get_domains(l) for l in sp0.leafs()]
        at1, at2 = [sp0.get_pruned_addtree(l) for l in sp0.leafs()]

        print("SP1")
        sp1 = SearchSpace(at1, doms1)
        sp1.split(2)
        doms3, doms4 = [sp1.get_domains(l) for l in sp1.leafs()]
        at3, at4 = [sp1.get_pruned_addtree(l) for l in sp1.leafs()]

        print("SP2")
        sp2 = SearchSpace(at2, doms2)
        sp2.split(2)
        doms5, doms6 = [sp2.get_domains(l) for l in sp2.leafs()]
        at5, at6 = [sp2.get_pruned_addtree(l) for l in sp2.leafs()]

        for x, y in [(at1, at3), (at1, at4), (at2, at5), (at2, at6)]:
            print(at.num_nodes(), "->", x.num_nodes(), "->", y.num_nodes())

        scores12 = sorted(sp0.scores() + sp1.scores() + sp2.scores(), reverse=True)

        print(scores)
        print(scores12)

        self.assertEqual(scores, scores12)

        #self.assertEqual(len(domains), len(domains12))
        #for d1 in domains:
        #    print(d1)
        #    domains12.remove(d1)
        #self.assertEqual(len(domains12), 0)

    def test_img4(self):
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        at = AddTree.read("tests/models/xgb-img-easy.json")

        sp = SearchSpace(at)
        sp.split(40)
        ats = [sp.get_pruned_addtree(l) for l in sp.leafs()]
        doms = [sp.get_domains(l) for l in sp.leafs()]
        scores = sp.scores()

        #print(sorted(scores))
        #print(sorted([x.num_nodes() for x in ats]))

        num_nodes = sum(map(lambda x: x.num_nodes(), ats))
        print("num_nodes 1×40:", num_nodes)

        for x in np.linspace(0.5, 100, 100):
            for y in np.linspace(0.5, 100, 100):
                f = at.predict_single([x, y])

                num_applicalbe_domains = 0
                for (at0, dom) in zip(ats, doms):
                    if dom[0].contains(x) and dom[1].contains(y):
                        f0 = at0.predict_single([x, y])
                        self.assertEqual(f, f0)
                        num_applicalbe_domains += 1
                self.assertEqual(num_applicalbe_domains, 1)

    def test_img5(self):
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        at = AddTree.read("tests/models/xgb-img-easy.json")

        sp = SearchSpace(at)
        sp.split(4)
        at1, at2, at3, at4 = [sp.get_pruned_addtree(l) for l in sp.leafs()]
        dom1, dom2, dom3, dom4 = [sp.get_domains(l) for l in sp.leafs()]

        sp1 = SearchSpace(at1, dom1)
        sp2 = SearchSpace(at2, dom2)
        sp3 = SearchSpace(at3, dom3)
        sp4 = SearchSpace(at4, dom4)
        sp1.split(10)
        sp2.split(10)
        sp3.split(10)
        sp4.split(10)

        ats = \
                [sp1.get_pruned_addtree(l) for l in sp1.leafs()]\
              + [sp2.get_pruned_addtree(l) for l in sp2.leafs()]\
              + [sp3.get_pruned_addtree(l) for l in sp3.leafs()]\
              + [sp4.get_pruned_addtree(l) for l in sp4.leafs()]
        doms = \
                [sp1.get_domains(l) for l in sp1.leafs()]\
              + [sp2.get_domains(l) for l in sp2.leafs()]\
              + [sp3.get_domains(l) for l in sp3.leafs()]\
              + [sp4.get_domains(l) for l in sp4.leafs()]

        scores = sp1.scores() + sp2.scores() + sp3.scores() + sp4.scores()

        #print(sorted(scores))
        #print(sorted([x.num_nodes() for x in ats]))
        num_nodes = sum(map(lambda x: x.num_nodes(), ats))
        print("num_nodes 4×10:", num_nodes)

        for x in np.linspace(0.5, 100, 100):
            for y in np.linspace(0.5, 100, 100):
                f = at.predict_single([x, y])

                num_applicalbe_domains = 0
                for (at0, dom) in zip(ats, doms):
                    if dom[0].contains(x) and dom[1].contains(y):
                        f0 = at0.predict_single([x, y])
                        self.assertEqual(f, f0)
                        num_applicalbe_domains += 1
                self.assertEqual(num_applicalbe_domains, 1)




if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)

    #plot_pruned_trees()
    unittest.main()
