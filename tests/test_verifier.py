import matplotlib.pyplot as plt
import unittest, json, math
import numpy as np
import z3

from treeck import *
from treeck.verifier import Verifier, not_in_domain_constraint, in_domain_constraint
from treeck.z3backend import Z3Backend as Backend

class TestVerifier(unittest.TestCase):

    def myAssertAlmostEqual(self, a, b, eps=1e-6):
        self.assertTrue(type(a) == type(b))
        if isinstance(a, list) or isinstance(a, tuple):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.myAssertAlmostEqual(x, y, eps=eps)
        elif isinstance(a, float):
            self.assertAlmostEqual(a, b, delta=eps)
        else:
            self.assertEqual(a, b)

    def test_single_tree(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        dt = DomTree(at, {})
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend())

        self.assertEqual(v._backend.check(v.fvar() < 0.0), Verifier.Result.SAT)
        v.add_tree(0)
        self.assertEqual(v._backend.check(v.fvar() < 0.0), Verifier.Result.UNSAT)
        self.assertEqual(v._backend.check(v.fvar() > 0.0), Verifier.Result.SAT)
        self.assertEqual(v._backend.check(v.fvar() < 0.41), Verifier.Result.SAT)
        self.assertEqual(v._backend.check(v.fvar() > 0.41), Verifier.Result.UNSAT)

        dt = DomTree(at, {0: RealDomain(1, 3)})
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend())
        v.add_constraint(v.xvar(0) < 2.0)
        v.add_tree(0)
        check = v.check(v.fvar() != t.get_leaf_value(t.right( t.left(t.root()))))
        self.assertEqual(check, Verifier.Result.UNSAT)

        dt = DomTree(at, {0: RealDomain(1, 3)})
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend())
        v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < 0.0), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() > 0.0), Verifier.Result.SAT)
        self.assertEqual(v.check(v.fvar() < 0.41), Verifier.Result.SAT)
        self.assertEqual(v.check(v.fvar() > 0.41), Verifier.Result.UNSAT)

    #def test_family(self):
    #    at = AddTree()
    #    t = at.add_tree();
    #    t.split(t.root(), 0, 2)
    #    t.split( t.left(t.root()), 0, 1)
    #    t.split(t.right(t.root()), 0, 3)
    #    t.set_leaf_value( t.left( t.left(t.root())), 0.1)
    #    t.set_leaf_value(t.right( t.left(t.root())), 0.2)
    #    t.set_leaf_value( t.left(t.right(t.root())), 0.3)
    #    t.set_leaf_value(t.right(t.right(t.root())), 0.4)

    #    print(at)

    #    dt = DomTree(at, {})
    #    l0 = dt.get_leaf(dt.tree().root())
    #    v = Verifier(l0, Backend())
    #    v.add_all_trees()

    #    v.check()
    #    m = v.model()

    #    print(m)
    #    v.instance(0)._xs_wide_family(m["xs"])

    def test_mark_paths(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)

        #print(at)

        dt = DomTree([(at, {0: RealDomain(0, 2)}), (at, {})])
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend())
        v.add_constraint(in_domain_constraint(v, l0.get_domains(0), instance=0))
        v.add_constraint(v.xvar(0, instance=0) == v.xvar(0, instance=1))

        self.assertFalse(l0.is_reachable(0, 0, 2))
        self.assertTrue( l0.is_reachable(1, 0, 2))

        v.instance(1).mark_unreachable_paths(0)

        v.add_constraint(v.xvar(0, instance=1) < 1.0)

        self.assertTrue(l0.is_reachable(0, 0, 4))
        self.assertTrue(l0.is_reachable(1, 0, 4))

        v.instance(0).mark_unreachable_paths(0, only_feat_id=999)
        v.instance(1).mark_unreachable_paths(0, only_feat_id=999)

        self.assertTrue(l0.is_reachable(0, 0, 4)) # no effect, wrong feat_id
        self.assertTrue(l0.is_reachable(1, 0, 4))

        v.instance(0).mark_unreachable_paths(0, only_feat_id=0)
        v.instance(1).mark_unreachable_paths(0, only_feat_id=0)

        self.assertFalse(l0.is_reachable(0, 0, 4))
        self.assertFalse(l0.is_reachable(1, 0, 4))

        self.assertFalse(l0.is_reachable(1, 0, 2))

        v.add_all_trees()
        #print(v._backend._solver)
        v.check()
        m = v.model()

        self.assertLess(m[0]["xs"][0], 1.0)
        self.assertGreaterEqual(m[0]["xs"][0], 0.0)
        self.assertEqual(m[0]["xs"][0], m[1]["xs"][0])
        self.myAssertAlmostEqual(m[0]["ws"][0], 0.1)

    def test_two_trees(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 1, 1)
        t.split(t.right(t.root()), 1, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), -0.3)
        t.set_leaf_value(t.right(t.right(t.root())), -0.4)

        dt = DomTree(at, {})
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend()); v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < -0.11), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.3, -0.4])
        self.assertEqual(v.check(v.fvar() > -0.09), Verifier.Result.SAT)
        self.assertEqual(v.check(v.fvar() > 0.41), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.2, 0.2])

        v = Verifier(l0, Backend()); v.add_all_trees()
        v.add_constraint(v.xvar(0) < 2.0)
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.2, 0.2])

        v = Verifier(l0, Backend()); v.add_all_trees()
        v.add_constraint(v.xvar(0) >= 2.0)
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.3, -0.4])
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.UNSAT)


        dt = DomTree(at, {0: RealDomain(-math.inf, 2.0)})
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend()); v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.2, 0.2])

        dt = DomTree(at, {0: RealDomain(2.0, math.inf)})
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend()); v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.SAT)
        self.myAssertAlmostEqual(v.model()["ws"], [0.3, -0.4])
        self.assertEqual(v.check(v.fvar() > 0.39), Verifier.Result.UNSAT)
        v.add_constraint(v.xvar(1) < 2.0)
        self.assertEqual(v.check(v.fvar() < -0.09), Verifier.Result.UNSAT)
        self.assertEqual(v.check(v.fvar() < 0.01), Verifier.Result.SAT)

        model = v.model()
        #print(model)
        self.myAssertAlmostEqual(model["ws"], [0.3, -0.3])
        #print(v.model_family(model))
        v.add_constraint(not_in_domain_constraint(v, v.model_family(model), 0))
        self.assertEqual(v.check(v.fvar() < 0.01), Verifier.Result.SAT)
        model = v.model()
        self.myAssertAlmostEqual(model["ws"], [0.3, -0.3])
        v.add_constraint(not_in_domain_constraint(v, v.model_family(model), 0))
        check = v.check(v.fvar() < 0.01) 
        self.assertEqual(check, Verifier.Result.UNSAT)

    def test_multi_instance(self):
        at = AddTree()
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 0, 1)
        t.split(t.right(t.root()), 0, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), 0.3)
        t.set_leaf_value(t.right(t.right(t.root())), 0.4)
        t = at.add_tree();
        t.split(t.root(), 0, 2)
        t.split( t.left(t.root()), 1, 1)
        t.split(t.right(t.root()), 1, 3)
        t.set_leaf_value( t.left( t.left(t.root())), 0.1)
        t.set_leaf_value(t.right( t.left(t.root())), 0.2)
        t.set_leaf_value( t.left(t.right(t.root())), -0.3)
        t.set_leaf_value(t.right(t.right(t.root())), -0.4)

        #print(at)

        dt = DomTree([(at, {}), (at, {})])
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend())
        v.add_all_trees();
        v.add_constraint(v.instance(0).fvar() > v.instance(1).fvar())
        v.add_constraint(v.instance(1).fvar() > 0)
        v.add_constraint(v.instance(0).xvar(0) == v.instance(1).xvar(0))
        v.add_constraint(v.instance(1).xvar(1) < 1)
        models = []
        while v.check() == Verifier.Result.SAT:
            m = v.model()

            self.assertGreater(m[0]["f"], m[1]["f"])
            self.assertGreater(m[1]["f"], 0.0)
            self.assertEqual(m[0]["xs"][0], m[1]["xs"][0])
            self.assertLess(m[1]["xs"][1], 1.0)

            print("MODEL xs", m[0]["xs"], m[1]["xs"])
            print("      ws", m[0]["ws"], m[1]["ws"])
            print("       f", m[0]["f"], m[1]["f"])
            models.append(m)
            fam = v.model_family(m)
            #print("FAM", fam)
            v.add_constraint(not_in_domain_constraint(v, fam[0], 0) |
                             not_in_domain_constraint(v, fam[1], 1))

        self.assertEqual(len(models), 4)

    def test_img(self):
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        at = AddTree.read("tests/models/xgb-img-easy.json")

        m, M = min(ys), max(ys)
        img = np.array(ys).reshape((100, 100))

        #fig, ax = plt.subplots(2, 2)
        #ax[0, 0].imshow(img0, vmin=m, vmax=M)
        #ax[0, 1].imshow(img1, vmin=m, vmax=M)
        #ax[1, 0].imshow(img2, vmin=m, vmax=M)
        #ax[1, 1].imshow(img3, vmin=m, vmax=M)
        #plt.show()

        print("< 0")
        dt = DomTree(at, {})
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend()); v.add_all_trees()
        self.assertEqual(v.check(v.fvar() < 0.0), Verifier.Result.SAT)
        model = v.model()
        self.assertLess(model["f"], 0.0)
        self.assertGreaterEqual(model["f"], m)

        print("< m, > M")
        self.assertEqual(v.check((v.fvar() < m) | (v.fvar() > M)), Verifier.Result.UNSAT)

        quandrant = 0
        img = np.array(ys).reshape((100, 100))
        for x0 in [0, 50]:
            for y0 in [0, 50]:
                print("quadrant", quandrant)
                x1, y1 = x0 + 50, y0 + 50
                imgq = img[x0:x1, y0:y1]
                m, M = imgq.min(), imgq.max()

                dt = DomTree(at, {0: RealDomain(x0, x1), 1: RealDomain(y0, y1)})
                l0 = dt.get_leaf(dt.tree().root())
                v = Verifier(l0, Backend()); v.add_all_trees()

                self.assertEqual(v.check(v.fvar() < m+1e-4), Verifier.Result.SAT)
                self.assertAlmostEqual(v.model()["f"], m, delta=1e-4)
                self.assertEqual(v.check(v.fvar() > M-1e-4), Verifier.Result.SAT)
                self.assertAlmostEqual(v.model()["f"], M, delta=1e-4)

                quandrant += 1

    def test_img_sampling(self):
        # find all points with predictions less than 0.0
        with open("tests/models/xgb-img-easy-values.json") as f:
            ys = json.load(f)
        img = np.array(ys).reshape((100, 100))
        at = AddTree.read("tests/models/xgb-img-easy.json")

        dt = DomTree(at, {})
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend()); v.add_all_trees()
        v.add_constraint(v.fvar() < 0.0)

        models = []
        while v.check() == Verifier.Result.SAT:
            m = v.model()
            x = int(np.floor(m["xs"][1]))
            y = int(np.floor(m["xs"][0]))
            self.assertLess(m["f"], 0.0)
            self.assertAlmostEqual(img[y][x], m["f"], delta=1e-4)
            models.append((x, y))
            v.add_constraint(not_in_domain_constraint(v, v.model_family(m), 0))

        self.assertEqual(len(models), 60)

        #fig, ax = plt.subplots()
        #ax.imshow(img)
        #for (x, y) in models:
        #    ax.scatter([x], [y], marker="s", c="b")
        #plt.show()

    def test_mnist_multi_instance(self):
        at = AddTree.read(f"tests/models/xgb-mnist-yis0-easy.json")

        dt = DomTree([(at, {}), (at, {})])
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend());
        v.add_all_trees(0); v.add_all_trees(1)
        v.add_constraint(v.fvar(0) >  5.0) # it is with high certainty X
        v.add_constraint(v.fvar(1) < -5.0) # it is with high certainty not X

        pbeq = []
        for feat_id in AddTreeFeatureTypes(at).feat_ids():
            bvar_name = f"b{feat_id}"
            v.add_bvar(bvar_name)

            bvar = v.bvar(bvar_name)
            xvar1 = v.xvar(feat_id, 0)
            xvar2 = v.xvar(feat_id, 1)

            v.add_constraint(z3.If(bvar.get(), xvar1.get() != xvar2.get(),
                                               xvar1.get() == xvar2.get()))
            pbeq.append((bvar.get(), 1))

        N = 4
        v.add_constraint(z3.PbLe(pbeq, N)) # at most N variables differ

        count = 0
        uniques = set()

        img1_prev, img2_prev = None, None
        while count < 10:
            check = v.check()
            if not check.is_sat():
                print("UNSAT")
                break

            model = v.model()
            img1 = np.zeros((28, 28))
            img2 = np.zeros((28, 28))
            hash1 = 127
            hash2 = 91
            for fid, x in model[0]["xs"].items():
                p = np.unravel_index(fid, (28, 28))
                if x is not None: img1[p[1], p[0]] = x
                hash1 = hash((hash1, fid, x))
            for fid, x in model[1]["xs"].items():
                p = np.unravel_index(fid, (28, 28))
                if x is not None: img2[p[1], p[0]] = x
                hash2 = hash((hash2, fid, x))
            uniques.add(hash((hash1, hash2)))

            if count > 0:
                diff1 = abs(img1-img1_prev).sum()
                diff2 = abs(img2-img2_prev).sum()
                print("mnist_multi_instance: norm difference:", diff1, diff2)
                self.assertGreater(diff1, 0.0)
                self.assertGreater(diff2, 0.0)
            img1_prev, img2_prev = img1, img2

            # Ensure that the different pixels have different values in the next iteration
            # We do not care about the other pixels
            fam = v.model_family(model)
            fam_diff0 = {}
            fam_diff1 = {}
            for n, b in model["bs"].items():
                if not b: continue
                i = int(n[1:])
                p = np.unravel_index(i, (28, 28))
                print("different pixel (bvar):", i, p, model[0]["xs"][i], model[1]["xs"][i])
                fam_diff0[i] = fam[0][i]
                fam_diff1[i] = fam[1][i]
            v.add_constraint(not_in_domain_constraint(v, fam_diff0, 0) &
                             not_in_domain_constraint(v, fam_diff1, 1))

            count += 1

            print("iteration", count, "uniques", len(uniques))
            if count % 100 != 0: continue

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img1, vmin=0, vmax=255)
            ax2.imshow(img2, vmin=0, vmax=255)
            ax1.set_title("instance 1: f={:.3f}".format(model[0]["f"]))
            ax2.set_title("instance 2: f={:.3f}".format(model[1]["f"]))

            for n, b in model["bs"].items():
                if not b: continue
                i = int(n[1:])
                p = np.unravel_index(i, (28, 28))
                ax1.scatter([p[0]], [p[1]], marker=".", color="r")
                ax2.scatter([p[0]], [p[1]], marker=".", color="r")

            plt.show()

        self.assertEqual(count, 10)
        self.assertEqual(len(uniques), 10)

    def test_bin_mnist(self):
        at = AddTree.read(f"tests/models/xgb-mnist-bin-yis1-easy.json")
        dt = DomTree(at, {})
        l0 = dt.get_leaf(dt.tree().root())
        v = Verifier(l0, Backend());
        v.add_all_trees()
        v.add_constraint(v.fvar(0) > 5.0) # it is with high certainty X

        count = 0;
        uniques = set()
        img_prev = None
        while v.check().is_sat() and count < 10:
            model = v.model()
            img = np.zeros((28, 28), dtype=bool)
            hash1 = 91
            for fid, x in model["xs"].items():
                p = np.unravel_index(fid, (28, 28))
                if x is not None: img[p[1], p[0]] = x
                hash1 = hash((hash1, fid, x))
            uniques.add(hash1)

            fam = v.model_family(model)
            v.add_constraint(not_in_domain_constraint(v, fam, 0))

            if count > 0:
                ndiff = (img_prev != img).sum()
                print("bin_mnist: different =", ndiff)
                self.assertGreater(ndiff, 0)

            img_prev = img
            count += 1

            if count % 100 != 0: continue

            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_title("instance: f={:.3f}".format(model["f"]))
            plt.show()

        self.assertEqual(count, 10)
        self.assertEqual(len(uniques), 10)

if __name__ == "__main__":
    z3.set_pp_option("rational_to_decimal", True)
    z3.set_pp_option("precision", 3)
    z3.set_pp_option("max_width", 130)
    unittest.main()
