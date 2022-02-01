import unittest
import numpy as np

import veritas

class TestGroot(unittest.TestCase):

    def get_groot_model(self):
        from sklearn.datasets import make_moons

        from groot.adversary import DecisionTreeAdversary
        #from groot.model import GrootTreeClassifier
        from groot.model import GrootRandomForestClassifier

        X, y = make_moons(noise=0.3, random_state=0)
        X_test, y_test = make_moons(noise=0.3, random_state=1)

        attack_model = [0.5, 0.5]
        is_numerical = [True, True]
        num_trees = 10
        forest = GrootRandomForestClassifier(attack_model=[0.1, 0.1],
                n_estimators=num_trees, max_depth=5, random_state=1)
        forest.fit(X, y)

        at = veritas.addtree_from_groot_ensemble(forest)

        return at, forest

    def test_groot_model(self):
        try:
            at, forest = self.get_groot_model()
        except ModuleNotFoundError as e:
            print("Skipping GROOT tests because GROOT not installed")
            return

        yat = at.eval(X_test) > 0.0
        ygr = forest.predict(X_test) == 1.0

        self.assertTrue(np.all(yat == ygr))

        pat = at.predict_proba(X_test)
        pgr = forest.predict_proba(X_test)[:,1]

        np.assertTrue(np.max(np.abs(pat-pgr)) < 1e-5) # float32 errors

        #accuracy = forest.score(X_test, y_test)
