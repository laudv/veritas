import numpy as np

from groot.adversary import DecisionTreeAdversary
from groot.model import GrootTreeClassifier
from groot.model import GrootRandomForestClassifier

from sklearn.datasets import make_moons

import veritas

X, y = make_moons(noise=0.3, random_state=0)
X_test, y_test = make_moons(noise=0.3, random_state=1)

attack_model = [0.5, 0.5]
is_numerical = [True, True]
num_trees = 10
forest = GrootRandomForestClassifier(attack_model=[0.1, 0.1],
        n_estimators=num_trees, max_depth=5, random_state=1)
forest.fit(X, y)

at = veritas.addtree_from_groot_ensemble(forest)

yat = at.eval(X_test) > 0.0
ygr = forest.predict(X_test) == 1.0

print("y prediction match?", np.all(yat == ygr))

pat = at.predict_proba(X_test)
pgr = forest.predict_proba(X_test)[:,1]

print("proba prediction match?", np.max(np.abs(pat-pgr)) < 1e-5) # float32 errors

accuracy = forest.score(X_test, y_test)
#adversarial_accuracy = DecisionTreeAdversary(tree, "groot").adversarial_accuracy(X_test, y_test)

print("Accuracy:", accuracy)
