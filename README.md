[![PyPi Version](https://img.shields.io/pypi/v/dtai-veritas)](https://pypi.org/project/dtai-veritas/)

# Versatile Verification of Tree Ensembles with VERITAS

[Veritas in action blog post](https://dtai.cs.kuleuven.be/sports/blog/versatile-verification-of-soccer-analytics-models/)

**Veritas** is a versatile verification tool for tree ensembles. You can use
Veritas to generate _adversarial examples_, check _robustness_, find _dominant
attributes_ or simply ask _domain specific questions_ about your model.

## Installation

Install from PyPI:

```sh
pip install dtai-veritas
```

From source:

```sh
# Clone this repository
git clone https://github.com/laudv/veritas.git
cd veritas
git submodule init && git submodule update
pip install .
```

Veritas should work on Linux (GCC), Mac (LLVM), and Windows (MSVC). If you encounter issues, feel free to contact me or open an issue.

To pull the latest updates from Github, simply `git pull` the changes and reinstall using `pip`: `pip install --force-reinstall .`.

## Example

You can convert an existing ensemble using the `veritas.get_addtree` function for XGBoost, LightGBM and scikit-learn.

Here's an example of a model trained by XGBoost that has been converted to Veritas' own tree ensemble representation.

```python
import veritas
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

(X,Y) = make_moons(100)

clf = RandomForestClassifier(
        max_depth=4,
        random_state=0,
        n_estimators=3)
clf.fit(X, Y)

# Convert the RandomForestClassifier model to a Veritas tree ensemble
at = veritas.get_addtree(clf)

print(at)
for tree in at:
    print(tree)
```

The output is an AddTree consisting of 3 trees, as was defined in the XGBClassifier.

```
SKLEARN: classifier with 1 classes
AddTree with 3 trees and base_scores [-1.5]
Node(id=0, split=[F0 < 1.01603], sz=13, left=1, right=2)
├─ Node(id=1, split=[F1 < 0.373695], sz=11, left=3, right=4)
│  ├─ Node(id=3, split=[F0 < 0.903847], sz=5, left=5, right=6)
│  │  ├─ Node(id=5, split=[F1 < -0.0490553], sz=3, left=7, right=8)
│  │  │  ├─ Leaf(id=7, sz=1, value=[1])
│  │  │  └─ Leaf(id=8, sz=1, value=[0.857143])
│  │  └─ Leaf(id=6, sz=1, value=[0])
│  ├─ Node(id=4, split=[F0 < 0.0170531], sz=5, left=9, right=10)
│  │  ├─ Node(id=9, split=[F1 < 0.522767], sz=3, left=11, right=12)
│  │  │  ├─ Leaf(id=11, sz=1, value=[0.375])
│  │  │  └─ Leaf(id=12, sz=1, value=[0])
│  │  └─ Leaf(id=10, sz=1, value=[0])
└─ Leaf(id=2, sz=1, value=[1])

Node(id=0, split=[F1 < 0.0366763], sz=11, left=1, right=2)
├─ Leaf(id=1, sz=1, value=[1])
├─ Node(id=2, split=[F1 < 0.463324], sz=9, left=3, right=4)
│  ├─ Node(id=3, split=[F1 < 0.343616], sz=7, left=5, right=6)
│  │  ├─ Node(id=5, split=[F1 < 0.311975], sz=3, left=7, right=8)
│  │  │  ├─ Leaf(id=7, sz=1, value=[0.48])
│  │  │  └─ Leaf(id=8, sz=1, value=[0])
│  │  ├─ Node(id=6, split=[F0 < -0.459353], sz=3, left=9, right=10)
│  │  │  ├─ Leaf(id=9, sz=1, value=[0])
│  │  │  └─ Leaf(id=10, sz=1, value=[1])
│  └─ Leaf(id=4, sz=1, value=[0])

Node(id=0, split=[F1 < 0.373695], sz=15, left=1, right=2)
├─ Node(id=1, split=[F1 < -0.0490553], sz=7, left=3, right=4)
│  ├─ Leaf(id=3, sz=1, value=[1])
│  ├─ Node(id=4, split=[F1 < 0.0650932], sz=5, left=5, right=6)
│  │  ├─ Leaf(id=5, sz=1, value=[0])
│  │  ├─ Node(id=6, split=[F1 < 0.126305], sz=3, left=7, right=8)
│  │  │  ├─ Leaf(id=7, sz=1, value=[1])
│  │  │  └─ Leaf(id=8, sz=1, value=[0.6])
├─ Node(id=2, split=[F1 < 0.522767], sz=7, left=9, right=10)
│  ├─ Node(id=9, split=[F0 < 1.46243], sz=5, left=11, right=12)
│  │  ├─ Node(id=11, split=[F0 < 0.450484], sz=3, left=13, right=14)
│  │  │  ├─ Leaf(id=13, sz=1, value=[0.333333])
│  │  │  └─ Leaf(id=14, sz=1, value=[0])
│  │  └─ Leaf(id=12, sz=1, value=[1])
│  └─ Leaf(id=10, sz=1, value=[0])

```

## Cite Veritas

> Versatile Verification of Tree Ensembles.
> Laurens Devos, Wannes Meert, and Jesse Davis.
> ICML 2021
> http://proceedings.mlr.press/v139/devos21a.html

> Robustness Verification of Multiclass Tree Ensembles.
> Laurens Devos, Lorenzo Cascioli, and Jesse Davis.
> AAAI 2024
