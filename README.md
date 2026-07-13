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

### Developer Installation

```sh
# clone this repository
git clone https://github.com/laudv/veritas.git
cd veritas
git submodule init && git submodule update

# editable install; C++ changes trigger an automatic rebuild on next import
pip install --no-build-isolation --editable .
```

The `veritas`'s Python sources are picked up from `src/python/veritas`. Editing
a C++ source under `src/cpp` or `src/bindings` triggers a rebuild the next time
you `import veritas`.

To manually trigger a rebuild instead (e.g. to see full compiler output/errors
directly), build the CMake directory scikit-build-core created under `build/`
(one per Python/platform combination, e.g. `build/cp312-cp312-linux_x86_64`):

```sh
cmake --build build/cp312-cp312-linux_x86_64
# or, when using Make:
make -C build/cp312-cp312-linux_x86_64
```

## Example

You can convert an existing ensemble using the `veritas.get_addtree` function for XGBoost, LightGBM and scikit-learn.

Here's an example of a model trained by XGBoost that has been converted to Veritas' own tree ensemble representation.

```python
import veritas
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

(X,Y) = make_moons(100, random_state=0)

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
SKLEARN: RF classifier with 1 classes
AddTree with 3 trees and base_scores [-1.5]
Node(id=0, split=[F0 < -0.0160258], sz=9, left=1, right=2)
├─ Leaf(id=1, sz=1, value=[0])
├─ Node(id=2, split=[F1 < 0.549055], sz=7, left=3, right=4)
│  ├─ Node(id=3, split=[F0 < 0.915447], sz=5, left=5, right=6)
│  │  ├─ Leaf(id=5, sz=1, value=[1])
│  │  ├─ Node(id=6, split=[F1 < 0.128637], sz=3, left=7, right=8)
│  │  │  ├─ Leaf(id=7, sz=1, value=[1])
│  │  │  └─ Leaf(id=8, sz=1, value=[0.428571])
│  └─ Leaf(id=4, sz=1, value=[0])

Node(id=0, split=[F1 < 0.522767], sz=9, left=1, right=2)
├─ Node(id=1, split=[F1 < -0.0227675], sz=7, left=3, right=4)
│  ├─ Leaf(id=3, sz=1, value=[1])
│  ├─ Node(id=4, split=[F1 < 0.495359], sz=5, left=5, right=6)
│  │  ├─ Node(id=5, split=[F1 < 0.463324], sz=3, left=7, right=8)
│  │  │  ├─ Leaf(id=7, sz=1, value=[0.535714])
│  │  │  └─ Leaf(id=8, sz=1, value=[0])
│  │  └─ Leaf(id=6, sz=1, value=[1])
└─ Leaf(id=2, sz=1, value=[0])

Node(id=0, split=[F1 < 0.522767], sz=13, left=1, right=2)
├─ Node(id=1, split=[F1 < 0.311975], sz=11, left=3, right=4)
│  ├─ Node(id=3, split=[F1 < -0.0227675], sz=5, left=5, right=6)
│  │  ├─ Leaf(id=5, sz=1, value=[1])
│  │  ├─ Node(id=6, split=[F1 < 0.00464122], sz=3, left=7, right=8)
│  │  │  ├─ Leaf(id=7, sz=1, value=[0])
│  │  │  └─ Leaf(id=8, sz=1, value=[0.933333])
│  ├─ Node(id=4, split=[F1 < 0.495359], sz=5, left=9, right=10)
│  │  ├─ Node(id=9, split=[F0 < 1.44638], sz=3, left=11, right=12)
│  │  │  ├─ Leaf(id=11, sz=1, value=[0.222222])
│  │  │  └─ Leaf(id=12, sz=1, value=[1])
│  │  └─ Leaf(id=10, sz=1, value=[1])
└─ Leaf(id=2, sz=1, value=[0])

```

## Cite Veritas

> Versatile Verification of Tree Ensembles.
> Laurens Devos, Wannes Meert, and Jesse Davis.
> ICML 2021
> http://proceedings.mlr.press/v139/devos21a.html

> Robustness Verification of Multiclass Tree Ensembles.
> Laurens Devos, Lorenzo Cascioli, and Jesse Davis.
> AAAI 2024
