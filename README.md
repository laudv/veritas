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

# Install the build backend first (required for --no-build-isolation below)
pip install "scikit-build-core>=0.10"

# editable install; C++ changes trigger an automatic rebuild on next import
pip install --no-build-isolation --editable .
```

The `veritas`'s Python sources are picked up from `src/python/veritas`. Editing
a C++ source under `src/cpp` or `src/bindings` triggers a rebuild the next time
you `import veritas`.

`--no-build-isolation` tells pip to build the extension using the packages
already installed in your current environment instead of a temporary, isolated
one. It's required for the editable-rebuild-on-import behavior above: that
mechanism reruns CMake using this environment's Python at import time, so the
initial install has to use this same environment too (an isolated build
environment gets deleted once the install finishes).

To manually trigger a rebuild instead (e.g. to see full compiler output/errors
directly), build the CMake directory scikit-build-core created under `build/`
(one per Python/platform combination, e.g. `build/cp312-cp312-linux_x86_64`):

```sh
cmake --build build/cp312-cp312-linux_x86_64
# or, when using Make:
make -C build/cp312-cp312-linux_x86_64
```

CMake's Python discovery (`find_package(Python3 ...)`) resolves against
whichever `python3` is first on `PATH`. Activate your virtual environment
before running `cmake` directly (e.g. for a from-scratch manual build, see
below), or it will pick up the system Python instead. This only matters for
manual CMake invocations — `pip install` always builds against the interpreter
running `pip` itself, regardless of `PATH`.

### Running the tests

Python tests (from the repository root):

```sh
pytest tests/
```

To run all python tests, install the test dependencies first (using `pip install -e ".[test]"`). Some tests exercise optional integrations (XGBoost, LightGBM, scikit-learn, Gurobi, z3) and are skipped automatically if the corresponding package isn't installed.

Alternatively, you can run the test suite against different combinations of supported Python versions and learner versions (xgboost, lightgbm, scikit-learn) using `nox`:

```sh
# Install nox
pip install nox

# Run all test configurations
nox

# Run a specific session (e.g. tests_xgboost)
nox -s "tests_xgboost"

# Run a specific version combination (forcing a specific Python version if needed):
nox -s "tests_xgboost(xgboost='3.2.0')" --force-python 3.12
```

C++ tests use a separate CMake configuration
(`-DVERITAS_BUILD_CPPTESTS=ON -DVERITAS_BUILD_PYBINDINGS=OFF`). The test
binary looks up its data files with paths relative to the repository root
(e.g. `../tests/models/...`), so the build directory must sit exactly one
level below the repository root:

```sh
mkdir manual_build_cpp && cd manual_build_cpp
cmake -DCMAKE_BUILD_TYPE=Release -DVERITAS_BUILD_CPPTESTS=ON -DVERITAS_BUILD_PYBINDINGS=OFF ..
make -j
ctest --output-on-failure
```

### Building a release wheel with cibuildwheel

CI builds release wheels with [`cibuildwheel`](https://cibuildwheel.pypa.io/),
which builds the extension in an isolated container/environment per target
Python version and platform, so you can reproduce a CI wheel build locally
(requires Docker on Linux):

```sh
pip install cibuildwheel
cibuildwheel --output-dir wheelhouse
```

This uses the `[tool.cibuildwheel]` settings in `pyproject.toml`: after
building each wheel, it installs it into a fresh venv along with
`test-requires` (the optional integrations exercised by the test suite) and
runs `test-command`, which is `cd {project} && pytest tests` — the same
test suite as above, but run against the actual built wheel rather than an
editable install. Restrict this to a single target while
iterating with e.g. `CIBW_BUILD="cp312-manylinux_x86_64" cibuildwheel
--output-dir wheelhouse`.

## Example

You can convert an existing ensemble using the `veritas.get_addtree` function for XGBoost, LightGBM and scikit-learn.

Here's an example of a model trained by XGBoost that has been converted to Veritas' own tree ensemble representation.

```python
import veritas
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

(X, Y) = make_moons(100, random_state=0)

clf = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=3)
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
