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

!code PART get_addtree_example!

The output is an AddTree consisting of 3 trees, as was defined in the XGBClassifier.

!output PART get_addtree_example!

## Cite Veritas

> Versatile Verification of Tree Ensembles.
> Laurens Devos, Wannes Meert, and Jesse Davis.
> ICML 2021
> http://proceedings.mlr.press/v139/devos21a.html

> Robustness Verification of Multiclass Tree Ensembles.
> Laurens Devos, Lorenzo Cascioli, and Jesse Davis.
> AAAI 2024
