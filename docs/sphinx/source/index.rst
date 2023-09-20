.. Veritas documentation master file, created by
   sphinx-quickstart on Thu Aug 31 16:05:30 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
======================================================
Versatile Verification of Tree Ensembles with VERITAS
======================================================

`Veritas in action blog post <https://dtai.cs.kuleuven.be/sports/blog/versatile-verification-of-soccer-analytics-models/>`__

**Veritas** is a versatile verification tool for tree ensembles. You can use
Veritas to generate `adversarial examples`, check `robustness`, find `dominant
attributes` or simply ask `domain specific questions` about your model.

Veritas uses its own tree representation and does not assume a specific model format (like XGBoost's JSON dump).
This makes it easy to use with many tree/ensemble learners. A translation function is included for XGBoost, LightGBM and scikit-learn ensembles.

For more information, refer to the paper:

   Versatile Verification of Tree Ensembles.
   Laurens Devos, Wannes Meert, and Jesse Davis.
   ICML 2021
   http://proceedings.mlr.press/v139/devos21a.html

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   usage/installation
   usage/tree_ensembles
   usage/queries
   usage/custom_trees

.. toctree::
   :maxdepth: 1
   :caption: API:

   api/python/python
   api/python/model_conversion
   api/cpp/cpp


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
