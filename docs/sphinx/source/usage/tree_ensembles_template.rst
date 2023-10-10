Trees Ensembles
===============

You can convert an existing ensemble using the ``get_addtree`` function for XGBoost, LightGBM and scikit-learn.

RandomForest
------------

Here's an example of a model trained by a RandomForestClassifier that has been converted to Veritas' own tree ensemble representation.

!code PART get_addtree_example_RF!

The output is an AddTree consisting of 3 trees, as was defined in the RandomForestClassifier.

!output PART get_addtree_example_RF!


XGBoost
-------

The same can be done for XGBoost. 

!code PART get_addtree_example_XGB!

!output PART get_addtree_example_XGB!


LightGBM
--------

!code PART get_addtree_example_LGBM!

!output PART get_addtree_example_LGBM!
