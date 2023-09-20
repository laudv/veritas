Trees Ensembles
===============

You can convert an existing ensemble using the ``get_addtree`` function for XGBoost, LightGBM and scikit-learn.

RandomForest
------------

Here's an example of a model trained by a RandomForestClassifier that has been converted to Veritas' own tree ensemble representation.

.. code-block:: py
	:caption: Python
	:linenos:

	from veritas import *
	from sklearn.datasets import make_moons
	from sklearn.ensemble import RandomForestClassifier
	
	(X,Y) = make_moons(100)
	
	clf = RandomForestClassifier(
	        max_depth=4,
	        random_state=0,
	        n_estimators=3)
	
	trained_model = clf.fit(X, Y)
	
	# Convert the RandomForestClassifier model to a Veritas tree ensemble
	addtree = get_addtree(trained_model)
	
	print(f"{addtree}\n")
	
	# Print all trees in the ensemble
	for tree in addtree:
	    print(tree)
	

The output is an AddTree consisting of 3 trees, as was defined in the RandomForestClassifier.

.. code-block:: sh
	:caption: Output

	SKLEARN: classifier with 1 classes
	AddTree with 3 trees and base_scores [0]
	
	Node(id=0, split=[F0 < -0.0149985], sz=9, left=1, right=2)
	├─ Leaf(id=1, sz=1, value=[1])
	├─ Node(id=2, split=[F1 < 0.463324], sz=7, left=3, right=4)
	│  ├─ Node(id=3, split=[F0 < 0.915447], sz=5, left=5, right=6)
	│  │  ├─ Leaf(id=5, sz=1, value=[0])
	│  │  ├─ Node(id=6, split=[F1 < -0.0227674], sz=3, left=7, right=8)
	│  │  │  ├─ Leaf(id=7, sz=1, value=[0])
	│  │  │  └─ Leaf(id=8, sz=1, value=[0.333333])
	│  └─ Leaf(id=4, sz=1, value=[1])
	
	Node(id=0, split=[F1 < 0.311975], sz=15, left=1, right=2)
	├─ Node(id=1, split=[F1 < -0.0227674], sz=7, left=3, right=4)
	│  ├─ Leaf(id=3, sz=1, value=[0])
	│  ├─ Node(id=4, split=[F1 < 0.00464122], sz=5, left=5, right=6)
	│  │  ├─ Leaf(id=5, sz=1, value=[1])
	│  │  ├─ Node(id=6, split=[F1 < 0.281248], sz=3, left=7, right=8)
	│  │  │  ├─ Leaf(id=7, sz=1, value=[0.4])
	│  │  │  └─ Leaf(id=8, sz=1, value=[0])
	├─ Node(id=2, split=[F0 < 1.4735], sz=7, left=9, right=10)
	│  ├─ Node(id=9, split=[F1 < 0.463324], sz=5, left=11, right=12)
	│  │  ├─ Node(id=11, split=[F1 < 0.434907], sz=3, left=13, right=14)
	│  │  │  ├─ Leaf(id=13, sz=1, value=[0.909091])
	│  │  │  └─ Leaf(id=14, sz=1, value=[0])
	│  │  └─ Leaf(id=12, sz=1, value=[1])
	│  └─ Leaf(id=10, sz=1, value=[0])
	
	Node(id=0, split=[F1 < 0.373695], sz=13, left=1, right=2)
	├─ Node(id=1, split=[F1 < -0.0227674], sz=9, left=3, right=4)
	│  ├─ Leaf(id=3, sz=1, value=[0])
	│  ├─ Node(id=4, split=[F1 < 0.156384], sz=7, left=5, right=6)
	│  │  ├─ Node(id=5, split=[F1 < 0.0366763], sz=3, left=7, right=8)
	│  │  │  ├─ Leaf(id=7, sz=1, value=[0.166667])
	│  │  │  └─ Leaf(id=8, sz=1, value=[1])
	│  │  ├─ Node(id=6, split=[F1 < 0.188025], sz=3, left=9, right=10)
	│  │  │  ├─ Leaf(id=9, sz=1, value=[0])
	│  │  │  └─ Leaf(id=10, sz=1, value=[0.2])
	├─ Node(id=2, split=[F0 < 1.44946], sz=3, left=11, right=12)
	│  ├─ Leaf(id=11, sz=1, value=[1])
	│  └─ Leaf(id=12, sz=1, value=[0])
	
	


XGBoost
-------

The same can be done for XGBoost. 

.. code-block:: py
	:caption: Python
	:linenos:

	import xgboost as xgb
	
	(X,Y) = make_moons(100)
	
	clf = xgb.XGBClassifier(
	    objective="binary:logistic",
	    nthread=4,
	    tree_method="hist",
	    max_depth=4,
	    learning_rate=0.6,
	    n_estimators=3)
	
	trained_model = clf.fit(X, Y)
	
	# Convert the XGBoost model to a Veritas tree ensemble
	addtree = get_addtree(trained_model)
	
	print(f"{addtree}\n")
	
	# Print all trees in the ensemble
	for tree in addtree:
	    print(tree)
	

.. code-block:: sh
	:caption: Output

	AddTree with 3 trees and base_scores [0]
	
	Node(id=0, split=[F1 < 0.127877], sz=13, left=1, right=2)
	├─ Node(id=1, split=[F1 < 0], sz=5, left=3, right=4)
	│  ├─ Leaf(id=3, sz=1, value=[1.06667])
	│  ├─ Node(id=4, split=[F1 < 0.0661163], sz=3, left=5, right=6)
	│  │  ├─ Leaf(id=5, sz=1, value=[-0.24])
	│  │  └─ Leaf(id=6, sz=1, value=[0.6])
	├─ Node(id=2, split=[F0 < 0.997945], sz=7, left=7, right=8)
	│  ├─ Node(id=7, split=[F1 < 0.545535], sz=5, left=9, right=10)
	│  │  ├─ Node(id=9, split=[F0 < -0.838088], sz=3, left=11, right=12)
	│  │  │  ├─ Leaf(id=11, sz=1, value=[-0.763636])
	│  │  │  └─ Leaf(id=12, sz=1, value=[-0.0705882])
	│  │  └─ Leaf(id=10, sz=1, value=[-1.06667])
	│  └─ Leaf(id=8, sz=1, value=[0.72])
	
	Node(id=0, split=[F1 < 0.375267], sz=13, left=1, right=2)
	├─ Node(id=1, split=[F0 < -0.926917], sz=7, left=3, right=4)
	│  ├─ Leaf(id=3, sz=1, value=[-0.547081])
	│  ├─ Node(id=4, split=[F0 < 0.926917], sz=5, left=5, right=6)
	│  │  ├─ Leaf(id=5, sz=1, value=[0.77477])
	│  │  ├─ Node(id=6, split=[F0 < 1.03205], sz=3, left=7, right=8)
	│  │  │  ├─ Leaf(id=7, sz=1, value=[-0.571337])
	│  │  │  └─ Leaf(id=8, sz=1, value=[0.711001])
	├─ Node(id=2, split=[F1 < 0.545535], sz=5, left=9, right=10)
	│  ├─ Node(id=9, split=[F0 < 0.00820999], sz=3, left=11, right=12)
	│  │  ├─ Leaf(id=11, sz=1, value=[0.0228037])
	│  │  └─ Leaf(id=12, sz=1, value=[-0.217119])
	│  └─ Leaf(id=10, sz=1, value=[-0.692829])
	
	Node(id=0, split=[F0 < 1.03205], sz=9, left=1, right=2)
	├─ Node(id=1, split=[F1 < 0], sz=7, left=3, right=4)
	│  ├─ Leaf(id=3, sz=1, value=[0.485168])
	│  ├─ Node(id=4, split=[F0 < 0.1596], sz=5, left=5, right=6)
	│  │  ├─ Node(id=5, split=[F0 < 0], sz=3, left=7, right=8)
	│  │  │  ├─ Leaf(id=7, sz=1, value=[-0.604258])
	│  │  │  └─ Leaf(id=8, sz=1, value=[0.531517])
	│  │  └─ Leaf(id=6, sz=1, value=[-0.664472])
	└─ Leaf(id=2, sz=1, value=[0.600694])
	
	


LightGBM
--------

.. code-block:: py
	:caption: Python
	:linenos:

	import lightgbm as lgbm
	
	(X,Y) = make_moons(100)
	
	clf = lgbm.LGBMClassifier(
	    objective="binary",
	    num_leaves=15,
	    nthread=4,
	    max_depth=3,
	    learning_rate=0.9,
	    n_estimators=3,
	    verbose=-1)
	
	trained_model = clf.fit(X, Y)
	
	# Convert the LGBM model to a Veritas tree ensemble
	addtree = get_addtree(trained_model)
	
	print(f"{addtree}\n")
	
	# Print all trees in the ensemble
	for tree in addtree:
	    print(tree)
	

.. code-block:: sh
	:caption: Output

	AddTree with 3 trees and base_scores [0]
	
	Node(id=0, split=[F1 < 0.126305], sz=7, left=1, right=2)
	├─ Node(id=1, split=[F1 < -0.301002], sz=3, left=3, right=4)
	│  ├─ Leaf(id=3, sz=1, value=[1.8])
	│  └─ Leaf(id=4, sz=1, value=[1.14545])
	├─ Node(id=2, split=[F1 < 0.545535], sz=3, left=5, right=6)
	│  ├─ Leaf(id=5, sz=1, value=[-0.2])
	│  └─ Leaf(id=6, sz=1, value=[-1.8])
	
	Node(id=0, split=[F0 < -1e-35], sz=7, left=1, right=2)
	├─ Leaf(id=1, sz=1, value=[-1.54316])
	├─ Node(id=2, split=[F0 < 1.12781], sz=5, left=3, right=4)
	│  ├─ Node(id=3, split=[F0 < 0.663468], sz=3, left=5, right=6)
	│  │  ├─ Leaf(id=5, sz=1, value=[0.818379])
	│  │  └─ Leaf(id=6, sz=1, value=[-1.14612])
	│  └─ Leaf(id=4, sz=1, value=[1.44767])
	
	Node(id=0, split=[F1 < -1e-35], sz=5, left=1, right=2)
	├─ Leaf(id=1, sz=1, value=[1.17241])
	├─ Node(id=2, split=[F1 < 0.495359], sz=3, left=3, right=4)
	│  ├─ Leaf(id=3, sz=1, value=[-0.13769])
	│  └─ Leaf(id=4, sz=1, value=[-0.877739])
	
	
