Custom Trees
============

Constructing an Additive Tree Ensemble or `AddTree`
---------------------------------------------------

Veritas uses its own tree ensemble representation. You can manually build one to try Veritas out.
Here's an example of a manually constructed tree ensemble.

.. code-block:: py
	:caption: Python
	:linenos:

	import numpy as np
	from veritas import *
	
	# Manually create a two-tree ensemble
	#
	#       F0 < 2                     F0 < 3
	#       /    \                     /    \         
	#   F0 < 1   F0 < 3     +     F1 < 5     F1 < 0
	#   /   \     /   \           /   \       /    \
	#  3     4   5     6         30   40     50     F2
	#                                             /    \
	#                                            70    80
	
	at = AddTree(1)  # Empty ensemble with int his case 1 value in the leafs
	t = at.add_tree();
	t.split(t.root(), 0, 2)   # split(node_id, feature_id, split_value)
	t.split( t.left(t.root()), 0, 1)
	t.split(t.right(t.root()), 0, 3)
	t.set_leaf_value( t.left( t.left(t.root())), 3)
	t.set_leaf_value(t.right( t.left(t.root())), 4)
	t.set_leaf_value( t.left(t.right(t.root())), 5)
	t.set_leaf_value(t.right(t.right(t.root())), 6)
	
	t = at.add_tree();
	t.split(t.root(), 0, 3)
	t.split( t.left(t.root()), 1, 5)
	t.split(t.right(t.root()), 1, 0)
	t.split(t.right(t.right(t.root())), 2) # Boolean split (ie < 1.0)
	t.set_leaf_value( t.left( t.left(t.root())), 30)
	t.set_leaf_value(t.right( t.left(t.root())), 40)
	t.set_leaf_value( t.left(t.right(t.root())), 50)
	t.set_leaf_value( t.left(t.right(t.right(t.root()))), 70)
	t.set_leaf_value(t.right(t.right(t.right(t.root()))), 80)
	
	# Print the trees (including the node-ids)
	print(at[0])
	print(at[1])
	
	# Evaluate this ensemble
	print("Eval:", at.eval(np.array([[0, 0, 0], [15, -3, 9]])))
	

This outputs the following. Note that the Boolean split on feature 2 is replaced with a less than split splitting on value 0.5 (``veritas.BOOL_SPLIT_VALUE``). You can use the pre-defined domains for `TRUE` and `FALSE`: ``veritas.TRUE_DOMAIN`` and ``veritas.FALSE_DOMAIN``.

.. code-block:: sh
	:caption: Output

	Node(id=0, split=[F0 < 2], sz=7, left=1, right=2)
	├─ Node(id=1, split=[F0 < 1], sz=3, left=3, right=4)
	│  ├─ Leaf(id=3, sz=1, value=[3])
	│  └─ Leaf(id=4, sz=1, value=[4])
	├─ Node(id=2, split=[F0 < 3], sz=3, left=5, right=6)
	│  ├─ Leaf(id=5, sz=1, value=[5])
	│  └─ Leaf(id=6, sz=1, value=[6])
	
	Node(id=0, split=[F0 < 3], sz=9, left=1, right=2)
	├─ Node(id=1, split=[F1 < 5], sz=3, left=3, right=4)
	│  ├─ Leaf(id=3, sz=1, value=[30])
	│  └─ Leaf(id=4, sz=1, value=[40])
	├─ Node(id=2, split=[F1 < 0], sz=5, left=5, right=6)
	│  ├─ Leaf(id=5, sz=1, value=[50])
	│  ├─ Node(id=6, split=[F2 < 0.5], sz=3, left=7, right=8)
	│  │  ├─ Leaf(id=7, sz=1, value=[70])
	│  │  └─ Leaf(id=8, sz=1, value=[80])
	
	Eval: [[33.]
	 [56.]]
	

Model Conversion implementation
-------------------------------

Converting representations of other learners or your own models should be easy and can be done by implementing the class ``AddTreeConverter``.
In the following example ``MyAddTreeConverter`` implements the ``get_addtree`` method from ``AddTreeConverter`` for a trivial tree representation. The trees consist of a boolean split in the root with only 2 leaves. After adding an instance of ``MyAddTreeConverter`` to the convertermanager, the same method ``get_addtree`` that was used in the previous example can be used for the new model representation aswell as the previously methoned ones.

.. code-block:: py
	:caption: Python
	:linenos:

	# Trivial Tree representation
	#
	#         F0             F0        
	#       /    \    +    /    \              
	#      10    20       12    13
	
	myModel = [[10,20,0],[12,13,0]] # [left leaf, Right leaf, Boolean Feature]
	
	class MyAddTreeConverter(AddTreeConverter):
	    def get_addtree(self,model):
	        # Implement AddTreeConverter using your own model
	        addtree = AddTree(1)
	        
	        for tree in model:
	            t = addtree.add_tree()
	            t.split(t.root(),1)
	            t.set_leaf_value(t.left(t.root()),tree[0])
	            t.set_leaf_value(t.right(t.root()),tree[1])
	
	        return addtree
	
	
	# Add converter instance to the converter_manager 
	add_addtree_converter(MyAddTreeConverter())
	
	# Use get_addtree() on your own models
	addtree = get_addtree(myModel)
	
	print(f"{addtree}\n")
	
	print(addtree[0])
	print(addtree[1])
	

This has the expected output:

.. code-block:: sh
	:caption: Output

	AddTree with 2 trees and base_scores [0]
	
	Node(id=0, split=[F1 < 0.5], sz=3, left=1, right=2)
	├─ Leaf(id=1, sz=1, value=[10])
	└─ Leaf(id=2, sz=1, value=[20])
	
	Node(id=0, split=[F1 < 0.5], sz=3, left=1, right=2)
	├─ Leaf(id=1, sz=1, value=[12])
	└─ Leaf(id=2, sz=1, value=[13])
	
	
