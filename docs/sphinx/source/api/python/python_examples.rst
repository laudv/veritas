Python Examples
---------------

Constructing an Additive Tree Ensemble or `AddTree`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Veritas uses its own tree ensemble representation. You can manually build one to try Veritas out.

Here's an example of a manually constructed tree ensemble.
(To execute this code, see `tests/test_readme.py`.)

.. code-block:: py

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


This outputs the following. Note that the Boolean split on feature 2 is replaced with a less than split splitting on value 0.5 (``veritas.BOOL_SPLIT_VALUE``). You can use the pre-defined domains for ``TRUE`` and ``FALSE``: ``veritas.TRUE_DOMAIN`` and ``veritas.FALSE_DOMAIN``.

.. code-block:: sh
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

    Eval: [[33.] [56.]]


Model Conversion using XGBoost 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can also convert an existing ensemble using the ``get_addtree`` function for XGBoost, LightGBM and scikit-learn.
Here's an example of a model trained by XGBoost that has been converted to Veritas' own tree ensemble representation.

.. code-block:: py

    from sklearn.datasets import make_moons
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

The output is an ``AddTree`` consisting of 3 trees, as was defined in the `XGBClassifier`.

.. code-block:: sh

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


Model Conversion implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converting representations of other learners or your own models should be easy and can be done by implementing the class ``AddTreeConverter``. In the following example ``MyAddTreeConverter`` implements the ``get_addtree`` method from ``AddTreeConverter`` for a trivial tree representation. The trees consist of a boolean split in the root with only 2 leaves. After adding an instance of ``MyAddTreeConverter`` to the convertermanager, the same method ``get_addtree`` that was used in the previous example can be used for the new model representation aswell as the previously methoned ones.

.. code-block:: py

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

    AddTree with 2 trees and base_scores [0]

    Node(id=0, split=[F1 < 0.5], sz=3, left=1, right=2)
    ├─ Leaf(id=1, sz=1, value=[10])
    └─ Leaf(id=2, sz=1, value=[20])

    Node(id=0, split=[F1 < 0.5], sz=3, left=1, right=2)
    ├─ Leaf(id=1, sz=1, value=[12])
    └─ Leaf(id=2, sz=1, value=[13])


Finding the Global Maximum of the Ensemble
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use Veritas to find the feature values for which the model's output is maximal as follows.

.. code-block:: py

    # What is the maximum of the ensemble?
    config = Config(HeuristicType.MAX_OUTPUT)
    s = config.get_search(at,{})

    s.steps(100)

    print("Global maximum")
    if s.num_solutions() > 0:
        sol = s.get_solution(0)
        print("- current best solution:", sol.output, "->",
            "optimal" if s.is_optimal() else "suboptimal", "solution")
        print("- feature value ranges", sol.box())
        sol_nodes = s.get_solution_nodes(0)
        print("  which lead to leaf nodes", sol_nodes,
            "with leaf values",
            [at[i].get_leaf_value(n,0) for i, n in enumerate(sol_nodes)])


This outputs:

.. code-block:: sh

    Global maximum
    - current best solution: 86.0 -> optimal solution
    - feature value ranges {0: Interval(>=3), 1: Interval(>=0), 2: Interval(>=0.5)}
    which lead to leaf nodes [6, 8] with leaf values [6.0, 80.0]


The solutions generated by ``Search`` are accessible using ``get_solution``. The solutions are sorted descendingly: the best solution is at index 0, the worst solution is at index ``s.num_solutions()-1``.

A best solution at index 0 is optimal when ``s.is_optimal()`` returns true. To know when the solution was generated, ``sol.time`` contains the number of seconds since the construction of the ``Search`` object.

The ``sol.box()`` method returns the value intervals of the features for which the output of the ensemble is unchanged. That is, for each possible assignment within the intervals, the trees always evaluate to the same leaf node (``s.get_solution_nodes``), and thus to the same output value. If a feature is missing from the box, it means that its value does not make a difference.