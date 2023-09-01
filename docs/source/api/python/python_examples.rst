Python Examples
---------------

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
