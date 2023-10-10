Custom Trees
============

Constructing an Additive Tree Ensemble or `AddTree`
---------------------------------------------------

Veritas uses its own tree ensemble representation. You can manually build one to try Veritas out.
Here's an example of a manually constructed tree ensemble.

!code PART example_at!

This outputs the following. Note that the Boolean split on feature 2 is replaced with a less than split splitting on value 0.5 (``veritas.BOOL_SPLIT_VALUE``). You can use the pre-defined domains for `TRUE` and `FALSE`: ``veritas.TRUE_DOMAIN`` and ``veritas.FALSE_DOMAIN``.

!output PART example_at!

Model Conversion implementation
-------------------------------

Converting representations of other learners or your own models should be easy and can be done by implementing the class ``AddTreeConverter``.
In the following example ``MyAddTreeConverter`` implements the ``get_addtree`` method from ``AddTreeConverter`` for a trivial tree representation. The trees consist of a boolean split in the root with only 2 leaves. After adding an instance of ``MyAddTreeConverter`` to the convertermanager, the same method ``get_addtree`` that was used in the previous example can be used for the new model representation aswell as the previously methoned ones.

!code PART AddTreeConverter!

This has the expected output:

!output PART AddTreeConverter!