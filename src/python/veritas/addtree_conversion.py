# Copyright 2023 DTAI Research Group - KU Leuven.
# License: Apache License 2.0
# Authors: Alexander Schoeters, Laurens Devos

from . import AddTree

import numpy as np

class InapplicableAddTreeConverter(Exception):
    pass

class NoRegisteredConverterException(Exception):
    pass

class AddTreeConverter:
    """AddTreeConverter Base Interface

    Interface that gives the opportunity to implement a conversion from one's
    own model to Veritas' represention of tree ensembles.

    The function to implement is ``convert(model, silent)``. The converter then
    needs to be added to the convertermanager using ``add_addtree_converter()``.
    """

    def convert(self, model, silent):
        """ Convert the given model to an `AddTree`

        This method throws an `InapplicableAddTreeConverter` if the given model
        is not of the right type.
        """
        raise NotImplementedError()


class AddTreeConverterRegistry:
    def __init__(self):
        self._converters = []

    def add_converter(self, converter):
        assert isinstance(converter, AddTreeConverter)

        # Prepend to the front so this new converter takes precedence
        self._converters.insert(0, converter)

    def get_addtree(self, model, silent):
        for converter in self._converters:
            try:
                addtree = converter.convert(model, silent)
                assert isinstance(addtree, AddTree)
                return addtree
            except InapplicableAddTreeConverter:
                pass

        raise NoRegisteredConverterException(
                f"No conversion possible for model of type `{type(model)}`. "
                "Implement an `veritas.AddTreeConverter` and add it using "
                "`veritas.add_addtree_converter`.")

_converter_registry = AddTreeConverterRegistry()

def add_addtree_converter(converter):
    """
    Adds an instance of AddTreeConverter to the `converter_manager`. The
    `get_addtree` function will now also use the added converter.

    :param converter: Instance of AddTreeConverter
    :type converter: AddTreeConverter
    """
    _converter_registry.add_converter(converter)

def get_addtree(model, silent=False):
    """Convert the given model to a Veritas `AddTree`.

    This will try each registered `AddTreeConverter` known to Veritas. There
    are default converters for XGBoost, LightGBM, and scikit-learn random
    forests.

    Add your own custom converters by implementing a `veritas.AddTreeConverter`
    and adding it to the registry using `veritas.add_addtree_converter`.

    If no converter is registered for the given `model`, this function will
    throw a `NoRegisteredConverterException` exception.

    :param model: model that needs to be converted to a Veritas tree ensemble
    :rtype: AddTree
    """
    return _converter_registry.get_addtree(model, silent=silent)

def test_conversion(at, X, ypred_model, single_rel_tol=1e-5, silent=False):
    """Test the conversion of a model to a Veritas `AddTree`

    Test whether the outputs of `AddTree` generated by `get_addtree` match
    the original models outputs.
    """
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
    except ModuleNotFoundError:
        pass

    is_correct = True
    at_pred = at.predict(X)
    ypred_model = ypred_model.reshape(at_pred.shape)
    rel_tol = single_rel_tol * len(at)

    if not np.all(np.isclose(at_pred, ypred_model, rtol=rel_tol)):
        print(f"test_conversion: problem detected! (rel_tol {rel_tol:g})")
        for i in range(X.shape[0]):
            if not np.all(np.isclose(at_pred[i], ypred_model[i], rtol=rel_tol)):
                err = np.abs((at_pred[i]-ypred_model[i]))
                print(f"┌ example {i:<6d}  at prediction: {at_pred[i]}")
                print(f"│              model prediction: {ypred_model[i]}")
                print( "│               abs / rel error:",
                      f"{err} / {err/ypred_model[i]}")
                is_correct = False
                is_split_float_error(at, X[i, :], rel_tol)
    elif not silent:
        print(f"test_conversion: no problems detected (rel_tol {rel_tol:g})")

    return is_correct

def is_split_float_error(at, x, rel_tol):
    for m, tree in enumerate(at):
        n = tree.eval_node(x)[0]
        leaf_values = tree.get_leaf_values(n)

        while True:
            if tree.is_internal(n):
                split = tree.get_split(n)
                if np.isclose(x[split.feat_id], split.split_value, rtol=rel_tol):
                    print(f"│   tree {m:<3d} node {n:<3d} `{split}`",
                          f"F{split.feat_id}={x[split.feat_id]}",
                          "(diff",
                          f"{np.abs((x[split.feat_id]-split.split_value)/x[split.feat_id])},",
                          f"leaf_values {leaf_values})")
            if not tree.is_root(n):
                n = tree.parent(n)
            else:
                break
