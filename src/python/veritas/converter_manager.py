from . import AddTreeConverter

from .lgb import LGB_AddTreeConverter
from .xgb import XGB_AddTreeConverter
from .sklearn import Sk_AddTreeConverter

class ConverterManager:
    def __init__(self):
        self._converters = []

    def add_converter(self, converter):
        assert isinstance(converter, AddTreeConverter)
        self._converters.insert(0,converter)

    def get_addtree(self,model):
        for converter in self._converters:
            try:
                return converter.get_addtree(model)
            except:
                pass
        raise ConversionException("No conversion possible. Implement a converter using `AddTreeConverter` and add it using `add_addtree_converter`.")

class ConversionException(Exception):
    """No conversion is possbible. Either install XGBoost, SkLearn, LightGBM or add your own converter.
        :meta private:
    """

_conv_manager = ConverterManager()

def add_addtree_converter(converter):
    """
    Adds an instance of AddTreeConverter to the `converter_manager`. The `get_addtree` function will now also use the added converter.

    :param converter: Instance of AddTreeConverter
    :type converter: AddTreeConverter
    """
    _conv_manager.add_converter(converter)

def get_addtree(model):
    """
    Returns a veritas Addtree equivalent to the given model. 

    This works seamlessly on every model where there is an implementation for the class `AddTreeConverter`. 
    Currently XGBoost, LightGBM and scikit-learn are supported. For an example see :ref:`Trees Ensembles`.
    You can always implement your own model using the `AddTreeConverter` interface.

    :param model: model that needs to be converted to a Veritas tree ensemble

    :rtype: AddTree
    """
    return _conv_manager.get_addtree(model)


add_addtree_converter(XGB_AddTreeConverter())
add_addtree_converter(LGB_AddTreeConverter())
add_addtree_converter(Sk_AddTreeConverter())