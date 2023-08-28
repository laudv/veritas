from . import AddTreeConverter

from .extensions.lgb import LGB_AddTreeConverter
from .extensions.xgb import XGB_AddTreeConverter
from .extensions.sklearn import Sk_AddTreeConverter

class ConverterManager:
    def __init__(self):
        self._converters = []

    def add_converter(self, converter):
        assert isinstance(converter, AddTreeConverter)
        self._converters.insert(0,converter)

    def get_addtree(self,model):
        for converter in self._converters:
            try:
                mo = converter.get_addtree(model)
                return mo
            except Exception:
                pass

_conv_manager = ConverterManager()

def add_addtree_converter(converter):
    _conv_manager.add_converter(converter)

def get_addtree(model):
    return _conv_manager.get_addtree(model)

add_addtree_converter(XGB_AddTreeConverter())
add_addtree_converter(LGB_AddTreeConverter())
add_addtree_converter(Sk_AddTreeConverter())