

class AddTreeConverter:
    """ AddTreeConverter Base Interface

    Interface that gives the opportunity to implement a conversion from one's own model to Veritas' represention of tree ensembles.
    The function to implement is ``get_addtree(model)``. The converter then needs to be added to the convertermanager using ``add_addtree_converter()``.
    For an example see: ######
    
    """
    def get_addtree(self, model):
        """ Needs implementation of the user
         
        :param model: model
        """
        raise NotImplementedError
    
    def test_conversion(self, model):
        """
        :meta private:
        """
        pass