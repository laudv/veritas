## Prerequisites

```sh
$ sudo apt-get install python3-sphinx
$ pip install sphinx-autoapi
$ pip install sphinx_rtd_theme
$ pip install breathe
$ sudo apt-get install doxygen
```

## Files

```
veritas
├── docs
│   ├── build
|	|   ├── doctrees
|	|   └── html
│   ├── doxybin
│   |   ├── build/xml
|   |   |   └── ...
│   |   └── Doxyfile
|   ├── sphinx/source
│   |   ├── _templates
|   |   |   └── ...
│   |   ├── api
|   |   |   └── ...
│   |   ├── usage
|   |   |   └── ...
|   |   ├── conf.py
|   |   └── index.rst
|   ├── generate_python_examples.py
|   ├── make.bat
|   └── Makefile
```

_HTML_ output is saved in `docs/build/html`.
Documentation flow and input should be modified in `docs/sphinx/source`.

## Generate Python examples .rst

Same principle and code as `generate_readme.py` but for rst. Run `generate_python_examples.py` before makefile when the template or code is updated.

- Template: `python_examples_template.rst`
- Code: `python_examples.py`
- Output (used in index): `python_examples.rst`

## Build HTML

- `cd veritas/docs`
- `make html`
- Generated HTML in docs/build

## Documenting

### Pure Python

Pure python files like `addtree_converter.py` are documentend using `autodoc` in [Sphinx](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html)

```python
class AddTreeConverter:
    """ AddTreeConverter Base Interface

    Text text text

    """
    def get_addtree(self, model):
        """ Needs implementation of the user

        :param model: model
        """
        raise NotImplementedError
```

### Pybind

Pybind uses its own docstring implementation but inside just use Sphinx rst.

```cpp
py::class_<AddTree, std::shared_ptr<AddTree>>(m, "AddTree", R"pbdoc(
        AddTree Class
        -------------

        :param num_leaf_values: Number of values in single leaf
        :param AddTreeType: Optional AdTreeType
        :type AddTreeType: AddTreeType or None
    )pbdoc")
```

Autosummary and templates are used for the module `veritas_core` to keep it clean.
The pybind doc is then called using `.. automodule:: src.python.veritas.veritas_core`

```
    m.doc() = R"pbdoc(

        Basic
        ~~~~~

        .. autosummary::
            :toctree: pybind_tree_classes
            :template: template.rst

            Tree
            AddTree
    )pbdoc";
```

### C++

Documenting C++ code is done with regular [Doxygen](https://www.doxygen.nl/manual/docblocks.html).

```cpp
/**
 * @brief Create a new AddTree instance
 * @param nleaf_values The number of values in a single leaf
 * @param type_ Type of AddTree @see veritas::AddTreeType
 *
 *  Create an empty AddTree. When an AddTreeType is not specified, the AddTree will have the `AddTreeType::RAW`
*/
inline GAddTree(int nleaf_values, AddTreeType type = AddTreeType::RAW)
    : trees_(), base_scores_(nleaf_values, {}), type_(type)
{
}
```

## Help / inspiration

- [Sphinx](https://www.sphinx-doc.org/en/master/index.html)
- [autoapi (not used, skips pybindings)](https://sphinx-autoapi.readthedocs.io/en/latest/index.html)
- [Documenting C++ Code with Sphinx](https://medium.com/@aytackahveci93/documenting-c-code-with-sphinx-d6315b338615)
- [C/C++ Documentation Using Sphinx](https://leimao.github.io/blog/CPP-Documentation-Using-Sphinx/)
- [Generating documentation using Sphinx](https://pybind11.readthedocs.io/en/stable/advanced/misc.html?highlight=sphinx#generating-documentation-using-sphinx)
- [Pybind Sphinx Example](https://github.com/pybind/python_example/blob/master/src/main.cpp)
