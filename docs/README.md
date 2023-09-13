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

## Help / inspiration

- [Sphinx](https://www.sphinx-doc.org/en/master/index.html)
- [autoapi](https://sphinx-autoapi.readthedocs.io/en/latest/index.html)
- [Documenting C++ Code with Sphinx](https://medium.com/@aytackahveci93/documenting-c-code-with-sphinx-d6315b338615)
- [C/C++ Documentation Using Sphinx](https://leimao.github.io/blog/CPP-Documentation-Using-Sphinx/)
- [Generating documentation using Sphinx](https://pybind11.readthedocs.io/en/stable/advanced/misc.html?highlight=sphinx#generating-documentation-using-sphinx)
