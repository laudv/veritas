## Prerequisites

```sh
apt-get install python3-sphinx
pip install sphinx-autoapi
pip install sphinx_rtd_theme
pip install breathe
apt-get install doxygen
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
|   ├── make.bat
|   └── Makefile
```

_HTML_ output is saved in `docs/build/html`.
Documentation flow and input should be modified in `docs/sphinx/source`.

## Build HTML

- `cd veritas/docs`
- `make html`
- Generated HTML in docs/build

## Help / inspiration

- [Sphinx](https://www.sphinx-doc.org/en/master/index.html)
- [autoapi](https://sphinx-autoapi.readthedocs.io/en/latest/index.html)
- [Documenting C++ Code with Sphinx](https://medium.com/@aytackahveci93/documenting-c-code-with-sphinx-d6315b338615)
- [C/C++ Documentation Using Sphinx](https://leimao.github.io/blog/CPP-Documentation-Using-Sphinx/)
