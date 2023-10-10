Installation
============

Dependencies
------------

* C++
* cmake
* pybind11 (included)
* python3 (numpy, optionally gurubipy)


Installation
------------

- Clone this repository: ``git clone https://github.com/laudv/veritas.git``
- Change directory: ``cd veritas``
- Initialize the `pybind11 <https://pybind11.readthedocs.io>`_ submodule ``git submodule init`` and ``git submodule update``
- If you use environments: activate a (new) Python3 environment (e.g. using ``venv``: ``python3 -m venv venv_name && source venv_name/bin/activate``)
- run ``pip install .`` in the root directory of Veritas

Veritas should work on Linux (GCC), Mac (LLVM), and Windows (MSVC). If you encounter issues, feel free to contact me or open an issue.

To pull the latest updates from Github, simply ``git pull` the changes and reinstall using`` ``pip``: ``pip install --force-reinstall .``.


Development
-----------

This is most likely not the proper way to use ``skbuild``, but that's how I have been doing it.
The editable install does not put the binary in the ``src/python`` folder, and it removes the build directory, so we use an editable ``pip`` install, and then manually invoke ``cmake`` to produce the shared ``veritas_core`` library.

.. code-block:: sh

   git clone git@github.com:laudv/veritas.git veritas
   cd veritas
   git submodule init && git submodule update
   pip install --editable .

   # this forgets to place the shared library in src/python/veritas, likely due to
   # wrong configuration but hey, doing it manually is faster than figuring that out...
   mkdir manual_build
   cd manual_build
   cmake ..
   make -j4

   # replace <...> with your shared library, name depends on platform
   ln -sfr <veritas_core.cpython-*.so> ../src/python/veritas/


Using this setup, you can change C++ code, rebuild, and then immediately restart Python.