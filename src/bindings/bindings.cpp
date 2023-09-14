/*
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

#include "bindings.h"

namespace py = pybind11;
using namespace veritas;


PYBIND11_MODULE(veritas_core, m) {

    // redirect C++ output to Pythons stdout
    // https://github.com/pybind/pybind11/issues/1005
    // https://github.com/pybind/pybind11/pull/1009
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html#capturing-standard-output-from-ostream
    m.attr("_redirect_output") = py::capsule(
            new py::scoped_ostream_redirect(
                std::cout, py::module::import("sys").attr("stdout")),
            [](void *sor) { delete static_cast<py::scoped_ostream_redirect *>(sor); });

    m.doc() = R"pbdoc(
        Basic
        ~~~~~
        .. autosummary::
            :toctree: pybind_tree_classes
            :template: template.rst

            Tree
            AddTree
            AddTreeType
            Interval
            IntervalPair
            FeatMap

        Search
        ~~~~~~
        .. autosummary::
            :toctree: pybind_tree_search
            :template: template.rst

            Search
            Config
            StopReason
            HeuristicType
            Bounds
            Statistics
            Solution
            
    )pbdoc";

    init_interval(m);
    init_box(m);
    init_tree(m);
    init_addtree(m);
    init_featmap(m);
    init_search(m);
} /* PYBIND11_MODULE */
