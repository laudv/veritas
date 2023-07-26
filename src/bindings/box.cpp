#include "bindings.h"
#include "box.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace veritas;

void init_box(py::module &m) {
    py::class_<IntervalPair>(m, "IntervalPair")
        .def_readonly("feat_id", &IntervalPair::feat_id)
        .def_readonly("interval", &IntervalPair::interval)
        ; // IntervalPair
}
