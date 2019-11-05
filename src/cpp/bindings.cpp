#include <pybind11/pybind11.h>
#include "domain.h"

namespace py = pybind11;
using namespace treeck;

PYBIND11_MODULE(treeck, m) {
    m.doc() = "Tree-CK: verification of ensembles of trees";

    py::class_<RealDomain>(m, "RealDomain")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_readonly("lo", &RealDomain::lo)
        .def_readonly("hi", &RealDomain::hi)
        .def("contains", &RealDomain::contains)
        .def("overlaps", &RealDomain::overlaps)
        .def("split", &RealDomain::split);
}
