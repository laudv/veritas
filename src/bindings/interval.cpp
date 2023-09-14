#include "bindings.h"
#include "interval.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace veritas;

void init_interval(py::module& m) {
    py::class_<Interval>(m, "Interval", R"pbdoc(
        Interval class

        )pbdoc")
        .def(py::init<>())
        .def(py::init<FloatT, FloatT>())
        .def_static("from_lo", &Interval::from_lo)
        .def_static("from_hi", &Interval::from_hi)
        .def_readwrite("lo", &Interval::lo)
        .def_readwrite("hi", &Interval::hi)
        .def("lo_is_unbound", &Interval::lo_is_unbound)
        .def("hi_is_unbound", &Interval::hi_is_unbound)
        .def("contains", &Interval::contains)
        .def("overlaps", &Interval::overlaps)
        .def("intersect", &Interval::intersect)
        .def("is_everything", &Interval::is_everything)
        .def("split", &Interval::split)
        .def("__eq__", [](const Interval& s, const Interval& t) { return s == t; })
        .def("__repr__", [](const Interval& d) { return tostr(d); })
        .def("__iter__", [](const Interval& d) { return py::iter(py::make_tuple(d.lo, d.hi)); })
        .def(py::pickle(
            [](const Interval& d) { return py::make_tuple(d.lo, d.hi); }, // __getstate__
            [](py::tuple t) { // __setstate__
                if (t.size() != 2) throw std::runtime_error("invalid pickle state");
                return Interval(t[0].cast<FloatT>(), t[1].cast<FloatT>());
            }))
        ; // Interval

    m.attr("BOOL_SPLIT_VALUE") = BOOL_SPLIT_VALUE;
    m.attr("TRUE_DOMAIN") = TRUE_DOMAIN;
    m.attr("FALSE_DOMAIN") = FALSE_DOMAIN;

    py::class_<LtSplit>(m, "LtSplit", R"pbdoc(
        LtSplit class

        )pbdoc")
        .def(py::init<FeatId, FloatT>())
        .def_readonly("feat_id", &LtSplit::feat_id)
        .def_readonly("split_value", &LtSplit::split_value)
        .def("test", [](const LtSplit& s, FloatT v) { return s.test(v); })
        .def("__eq__", [](const LtSplit& s, const LtSplit& t) { return s == t; })
        .def("__repr__", [](const LtSplit& s) { return tostr(s); })
        .def(py::pickle(
            [](const LtSplit& s) { return py::make_tuple(s.feat_id, s.split_value); }, // __getstate__
            [](py::tuple t) -> LtSplit { // __setstate__
                if (t.size() != 2) throw std::runtime_error("invalid pickle state");
                return { t[0].cast<FeatId>(), t[1].cast<FloatT>() };
            }))
        ; // LtSplit
}
