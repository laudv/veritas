#include "bindings.h"
#include "features.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace veritas;

void init_featmap(py::module &m) {
    py::class_<FeatMap>(m, "FeatMap")
        .def(py::init<FeatId>())
        .def(py::init<const std::vector<std::string>&>())
        .def("num_features", &FeatMap::num_features)
        .def("__len__", &FeatMap::num_features)
        .def("get_index", [](const FeatMap& fm, FeatId id, int inst) { return fm.get_index(id, inst); })
        .def("get_index", [](const FeatMap& fm, const std::string& n, int inst) { return fm.get_index(n, inst); })
        .def("get_instance", &FeatMap::get_instance)
        .def("get_name", &FeatMap::get_name)
        .def("get_indices_map", [](const FeatMap& fm) {
            auto map = fm.get_indices_map();
            py::object d = py::dict();
            for (auto&& [id, idx] : map)
            {
                if (d.contains(py::int_(id)))
                {
                    py::list l = d[py::int_(id)];
                    l.append(py::int_(idx));
                }
                else
                {
                    py::list l;
                    l.append(py::int_(idx));
                    d[py::int_(id)] = l;
                }
            }
            return d;
        })
        .def("share_all_features_between_instances", &FeatMap::share_all_features_between_instances)
        .def("get_feat_id", [](const FeatMap& fm, FeatId id) { return fm.get_feat_id(id); })
        .def("get_feat_id", [](const FeatMap& fm, const std::string& n, int i) { return fm.get_feat_id(n, i); })
        .def("use_same_id_for", &FeatMap::use_same_id_for)
        .def("transform", [](const FeatMap& fm, const AddTree& at, int i) { return fm.transform(at, i); })
        .def("__iter__", [](const FeatMap &fm) { return py::make_iterator(fm.begin(), fm.end()); },
                    py::keep_alive<0, 1>())
        .def("iter_instance", [](const FeatMap& fm, int i) {
                auto h = fm.iter_instance(i);
                return py::make_iterator(h.begin(), h.end()); },
                py::keep_alive<0, 1>())
        .def("__str__", [](const FeatMap& fm) { return tostr(fm); })
        ; // FeatMap
}
