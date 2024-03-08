#include "bindings.h"
#include "addtree.hpp"
#include "json_io.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>


namespace py = pybind11;
using namespace veritas;

void init_addtree(py::module &m) {
    py::enum_<AddTreeType>(m, "AddTreeType")
        .value("REGR", AddTreeType::REGR)
        .value("REGR_MEAN", AddTreeType::REGR_MEAN)
        .value("CLF_MEAN", AddTreeType::CLF_MEAN)
        .value("CLF_SOFTMAX", AddTreeType::CLF_SOFTMAX)
        ; // AddTreeType

    py::class_<AddTree, std::shared_ptr<AddTree>>(m, "AddTree", R"pbdoc(
            Additive binary tree ensemble.

            :param num_leaf_values: Number of values in single leaf 
            :param AddTreeType: Optional AdTreeType
            :type AddTreeType: AddTreeType or None
        )pbdoc")
        .def(py::init<int>())
        .def(py::init<int, AddTreeType>())
        //.def(py::init<const AddTree&, size_t, size_t>())
        .def("get_base_score", [](const AddTree &at, int idx)
             { return at.base_score(idx); },
             "One constant base score for each leaf value")
        .def("set_base_score", [](AddTree &at, int idx, FloatT value)
            { at.base_score(idx) = value; })
        .def("copy", [](const AddTree &at)
            { return AddTree(at); })
        .def("__getitem__", [](const std::shared_ptr<AddTree> &at, size_t i)
            {
                if (i < at->size())
                    return TreeRef{at, i};
                throw py::value_error("out of bounds access into AddTree");
            })
        .def("__len__", &AddTree::size)
        .def("num_nodes", &AddTree::num_nodes, ":ref:`C++ API`")
        .def("num_leafs", &AddTree::num_leafs, ":ref:`C++ API`")
        .def("get_type", &AddTree::get_type)
        .def("num_leaf_values", &AddTree::num_leaf_values, ":ref:`C++ API`")
        .def("get_splits", &AddTree::get_splits, ":ref:`C++ API`")
        .def("get_maximum_feat_id", &AddTree::get_maximum_feat_id)
        .def("add_tree", [](const std::shared_ptr<AddTree> &at)
            { at->add_tree(); return TreeRef{at, at->size()-1}; })
        .def("add_tree", [](const std::shared_ptr<AddTree> &at, const TreeRef &tref)
            {
                at->add_tree(tref.get()); // copy
                return TreeRef{at, at->size()-1};
            })
        .def("add_trees", [](AddTree &at, const AddTree &other, std::optional<int> c)
            {
                if (c.has_value()) {
                    at.add_trees(other, c.value());
                } else {
                    at.add_trees(other);
                }
            },
            py::arg("other"), py::arg("c") = py::none())
        .def("make_multiclass", &AddTree::make_multiclass)
        .def("make_singleclass", &AddTree::make_singleclass)
        .def("contrast_classes", &AddTree::contrast_classes)
        .def("swap_class", &AddTree::swap_class)
        .def("prune", [](AddTree &at, const py::object &pybox)
            {
                Box::BufT buf = tobox(pybox);
                Box box{buf};
                return at.prune(BoxRef{box});
            })
        .def("neutralize_negative_leaf_values",
            &AddTree::neutralize_negative_leaf_values,
            ":ref:`C++ API`")
        .def("negate_leaf_values",
            &AddTree::negate_leaf_values,
            ":ref:`C++ API`")
        .def("concat_negated",
            &AddTree::concat_negated,
            ":ref:`C++ API`")
        .def("to_json", [](const AddTree &at)
            {
                std::stringstream ss;
                addtree_to_json(ss, at);
                return ss.str();
            })
        .def_static("from_json", [](const std::string &json)
            {
                std::stringstream s(json);
                return addtree_from_json<AddTree>(s);
            })
        .def_static("from_oldjson", [](const std::string &json)
            {
                std::stringstream s(json);
                return addtree_from_oldjson(s);
            })
        .def("predict", [](const AddTree& at, py::handle arr)
            {
                size_t min_num_cols = static_cast<size_t>(
                        at.get_maximum_feat_id()) + 1;
                data d = get_data(arr, min_num_cols);
                int nlv = at.num_leaf_values();

                py::array_t<FloatT, py::array::c_style | py::array::forcecast>
                    result(d.num_rows * nlv);
                result = result.reshape({ (long)d.num_rows, (long)nlv });

                data rdata = get_data(result, nlv);

                for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i) {
                    data rrow = rdata.row(i);
                    at.predict(d.row(i), rrow);
                }

                return result;
            }, R"pbdoc(
                Make prediction for the given examples.

                :param data: a numpy array of examples
            )pbdoc")
        .def("eval", [](const AddTree &at, py::handle arr)
            {
                size_t min_num_cols = static_cast<size_t>(
                        at.get_maximum_feat_id()) + 1;
                data d = get_data(arr, min_num_cols);
                int nlv = at.num_leaf_values();

                py::array_t<FloatT, py::array::c_style | py::array::forcecast>
                    result(d.num_rows * nlv);
                result = result.reshape({(long)d.num_rows, (long)nlv});

                data rdata = get_data(result, nlv);

                for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i) {
                    data rrow = rdata.row(i);
                    at.eval(d.row(i), rrow);
                }

                return result;
            })
        .def("compute_box", [](const AddTree &at, const std::vector<NodeId> &leaf_ids)
            {
                if (at.size() != leaf_ids.size())
                    throw std::runtime_error("one leaf_id per tree in AddTree");

                Box::BufT buf;
                Box box{buf};
                at.compute_box(box, leaf_ids);

                py::dict d;
                for (auto&& [feat_id, dom] : box)
                    d[py::int_(feat_id)] = dom;
                return d;
            })
        .def("__str__", [](const AddTree &at)
             { return tostr(at); })
        .def(py::pickle(
            [](const AddTree &at) { // __getstate__
                std::stringstream s;
                addtree_to_json(s, at);
                return s.str();
            },
            [](const std::string &json) { // __setstate__
                std::stringstream s(json);
                return addtree_from_json<AddTree>(s);
            }))
        ; // AddTree
}
