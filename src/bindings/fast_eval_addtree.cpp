#include "bindings.h"
#include "fast_eval_addtree.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
using namespace veritas;

void init_fast_eval_addtree(py::module &m) {
    py::class_<FastEvalAddTree>(m, "FastEvalAddTree", R"pbdoc(
            Read-only snapshot of an AddTree optimized for batch evaluation.

            The trees are linearized at construction time. Later modifications
            to the source AddTree are not reflected.
        )pbdoc")
        .def(py::init<const AddTree&>())
        .def("__len__", &FastEvalAddTree::size)
        .def("num_nodes", &FastEvalAddTree::num_nodes)
        .def("num_leaf_values", &FastEvalAddTree::num_leaf_values)
        .def("get_maximum_feat_id", &FastEvalAddTree::get_maximum_feat_id)
        .def("eval", [](const FastEvalAddTree &fat, py::handle arr)
            {
                data d = get_data(arr, static_cast<size_t>(fat.get_maximum_feat_id()) + 1);
                size_t nlv = static_cast<size_t>(fat.num_leaf_values());

                py::array_t<FloatT> result({d.num_rows, nlv});
                data<FloatT> rdata(result.mutable_data(), d.num_rows, nlv, nlv, 1);

                {
                    py::gil_scoped_release release;
                    fat.eval(d, rdata);
                }
                return result;
            })
        .def("eval_node", [](const FastEvalAddTree &fat, py::handle arr)
            {
                data d = get_data(arr, static_cast<size_t>(fat.get_maximum_feat_id()) + 1);
                size_t ntrees = fat.size();

                py::array_t<NodeId> result({d.num_rows, ntrees});
                data<NodeId> rdata(result.mutable_data(), d.num_rows, ntrees, ntrees, 1);

                {
                    py::gil_scoped_release release;
                    fat.eval_node(d, rdata);
                }
                return result;
            })
        ; // FastEvalAddTree
}
