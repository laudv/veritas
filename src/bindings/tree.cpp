#include "bindings.h"
#include "tree.hpp"
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace py = pybind11;
using namespace veritas;

void init_tree(py::module &m) {
    py::class_<TreeRef>(m, "Tree", R"pbdoc(
        Tree class

        )pbdoc")
        .def("root", [](const TreeRef& r) { return r.get().root(); })
        .def("num_leaves", [](const TreeRef& r) { return r.get().num_leaves(); })
        .def("num_nodes", [](const TreeRef& r) { return r.get().num_nodes(); })
        .def("num_leaf_values", [](const TreeRef& r) { return r.get().num_leaf_values(); })
        .def("is_root", [](const TreeRef& r, NodeId n) { return r.get().is_root(n); })
        .def("is_leaf", [](const TreeRef& r, NodeId n) { return r.get().is_leaf(n); })
        .def("is_internal", [](const TreeRef& r, NodeId n) { return r.get().is_internal(n); })
        .def("left", [](const TreeRef& r, NodeId n) { return r.get().left(n); })
        .def("right", [](const TreeRef& r, NodeId n) { return r.get().right(n); })
        .def("parent", [](const TreeRef& r, NodeId n) { return r.get().parent(n); })
        .def("tree_size", [](const TreeRef& r, NodeId n) { return r.get().tree_size(n); })
        .def("depth", [](const TreeRef& r, NodeId n) { return r.get().depth(n); })
        .def("get_leaf_value", [](const TreeRef& r, NodeId n, int i) {
            return r.get().leaf_value(n, i);
        })
        .def("get_leaf_values", [](const TreeRef& r, NodeId n) {
            py::array_t<FloatT> arr(r.get().num_leaf_values());
            for (int i = 0; i < r.get().num_leaf_values(); ++i)
                arr.mutable_at(i) = r.get().leaf_value(n, i);
            return arr;
        })
        .def("set_leaf_value", [](TreeRef& r, NodeId n, int i, FloatT v) {
            r.get().leaf_value(n, i) = v;
        })
        .def("set_leaf_values", [](TreeRef& r, NodeId n, py::handle values) {
            data vs = get_data(values);
            Tree& t = r.get();
            if (vs.num_rows * vs.num_cols != static_cast<size_t>(t.num_leaf_values()))
                throw std::invalid_argument("wrong number of leaf values");
            for (int i = 0; i < t.num_leaf_values(); ++i)
                t.leaf_value(n, i) = vs[i];
        })
        .def("set_leaf_value", [](TreeRef& r, NodeId n, FloatT v) {
                if(r.get().num_leaf_values() == 1) r.get().leaf_value(n, 0) = v;
                else throw std::runtime_error("Specify leaf value index for tree with multiple leaf values");
        })
        .def("get_split", [](const TreeRef& r, NodeId n) { return r.get().get_split(n); })
        .def("find_minmax_leaf_value", [](const TreeRef& r, NodeId n) {
            std::vector<std::pair<FloatT, FloatT>> buf(r.get().num_leaf_values());
            r.get().find_minmax_leaf_value(n, buf);
            return buf;
        })
        .def("get_leaf_ids", [](const TreeRef& r) { return r.get().get_leaf_ids(); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid, FloatT sv) { r.get().split(n, {fid, sv}); })
        .def("split", [](TreeRef& r, NodeId n, FeatId fid) { r.get().split(n, bool_ltsplit(fid)); })
        .def("eval", [](const TreeRef& r, py::handle arr, NodeId nid) {
            int nlv = r.get().num_leaf_values();
            size_t min_num_cols = static_cast<size_t>(
                    r.get().get_maximum_feat_id(r.get().root())) + 1;

            data d = get_data(arr, min_num_cols);

            py::array_t<FloatT, py::array::c_style | py::array::forcecast>
                result(d.num_rows * nlv);
            result = result.reshape({(long)d.num_rows, (long)nlv});

            data rdata = get_data(result, nlv);

            for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i) {
                data rrow = rdata.row(i);
                for (int i = 0; i < nlv; ++i)
                    rrow[i] = 0.0;
                r.get().eval(nid, d.row(i), rrow);
            }

            return result;
        })
        .def("eval_node", [](const TreeRef& r, py::handle arr, NodeId nid) {
            size_t min_num_cols = static_cast<size_t>(
                    r.get().get_maximum_feat_id(r.get().root())) + 1;
            data d = get_data(arr, min_num_cols);

            auto result = py::array_t<NodeId>(d.num_rows);
            py::buffer_info out = result.request();
            NodeId *out_ptr = static_cast<NodeId *>(out.ptr);

            for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i)
                out_ptr[i] = r.get().eval_node(nid, d.row(i));

            return result;
        })

        .def("__str__", [](const TreeRef& r) { return tostr(r.get()); })
        .def("compute_box", [](const TreeRef& r, NodeId n) {
            Box::BufT buf;
            Box box{buf};
            r.get().compute_box(n, box);
            py::dict d;
            for (auto&& [feat_id, dom] : box)
                d[py::int_(feat_id)] = dom;
            return d;
        })
        .def("get_maximum_feat_id", [](const TreeRef& r)
            { return r.get().get_maximum_feat_id(r.get().root()); })
        .def("prune", [](const TreeRef& r, const py::object& pybox) {
            Box::BufT buf = tobox(pybox);
            Box box{buf};
            AddTree at(r.get().num_leaf_values());
            at.add_tree(r.get().prune(BoxRef{box}));
            return at;
        })
        .def("__getitem__", [](const TreeRef& r, const py::str& str) {
            std::string s = str;
            return r.get()[s.c_str()];
        })
        ; // TreeRef
}
