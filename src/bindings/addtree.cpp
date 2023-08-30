#include "bindings.h"
#include "addtree.hpp"
#include "json_io.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>


namespace py = pybind11;
using namespace veritas;

void init_addtree(py::module &m)
{
    py::class_<AddTree, std::shared_ptr<AddTree>>(m, "AddTree")
        .def(py::init<int>())
        .def(py::init<int, AddTreeType>())
        //.def(py::init<const AddTree&, size_t, size_t>())
        .def("get_base_score", [](const AddTree &at, int idx)
             { return at.base_score(idx); })
        .def("set_base_score", [](AddTree &at, int idx, FloatT value)
             { at.base_score(idx) = value; })
        .def("copy", [](const AddTree &at)
             { return AddTree(at); })
        .def("__getitem__", [](const std::shared_ptr<AddTree> &at, size_t i)
             {
            if (i < at->size())
                return TreeRef{at, i};
            throw py::value_error("out of bounds access into AddTree"); })
        .def("__len__", &AddTree::size)
        .def("num_nodes", &AddTree::num_nodes)
        .def("num_leafs", &AddTree::num_leafs)
        .def("num_leaf_values", &AddTree::num_leaf_values)
        .def("get_splits", &AddTree::get_splits)
        .def("add_tree", [](const std::shared_ptr<AddTree> &at)
             {
            at->add_tree(); return TreeRef{at, at->size()-1}; })
        .def("add_tree", [](const std::shared_ptr<AddTree> &at, const TreeRef &tref)
             {
            at->add_tree(tref.get()); // copy
            return TreeRef{at, at->size()-1}; })
        .def(
            "add_trees", [](AddTree &at, const AddTree &other, std::optional<int> c)
            {
            if (c.has_value()) {
                at.add_trees(other, c.value());
            } else {
                at.add_trees(other);
            } },
            py::arg("other"), py::arg("c") = py::none())
        .def("make_multiclass", &AddTree::make_multiclass)
        .def("make_singleclass", &AddTree::make_singleclass)
        .def("swap_class", &AddTree::swap_class)
        .def("prune", [](AddTree &at, const py::object &pybox)
             {
            Box::BufT buf = tobox(pybox);
            Box box{buf};
            return at.prune(BoxRef{box}); })
        .def("neutralize_negative_leaf_values", &AddTree::neutralize_negative_leaf_values)
        .def("negate_leaf_values", &AddTree::negate_leaf_values)
        .def("concat_negated", &AddTree::concat_negated)
        .def("to_json", [](const AddTree &at)
             {
            std::stringstream ss;
            addtree_to_json(ss, at);
            return ss.str(); })
        .def_static("from_json", [](const std::string &json)
                    {
            std::stringstream s(json);
            return addtree_from_json<AddTree>(s); })
        .def_static("from_oldjson", [](const std::string &json)
                    {
            std::stringstream s(json);
            return addtree_from_oldjson(s); })
        .def("predict", [](const AddTree& at, py::handle arr)
        {
            data d = get_data(arr);
            int nlv = at.num_leaf_values();

            py::array_t<FloatT, py::array::c_style | py::array::forcecast>
                result(d.num_rows * nlv);
            result = result.reshape({ (long)d.num_rows, (long)nlv });

            data rdata = get_data(result);

            AddTreeType type_ = at.get_type();
            using flags = std::underlying_type_t<AddTreeType>;
            flags type_flags = static_cast<flags>(type_);

            for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i) 
            {
                data rrow = rdata.row(i);
                at.eval(d.row(i), rrow);

                if(type_flags & static_cast<flags>(AddTreeType::RF)) 
                {
                    for(int j = 0 ; j < nlv ; ++j) rrow[j] = rrow[j]/at.size();
                    continue;
                }
                if(type_flags & static_cast<flags>(AddTreeType::MULTI))
                {
                    if(nlv > 1)
                    {
                        float e = 0;
                        for(int j = 0 ; j < nlv ; ++j) e += exp(rrow[j]);

                        for(int j = 0 ; j < nlv ; ++j) rrow[j] = exp(rrow[j]) / e;

                    } else throw std::runtime_error("Cannot predict multiclass probability vector on single leaf addtrees");
                } else if(type_flags & static_cast<flags>(AddTreeType::CLF))
                {
                    rrow[0] = 1 / (1 + exp(-rrow[0])); // Sigmoid: alternative: erf(sqrt(pi)*x/2) or tanh(x)?
                } 
            }
            return result;
        }
        )
        .def("eval", [](const AddTree &at, py::handle arr)
             {
            data d = get_data(arr);
            int nlv = at.num_leaf_values();

            py::array_t<FloatT, py::array::c_style | py::array::forcecast>
                result(d.num_rows * nlv);
            result = result.reshape({(long)d.num_rows, (long)nlv});

            data rdata = get_data(result);

            for (size_t i = 0; i < static_cast<size_t>(d.num_rows); ++i) {
                data rrow = rdata.row(i);
                at.eval(d.row(i), rrow);
            }

            return result; })
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
            return d; })
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
            })); // AddTree

    py::enum_<AddTreeType>(m, "AddTreeType")
        .value("RAW", AddTreeType::RAW)
        .value("RF_REGR", AddTreeType::RF_REGR)
        .value("RF_CLF", AddTreeType::RF_CLF)
        .value("RF_MULTI", AddTreeType::RF_MULTI)
        .value("GB_REGR", AddTreeType::GB_REGR)
        .value("GB_CLF", AddTreeType::GB_CLF)
        .value("GB_MULTI", AddTreeType::GB_MULTI); // AddTreeType
}
