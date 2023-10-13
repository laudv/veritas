/**
 * \file json_io_nlohmann.cpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "addtree.hpp"
#include "basics.hpp"
#include "interval.hpp"
#include "json_io.hpp"
#include "tree.hpp"

#include <iostream>
#include <istream>
#include <sstream>

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace veritas {

using json = nlohmann::json;

namespace json_detail {

template <typename ValueT>
struct TreeTypeName {};

template <>
struct TreeTypeName<FloatT> {
    const char *name() const { return "FloatT"; }
};

template <>
struct TreeTypeName<FpT> {
    const char *name() const { return "FpT"; }
};

template <>
struct TreeTypeName<LtSplit> {
    const char *name() const { return "LtSplit"; }
};

template <>
struct TreeTypeName<LtSplitFp> {
    const char *name() const { return "LtSplitFp"; }
};

template <>
struct TreeTypeName<std::string> {
    const char *name() const { return "std::string"; }
};

template <typename SplitT>
struct SplitEncDec {};

template <>
struct SplitEncDec<LtSplit> {
    void encode(json& j, const LtSplit& s) const {
        j["feat_id"] = s.feat_id;
        j["split_value"] = s.split_value;
    }

    LtSplit decode(json j) const {
        FeatId fid = j["feat_id"];
        FloatT split_value = j["split_value"];
        return { fid, split_value };
    }
};

template <>
struct SplitEncDec<LtSplitFp> {
    void encode(json& j, const LtSplitFp& s) const {
        j["feat_id"] = s.feat_id;
        j["split_value"] = s.split_value;
    }

    LtSplitFp decode(json j) const {
        std::cout << "decode " << j << std::endl;
        FeatId fid = j["feat_id"];
        FpT split_value = j["split_value"];
        return { fid, split_value };
    }
};

} // namespace json_detail

template <typename TreeT>
static json tree_to_nlohmann_json(const TreeT& t, NodeId n) {
    json j;
    if (t.is_leaf(n)) {
        j["leaf_value"] = json::array();
        for (int i = 0; i < t.num_leaf_values(); ++i)
            j["leaf_value"].push_back(t.leaf_value(n, i));
    }
    else {
        json_detail::SplitEncDec<typename TreeT::SplitType>()
            .encode(j, t.get_split(n));
        j["left"] = tree_to_nlohmann_json(t, t.left(n));
        j["right"] = tree_to_nlohmann_json(t, t.right(n));
    }
    return j;
}

template <typename TreeT>
static void tree_from_nlohmann_json(const json& j, TreeT& t, NodeId n) {
    using SplitEncDec = json_detail::SplitEncDec<typename TreeT::SplitType>;

    if (j.contains("leaf_value")) {
        for (int i = 0; i < t.num_leaf_values(); ++i)
            t.leaf_value(n, i) = j["leaf_value"][i];
    }
    else {
        t.split(n, SplitEncDec().decode(j));
        tree_from_nlohmann_json(j["left"], t, t.left(n));
        tree_from_nlohmann_json(j["right"], t, t.right(n));
    }
}

template <typename TreeT>
static TreeT tree_from_nlohmann_json(const json& j) {
    using SplitTypeName = json_detail::TreeTypeName<typename TreeT::SplitType>;
    using ValueTypeName = json_detail::TreeTypeName<typename TreeT::LeafValueType>;

    if (SplitTypeName().name() != j["split_type"])
        throw std::runtime_error("invalid split_type");
    if (ValueTypeName().name() != j["value_type"])
        throw std::runtime_error("invalid value_type");

    int num_leaf_values = j["num_leaf_values"];
    TreeT t(num_leaf_values);

    tree_from_nlohmann_json(j["structure"], t, t.root());

    return t;
}


template <typename TreeT>
TreeT tree_from_json(std::istream& s) {
    json j = json::parse(s);
    return tree_from_nlohmann_json<TreeT>(j);
}

template Tree tree_from_json(std::istream& s);
template TreeFp tree_from_json(std::istream& s);
template GTree<LtSplit, std::string> tree_from_json(std::istream& s);

template <typename TreeT>
static json tree_to_nlohmann_json(const TreeT& t) {
    using SplitTypeName = json_detail::TreeTypeName<typename TreeT::SplitType>;
    using ValueTypeName = json_detail::TreeTypeName<typename TreeT::LeafValueType>;

    json t_json;
    t_json["num_leaf_values"] = t.num_leaf_values();
    t_json["split_type"] = SplitTypeName().name();
    t_json["value_type"] = ValueTypeName().name();
    t_json["structure"] = tree_to_nlohmann_json(t, t.root());

    return t_json;
}

template <typename SplitT, typename ValueT>
void tree_to_json(std::ostream& s, const GTree<SplitT, ValueT>& t) {
    s << tree_to_nlohmann_json(t);
}

template void tree_to_json(std::ostream& s, const Tree& t);
template void tree_to_json(std::ostream& s, const TreeFp& t);
template void tree_to_json(std::ostream& s, const GTree<LtSplit, std::string>& t);

template <typename TreeT>
void addtree_to_json(std::ostream& s, const GAddTree<TreeT>& at) {
    json at_json;

    at_json["at_type"] = addtree_type_to_str(at.get_type());
    at_json["base_scores"] = json::array();
    for (int i = 0; i < at.num_leaf_values(); ++i)
        at_json["base_scores"].push_back(at.base_score(i));

    json trees_json = json::array();
    for (size_t m = 0; m < at.size(); ++m) {
        trees_json.push_back(tree_to_nlohmann_json(at[m]));
    }
    at_json["trees"] = std::move(trees_json);

    s << at_json;
}

template void addtree_to_json(std::ostream& s, const AddTree& t);
template void addtree_to_json(std::ostream& s, const AddTreeFp& t);

template <typename AddTreeT>
AddTreeT addtree_from_json(std::istream& s) {
    using TreeT = typename AddTreeT::TreeType;

    json at_json = json::parse(s);
    int num_leaf_values = static_cast<int>(at_json["base_scores"].size());
    AddTreeType type = addtree_type_from_str(at_json["at_type"]);

    AddTreeT at(num_leaf_values, type);

    for (int i = 0; i < at.num_leaf_values(); ++i)
        at.base_score(i) = at_json["base_scores"][i];

    for (size_t m = 0; m < at_json["trees"].size(); ++m) {
        at.add_tree(tree_from_nlohmann_json<TreeT>(at_json["trees"][m]));
    }

    return at;
}

template AddTree addtree_from_json(std::istream& s);
template AddTreeFp addtree_from_json(std::istream& s);

} // namespace veritas
