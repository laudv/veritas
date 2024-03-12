/**
 * \file tree.cpp
 *
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "addtree.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace veritas {

std::size_t argmax(const std::vector<FloatT>& v) {
    if (v.empty()) { throw std::runtime_error("argmax of empty vector...?"); }

    auto max_value = std::max_element(v.begin(), v.end());
    return std::distance(v.begin(), max_value);
}

static uint8_t _to_uint8(AddTreeType a) {
    return static_cast<uint8_t>(a);
}

static bool _is_a(AddTreeType a, AddTreeType b) {
    return (_to_uint8(a) & _to_uint8(b)) == _to_uint8(b);
}

// static AddTreeType operator| (AddTreeType a, AddTreeType b) {
//     return static_cast<AddTreeType>(_to_uint8(a) | _to_uint8(b));
// }

// static AddTreeType operator& (AddTreeType a, AddTreeType b) {
//     return static_cast<AddTreeType>(_to_uint8(a) & _to_uint8(b));
// }

// static AddTreeType operator~ (AddTreeType a) {
//     return static_cast<AddTreeType>(~_to_uint8(a));
// }

// static AddTreeType _make_binary(AddTreeType a) {
//     if (a == AddTreeType::RAW)
//         return AddTreeType::RAW;
//     return (a & ~AddTreeType::MULTI) | AddTreeType::BINARY;
// }

// static AddTreeType _make_multi(AddTreeType a) {
//     if (a == AddTreeType::RAW)
//         return AddTreeType::RAW;
//     return (a & ~AddTreeType::BINARY) | AddTreeType::MULTI;
// }


const char *
addtree_type_to_str(AddTreeType t) {

#define VER_ADDTREE_TYPE_CASE(name) case AddTreeType::name: \
    return #name; \
    break

    switch (t) {
    VER_ADDTREE_TYPE_CASE(REGR);
    VER_ADDTREE_TYPE_CASE(CLF);

    VER_ADDTREE_TYPE_CASE(REGR_MEAN);
    VER_ADDTREE_TYPE_CASE(CLF_MEAN);
    VER_ADDTREE_TYPE_CASE(CLF_SOFTMAX);
    default:
        throw std::runtime_error("unknown AddTreeType");
    }

#undef VER_ADDTREE_TYPE_CASE
}

AddTreeType
addtree_type_from_str(const std::string& s) {
#define VER_ADDTREE_TYPE_CASE(name) \
    if (s == #name) return veritas::AddTreeType::name;

    VER_ADDTREE_TYPE_CASE(REGR);
    VER_ADDTREE_TYPE_CASE(CLF);

    VER_ADDTREE_TYPE_CASE(REGR_MEAN);
    VER_ADDTREE_TYPE_CASE(CLF_MEAN);
    VER_ADDTREE_TYPE_CASE(CLF_SOFTMAX);
    throw std::runtime_error("unknown AddTreeType");

#undef VER_ADDTREE_TYPE_CASE
}

std::ostream&
operator<<(std::ostream& strm, AddTreeType t) {
    return strm << addtree_type_to_str(t);
}

template <typename TreeT>
void
GAddTree<TreeT>::add_trees(const GAddTree<TreeT>& other) {
    if (other.num_leaf_values() != num_leaf_values())
        throw std::runtime_error("incompatible num_leaf_values");
    for (int i = 0; i < num_leaf_values(); ++i)
        base_score(i) += other.base_score(i);
    for (const TreeT& t : other)
        add_tree(t);
}

template <typename TreeT>
void
GAddTree<TreeT>::add_trees(const GAddTree<TreeT>& other, int c) {
    if (other.num_leaf_values() != 1)
        throw std::runtime_error("AddTree::add_trees: make_multiclass on multiclass");
    for (const TreeT& t : other)
        add_tree(t.make_multiclass(c, num_leaf_values()));
    base_score(c) = other.base_score(0);
}

template <typename TreeT>
GAddTree<TreeT>
GAddTree<TreeT>::make_multiclass(int c, int num_leaf_values) const {
    if (this->num_leaf_values() != 1)
        throw std::runtime_error("AddTree::make_multiclass on multiclass");

    GAddTree<TreeT> new_at(num_leaf_values, at_type_);

    for (const TreeT& t : *this)
        new_at.add_tree(t.make_multiclass(c, num_leaf_values));

    new_at.base_score(c) = base_score(0);

    return new_at;
}

template <typename TreeT>
GAddTree<TreeT>
GAddTree<TreeT>::make_singleclass(int c) const {
    if (this->num_leaf_values() == 1)
        throw std::runtime_error("AddTree::make_singleclass: already singleclass");
    if (this->num_leaf_values() <= c)
        throw std::runtime_error("AddTree::make_singleclass: num_leaf_values <= c");

    GAddTree<TreeT> new_at(1, at_type_);

    for (const TreeT& t : *this) {
        if (!t.is_all_zeros(c))
            new_at.add_tree(t.make_singleclass(c));
    }

    new_at.base_score(0) = base_score(c);

    return new_at;
}

template <typename TreeT>
GAddTree<TreeT>
GAddTree<TreeT>::contrast_classes(int pos_c, int neg_c) const {
    if (this->num_leaf_values() == 1)
        throw std::runtime_error("AddTree::make_singleclass: already singleclass");
    if (this->num_leaf_values() <= pos_c)
        throw std::runtime_error("AddTree::make_singleclass: num_leaf_values <= pos_c");
    if (this->num_leaf_values() <= neg_c)
        throw std::runtime_error("AddTree::make_singleclass: num_leaf_values <= neg_c");

    GAddTree<TreeT> new_at(1, at_type_);

    for (const TreeT& t : *this) {
        TreeT new_tree = t.contrast_classes(pos_c, neg_c);
        if (!new_tree.is_all_zeros(0))
            new_at.add_tree(std::move(new_tree));
    }

    new_at.base_score(0) = base_score(pos_c) - base_score(neg_c);

    return new_at;
}

template <typename TreeT>
void
GAddTree<TreeT>::swap_class(int c) {
    for (auto& t : *this) {
        t.swap_class(c);
    }
}



template <typename TreeT>
size_t GAddTree<TreeT>::num_nodes() const {
    size_t c = 0;
    for (const TreeT& tree : *this)
        c += tree.num_nodes();
    return c;
}

template <typename TreeT>
size_t GAddTree<TreeT>::num_leafs() const {
    size_t c = 0;
    for (const TreeT& tree : *this)
        c += tree.num_leaves();
    return c;
}

template <typename TreeT>
typename GAddTree<TreeT>::SplitMapT
GAddTree<TreeT>::get_splits() const {
    SplitMapT splits;

    // collect all the split values
    for (const TreeT& tree : *this) {
        tree.collect_split_values(tree.root(), splits);
    }

    // sort the split values, remove duplicates
    for (auto& n : splits) {
        std::vector<SplitValueT>& v = n.second;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    return splits;
}


template <typename TreeT>
FeatId
GAddTree<TreeT>::get_maximum_feat_id() const {
    FeatId max = 0;
    for (const TreeT& tree : *this)
        max = std::max(max, tree.get_maximum_feat_id(tree.root()));
    return max;
}

template <typename TreeT>
GAddTree<TreeT>
GAddTree<TreeT>::prune(const BoxRefT& box) const
{
    GAddTree<TreeT> new_at(num_leaf_values(), at_type_);
    new_at.base_scores_ = base_scores_;
    for (const TreeT& t : *this)
        new_at.add_tree(t.prune(box));
    return new_at;
}

template <typename TreeT>
GAddTree<TreeT>
GAddTree<TreeT>::neutralize_negative_leaf_values() const {
    GAddTree<TreeT> new_at(*this);

    for (size_t m = 0; m < size(); ++m) {
        const TreeT& tree = trees_[m];
        TreeT& new_tree = new_at[m];
        auto minmax = tree.find_minmax_leaf_value();
        for (int c = 0; c < num_leaf_values(); ++c) {
            auto &&[min, max] = minmax[c];
            LeafValueType offset{};
            offset = std::min(min, offset);

            // add offset to base_score ...
            new_at.base_score(c) += offset;
            // ... and subtract offsets from leaf_values
            for (NodeId id = 0; id < static_cast<int>(tree.num_nodes()); ++id)
                if (new_tree.is_leaf(id))
                    new_tree.leaf_value(id, c) -= offset;
        }
    }
    return new_at;
}

//template <>
//GAddTree<TreeFp> GAddTree<TreeFp>::neutralize_negative_leaf_values() const;

//template <typename TreeT>
//GAddTree<TreeT>
//GAddTree<TreeT>::limit_depth(int max_depth) const
//{
//    AddTree new_at;
//    new_at.base_score = base_score;
//    for (const Tree& tree : *this)
//        new_at.add_tree(tree.limit_depth(max_depth));
//    return new_at;
//}

//template <typename TreeT>
//GAddTree<TreeT>
//GAddTree<TreeT>::sort_by_leaf_value_variance() const
//{
//    std::vector<std::tuple<size_t, FloatT>> v;
//    for (size_t i = 0; i < size(); ++i)
//        v.push_back({i, trees_[i].leaf_value_variance()});
//    std::sort(v.begin(), v.end(), [](const auto& v, const auto& w) {
//        return std::get<1>(v) > std::get<1>(w); // sort desc
//    });

//    GAddTree<TreeT> new_at;
//    for (auto [id, x] : v)
//    {
//        std::cout << "id: " << id << ", var: " << x << std::endl;
//        new_at.add_tree(trees_[id]);
//    }
//    return new_at;
//}
template <typename TreeT>
GAddTree<TreeT>
GAddTree<TreeT>::concat_negated(const GAddTree<TreeT>& other) const {
    GAddTree<TreeT> new_at(*this);
    for (int i = 0; i < num_leaf_values(); ++i)
        new_at.base_scores_[i] -= other.base_scores_[i];
    for (const TreeT& t : other)
        new_at.add_tree(t.negate_leaf_values());
    return new_at;
}

template <typename TreeT>
GAddTree<TreeT>
GAddTree<TreeT>::negate_leaf_values() const {
    return GAddTree<TreeT>(num_leaf_values(), at_type_).concat_negated(*this);
}

template <typename TreeT>
std::enable_if_t<std::is_same_v<typename TreeT::LeafValueType, FloatT>>
GAddTree<TreeT>::predict(const data<SplitValueT>& row, data<LeafValueType>& result) const {
    const int nlv = num_leaf_values();
    eval(row, result);

    if (_is_a(at_type_, AddTreeType::MEAN)) {
        FloatT shift =_is_a(at_type_, AddTreeType::CLF) ? 0.5 :  0.0;
        double denom = static_cast<double>(size());
        for (int j = 0; j < nlv; ++j) result[j] = result[j] / denom + shift;
    } else if (_is_a(at_type_, AddTreeType::SOFTMAX)) {
        FloatT e = 0;
        for (int j = 0; j < nlv; ++j) e += std::exp(result[j]);
        if (nlv == 1) { e += 1; }; // in binary clf, it becomes a sigmoid
        for (int j = 0; j < nlv; ++j) result[j] = std::exp(result[j]) / e;
    }
}

template <typename TreeT>
std::ostream&
operator<<(std::ostream& strm, const GAddTree<TreeT>& at) {
    strm << "AddTree with " << at.size() << " trees and base_scores [";
    for (int i = 0; i < at.num_leaf_values(); ++i)
        strm << (i>0 ? ", " : "") << at.base_score(i);
    strm << ']';

    return strm;
}

template std::ostream& operator<<(std::ostream&, const AddTree& at);
template std::ostream& operator<<(std::ostream&, const AddTreeFp& at);

/*
template <typename TreeT>
void
GAddTree<TreeT>::to_json(std::ostream& s) const
{
    //s << "{\"base_score\": " << base_score
    //    << ", \"trees\": [" << std::endl;
    //auto it = begin();
    //if (it != end())
    //    (it++)->to_json(s);
    //for (; it != end(); ++it)
    //{
    //    s << ',' << std::endl;
    //    it->to_json(s);
    //}

    //s << "]}";
}

template <typename TreeT>
void
GAddTree<TreeT>::from_json(std::istream& s)
{
    std::string buf;
    char c;
    Tree tree;

    while (s.get(c))
    {
        switch (c)
        {
            case ' ':
            case '\n':
            case ',':
            case '{':
                break;
            case '"':
            case '\'':
                buf.clear();
                while (s.get(c))
                    if (c != '"') buf.push_back(c); else break;
                break;
            case ':':
                if (buf == "base_score")
                    s >> base_score;
                else if (buf == "trees")
                    goto loop2;
                else
                    throw std::runtime_error("addtree parse error: unknown key");
                break;
            default:
                throw std::runtime_error("addtree parse error: unexpected char");
        }
    }

    loop2: while (s.get(c))
    {
        switch (c)
        {
            case ' ':
            case ']':
            case '}':
            case '\n':
                break;
            case '[':
            case ',':
                //tree.from_json(s);
                add_tree(std::move(tree));
                tree.clear();
                break;
            default:
                throw std::runtime_error("addtree parse error (2): unexpected char");
        }
    }
} 
*/

template <typename TreeT>
void
GAddTree<TreeT>::compute_box(typename TreeT::BoxT& box,
        const std::vector<NodeId>& node_ids) const {
    if (size() != node_ids.size())
        throw std::runtime_error("compute_box: one node_id per tree in AddTree");

    for (size_t tree_index = 0; tree_index < size(); ++tree_index) {
        NodeId leaf_id = node_ids[tree_index];
        const TreeT& tree = trees_[tree_index];
        if (!tree.is_leaf(leaf_id))
            throw std::runtime_error("leaf_id does not point to leaf");
        if (!tree.compute_box(leaf_id, box))
            throw std::runtime_error("leaves with non-overlapping boxes");
    }
}


template class GAddTree<Tree>;
template class GAddTree<TreeFp>;

} // namespace veritas
