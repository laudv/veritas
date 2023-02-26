/**
 * \file tree.cpp
 *
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "addtree.hpp"
namespace veritas {


template <typename TreeT>
size_t GAddTree<TreeT>::num_nodes() const
{
    size_t c = 0;
    for (const auto& t : trees_)
        c += t.num_nodes();
    return c;
}

template <typename TreeT>
size_t GAddTree<TreeT>::num_leafs() const
{
    size_t c = 0;
    for (const auto& t : trees_)
        c += t.num_leaves();
    return c;
}

template <typename TreeT>
typename GAddTree<TreeT>::SplitMapT
GAddTree<TreeT>::get_splits() const
{
    SplitMapT splits;

    // collect all the split values
    for (const Tree& tree : trees_)
        tree.collect_split_values(tree.root(), splits);

    // sort the split values, remove duplicates
    for (auto& n : splits)
    {
        std::vector<FloatT>& v = n.second;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    return splits;
}

//template <typename TreeT>
//GAddTree<TreeT>
//GAddTree<TreeT>::prune(BoxRef box) const
//{
//    throw std::runtime_error("not implemented");
//    //GAddTree<TreeT> new_at;
//    //for (const Tree& t : *this)
//    //    new_at.add_tree(t.prune(box));
//    //return new_at;
//}

//template <typename TreeT>
//GAddTree<TreeT>
//GAddTree<TreeT>::neutralize_negative_leaf_values() const
//{
//    throw std::runtime_error("not implemented");
//    //GAddTree<TreeT> new_at;
//    //new_at.base_score = base_score;
//    //for (const Tree& tree : *this)
//    //{
//    //    Tree& new_tree = new_at.add_tree();
//    //    std::stack<Tree::ConstRef, std::vector<Tree::ConstRef>> stack1;
//    //    std::stack<Tree::MutRef, std::vector<Tree::MutRef>> stack2;
//    //    stack1.push(tree.root());
//    //    stack2.push(new_tree.root());
//
//    //    FloatT offset = std::min<FloatT>(0.0, std::get<0>(tree.find_minmax_leaf_value()));
//    //    new_at.base_score += offset;
//
//    //    while (stack1.size() > 0)
//    //    {
//    //        auto n1 = stack1.top(); stack1.pop();
//    //        auto n2 = stack2.top(); stack2.pop();
//    //        if (n1.is_internal())
//    //        {
//    //            n2.split(n1.get_split());
//    //            stack1.push(n1.right()); stack1.push(n1.left());
//    //            stack2.push(n2.right()); stack2.push(n2.left());
//    //        }
//    //        else
//    //        {
//    //            n2.set_leaf_value(n1.leaf_value() - offset);
//    //        }
//    //    }
//
//    //}
//    ////std::cout << "neutralize_negative_leaf_values: base_score "
//    ////    << base_score << " -> " << new_at.base_score << std::endl;
//    //return new_at;
//}

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
//template <typename TreeT>
//GAddTree<TreeT>
//GAddTree<TreeT>::concat_negated(const GAddTree<TreeT>& other) const
//{
//    GAddTree<TreeT> new_at(*this);
//    new_at.base_score -= other.base_score;
//    for (const Tree& t : other)
//        new_at.add_tree(t.negate_leaf_values());
//    return new_at;
//}

//template <typename TreeT>
//GAddTree<TreeT>
//GAddTree<TreeT>::negate_leaf_values() const
//{
//    return GAddTree<TreeT>().concat_negated(*this);
//}

template <typename TreeT>
std::ostream&
operator<<(std::ostream& strm, const GAddTree<TreeT>& at)
{
    return
        strm << "AddTree with " << at.size() << " trees and base_score "
             << at.base_score;
}

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
GAddTree<TreeT>::compute_box(Box& box, const std::vector<NodeId>& node_ids) const
{
    if (size() != node_ids.size())
        throw std::runtime_error("compute_box: one node_id per tree in AddTree");

    for (size_t tree_index = 0; tree_index < size(); ++tree_index)
    {
        NodeId leaf_id = node_ids[tree_index];
        trees_[tree_index].compute_box(leaf_id, box);
    }
}

} // namespace veritas
