/**
 * \file tree.cpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "tree.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <stack>

namespace veritas {

template <typename SplitT, typename ValueT>
GTree<SplitT, ValueT>
GTree<SplitT, ValueT>::prune(const BoxRefT& box) const {
    std::stack<NodeId, std::vector<NodeId>> stack1;
    std::stack<NodeId, std::vector<NodeId>> stack2;

    GTree<SplitT, ValueT> new_tree(num_leaf_values());
    stack1.push(root());
    stack2.push(new_tree.root());

    while (stack1.size() != 0) {
        NodeId n1 = stack1.top();
        stack1.pop();
        NodeId n2 = stack2.top();

        if (is_leaf(n1)) {
            stack2.pop();
            for (int i = 0; i < nleaf_values_; ++i)
                new_tree.leaf_value(n2, i) = leaf_value(n1, i);
        } else {
            const auto& split = get_split(n1);
            auto&& [ival_l, ival_r] = split.get_intervals();
            auto box_ival = box.get(split.feat_id);

            bool overlaps_left = box_ival.overlaps(ival_l);
            bool overlaps_right = box_ival.overlaps(ival_r);

            if (overlaps_left && overlaps_right) {
                stack2.pop();
                new_tree.split(n2, split);
                stack2.push(new_tree.right(n2));
                stack2.push(new_tree.left(n2));
            }

            if (overlaps_right) {
                stack1.push(right(n1));
            }
            if (overlaps_left) {
                stack1.push(left(n1));
            }
        }
    }

    return new_tree;
}

// Tree
// Tree::limit_depth(int max_depth) const
//{
//     Tree new_tree;

//    std::stack<std::tuple<ConstRef, MutRef, int>,
//        std::vector<std::tuple<ConstRef, MutRef, int>>> stack;
//    stack.push({root(), new_tree.root(), 0});

//    while (stack.size() != 0)
//    {
//        auto [n, m, depth] = stack.top();
//        stack.pop();

//        if (depth < max_depth && n.is_internal())
//        {
//            m.split(n.get_split());
//            stack.push({n.right(), m.right(), depth+1});
//            stack.push({n.left(), m.left(), depth+1});
//        }
//        else
//        {
//            // set leaf value to maximum leaf value in subtree
//            m.set_leaf_value(std::get<1>(n.find_minmax_leaf_value()));
//        }
//    }

//    return new_tree;
//}

// FloatT
// Tree::leaf_value_variance() const
//{
//     std::stack<ConstRef, std::vector<ConstRef>> stack;
//     stack.push(root());

//    double sum = 0.0, sum2 = 0.0;
//    int count = 0;
//    while (!stack.empty())
//    {
//        ConstRef n = stack.top();
//        stack.pop();

//        if (n.is_internal())
//        {
//            stack.push(n.right());
//            stack.push(n.left());
//        }
//        else
//        {
//            double lv = static_cast<double>(n.leaf_value());
//            sum += lv;
//            sum2 += lv * lv;
//            count += 1;
//        }
//    }

//    return static_cast<FloatT>((sum2 - (sum*sum) / count) / count);
//}

template <typename SplitT, typename ValueT>
GTree<SplitT, ValueT>
GTree<SplitT, ValueT>::negate_leaf_values() const {
    const GTree<SplitT, ValueT>& tn = *this;
    GTree<SplitT, ValueT> tm(num_leaf_values());

    std::stack<std::tuple<NodeId, NodeId>,
        std::vector<std::tuple<NodeId, NodeId>>> stack;
    stack.push({root(), tm.root()});

    while (stack.size() != 0) {
        auto [n, m] = stack.top();
        stack.pop();

        if (tn.is_internal(n)) {
            tm.split(m, tn.get_split(n));
            stack.push({tn.right(n), tm.right(m)});
            stack.push({tn.left(n), tm.left(m)});
        }
        else {
            for (int i = 0; i < nleaf_values_; ++i)
                tm.leaf_value(m, i) = -tn.leaf_value(n, i);
        }
    }

    return tm;
}


template class GTree<LtSplit, FloatT>;
template class GTree<LtSplitFp, FloatT>;


} // namespace veritas
