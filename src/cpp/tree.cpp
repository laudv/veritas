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

namespace veritas {

    //template <typename SplitT, typename ValueT>
    //GTree<SplitT, ValueT>
    //GTree<SplitT, ValueT>::prune(const BoxRefT<typename SplitT::Interval::ValueT>& box) const
    //{
    //    std::stack<ConstRef, std::vector<ConstRef>> stack1;
    //    std::stack<MutRef, std::vector<MutRef>> stack2;
    //
    //    Tree new_tree;
    //    stack1.push(root());
    //    stack2.push(new_tree.root());
    //
    //    while (stack1.size() != 0)
    //    {
    //        ConstRef n1 = stack1.top();
    //        stack1.pop();
    //        MutRef n2 = stack2.top();
    //
    //        if (n1.is_leaf())
    //        {
    //            stack2.pop();
    //            n2.set_leaf_value(n1.leaf_value());
    //        }
    //        else
    //        {
    //            Domain ldom, rdom;
    //            int flag = box.overlaps(n1.get_split());
    //
    //            if (flag == (BoxRef::OVERLAPS_LEFT | BoxRef::OVERLAPS_RIGHT))
    //            {
    //                stack2.pop();
    //                n2.split(n1.get_split());
    //                stack2.push(n2.right());
    //                stack2.push(n2.left());
    //            }
    //
    //            if ((flag & BoxRef::OVERLAPS_RIGHT) != 0)
    //            {
    //                stack1.push(n1.right());
    //            }
    //            if ((flag & BoxRef::OVERLAPS_LEFT) != 0)
    //            {
    //                stack1.push(n1.left());
    //            }
    //        }
    //    }
    //
    //    return new_tree;
    //}

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

// Tree
// Tree::negate_leaf_values() const
//{
//     Tree new_tree;

//    std::stack<std::tuple<ConstRef, MutRef>,
//        std::vector<std::tuple<ConstRef, MutRef>>> stack;
//    stack.push({root(), new_tree.root()});

//    while (stack.size() != 0)
//    {
//        auto [n, m] = stack.top();
//        stack.pop();

//        if (n.is_internal())
//        {
//            m.split(n.get_split());
//            stack.push({n.right(), m.right()});
//            stack.push({n.left(), m.left()});
//        }
//        else m.set_leaf_value(-n.leaf_value());
//    }

//    return new_tree;
//}


} // namespace veritas
