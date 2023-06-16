/**
 * \file leafiter.hpp
 *
 * Iterate over the leaves of a tree that overlap with a given box.
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_LEAFITER_HPP
#define VERITAS_LEAFITER_HPP

#include "box.hpp"

namespace veritas {

template <typename TreeT>
class LeafIter {
public:
    using TreeType = TreeT;
    using IntervalT = typename TreeT::SplitType::IntervalT;
    using ValueT = typename IntervalT::ValueT;
    using BoxRefT = GBoxRef<ValueT>;
    using FlatBoxT = GFlatBox<ValueT>; 

    FlatBoxT flatbox;

private:
    std::vector<NodeId> stack_;
    const TreeT* tree_;

public:
    LeafIter() : flatbox{}, stack_{}, tree_{nullptr} {}

    void setup_tree(const TreeT& t) {
        tree_ = &t;
        if (!stack_.empty())
            throw std::runtime_error("iter stack not empty");
        stack_.push_back(t.root());
    }

    void setup_flatbox(BoxRefT box, const FlatBoxT& prune_box) {
        if (prune_box.size() > flatbox.size())
            flatbox.resize(prune_box.size(), IntervalT());
        std::fill(flatbox.begin(), flatbox.end(), IntervalT());
        std::copy(prune_box.begin(), prune_box.end(), flatbox.begin());
        box.to_flatbox(flatbox, false);
    }

    /* setup the iterator */
    void setup(const TreeT& t, BoxRefT box, const FlatBoxT& prune_box) {
        setup_tree(t);
        setup_flatbox(box, prune_box);
    }

    /* find next overlapping leaf */
    NodeId next() {
        while (!stack_.empty())
        {
            NodeId id = stack_.back();
            stack_.pop_back();

            if (tree_->is_leaf(id))
                return id;

            const GLtSplit<ValueT>& s = tree_->get_split(id);
            IntervalT d;
            if (static_cast<size_t>(s.feat_id) < flatbox.size())
                d = flatbox[s.feat_id];

            if (d.hi > s.split_value) // high is exclusive
                stack_.push_back(tree_->right(id));
            if (d.lo < s.split_value) // lo is inclusive, but split is LT
                stack_.push_back(tree_->left(id));
        }

        // clean up iterator
        tree_ = nullptr;
        return -1;
    }

    void clear() {
        stack_.clear();
        tree_ = nullptr;
    }
}; // class LeafIter

} // namespace veritas

#endif // VERITAS_LEAFITER_HPP
