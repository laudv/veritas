/**
 * \file fast_eval_addtree.cpp
 *
 * Copyright 2026 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Senne Stans, Laurens Devos
*/

#include "fast_eval_addtree.hpp"

#include <algorithm>
#include <stdexcept>

namespace veritas {

/* Rows are processed in blocks: within a block, loop over the trees and then
 * over the rows, so that both the tree's nodes and the rows stay in cache. */
static constexpr size_t ROW_BLOCK_SIZE = 256;

FastEvalAddTree::FastEvalAddTree(const AddTree& at)
    : nodes_()
    , tree_offsets_()
    , leaf_values_()
    , base_scores_()
    , max_feat_id_(at.get_maximum_feat_id())
{
    const int nlv = at.num_leaf_values();
    for (int k = 0; k < nlv; ++k)
        base_scores_.push_back(at.base_score(k));

    tree_offsets_.reserve(at.size());
    for (const Tree& tree : at) {
        tree_offsets_.push_back(nodes_.size());
        for (size_t j = 0; j < tree.num_nodes(); ++j) {
            NodeId id = static_cast<NodeId>(j);
            if (tree.is_leaf(id)) {
                nodes_.push_back({static_cast<int>(leaf_values_.size()), -1, 0.0});
                for (int k = 0; k < nlv; ++k)
                    leaf_values_.push_back(tree.leaf_value(id, k));
            } else {
                const LtSplit& split = tree.get_split(id);
                nodes_.push_back({split.feat_id, tree.left(id), split.split_value});
            }
        }
    }
}

void FastEvalAddTree::eval(const data<FloatT>& x, data<FloatT>& result) const {
    if (result.num_rows != x.num_rows
            || result.num_cols != static_cast<size_t>(num_leaf_values()))
        throw std::runtime_error("invalid result shape");

    const size_t nlv = static_cast<size_t>(num_leaf_values());
    for (size_t i = 0; i < x.num_rows; ++i)
        for (size_t k = 0; k < nlv; ++k)
            result.get_elem(i, k) = base_scores_[k];

    for (size_t begin = 0; begin < x.num_rows; begin += ROW_BLOCK_SIZE) {
        size_t end = std::min(begin + ROW_BLOCK_SIZE, x.num_rows);
        for (size_t t = 0; t < size(); ++t) {
            const Node *nodes = nodes_.data() + tree_offsets_[t];
            for (size_t i = begin; i < end; ++i) {
                NodeId leaf = eval_tree(t, &x.get_elem(i, 0), x.stride_col);
                const FloatT *values = leaf_values_.data() + nodes[leaf].index;
                for (size_t k = 0; k < nlv; ++k)
                    result.get_elem(i, k) += values[k];
            }
        }
    }
}

void FastEvalAddTree::eval_node(const data<FloatT>& x, data<NodeId>& result) const {
    if (result.num_rows != x.num_rows || result.num_cols != size())
        throw std::runtime_error("invalid result shape");

    for (size_t begin = 0; begin < x.num_rows; begin += ROW_BLOCK_SIZE) {
        size_t end = std::min(begin + ROW_BLOCK_SIZE, x.num_rows);
        for (size_t t = 0; t < size(); ++t)
            for (size_t i = begin; i < end; ++i)
                result.get_elem(i, t) = eval_tree(t, &x.get_elem(i, 0), x.stride_col);
    }
}

} // namespace veritas
