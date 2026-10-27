/**
 * \file fast_eval_addtree.hpp
 *
 * Copyright 2026 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Senne Stans, Laurens Devos
 */

#ifndef VERITAS_FAST_EVAL_ADDTREE_HPP
#define VERITAS_FAST_EVAL_ADDTREE_HPP

#include "addtree.hpp"
#include "basics.hpp"

#include <vector>

namespace veritas {

/**
 * Read-only snapshot of an AddTree optimized for batch evaluation.
 *
 * The trees are linearized into one compact node array at construction time.
 * Later modifications to the source AddTree are not reflected.
 */
class FastEvalAddTree {
public:
    /** Compact node: internal if `left >= 0`, leaf otherwise. */
    struct Node {
        /** Internal: feature id of split. Leaf: offset into leaf values. */
        int index;
        /** Internal: id of left child (right = left + 1). Leaf: -1. */
        NodeId left;
        /** Internal: split value, `x < value` goes left. Leaf: unused. */
        FloatT value;
    };

private:
    std::vector<Node> nodes_;
    std::vector<size_t> tree_offsets_; /**< Index of each tree's root in nodes_. */
    std::vector<FloatT> leaf_values_;
    std::vector<FloatT> base_scores_;
    FeatId max_feat_id_;

public:
    explicit FastEvalAddTree(const AddTree& at);

    inline size_t size() const { return tree_offsets_.size(); }
    inline int num_leaf_values() const { return static_cast<int>(base_scores_.size()); }
    inline size_t num_nodes() const { return nodes_.size(); }
    inline FeatId get_maximum_feat_id() const { return max_feat_id_; }

    /**
     * Evaluate the ensemble, equivalent to AddTree::eval for each row.
     * `result` has one row per row in `x` and `num_leaf_values()` columns.
     */
    void eval(const data<FloatT>& x, data<FloatT>& result) const;

    /**
     * Leaf node id reached in each tree. `result` has one row per row in `x`
     * and `size()` columns.
     */
    void eval_node(const data<FloatT>& x, data<NodeId>& result) const;

private:
    inline NodeId eval_tree(size_t tree_index, const FloatT *row, size_t stride_col) const {
        const Node *nodes = nodes_.data() + tree_offsets_[tree_index];
        NodeId id = 0;
        while (nodes[id].left >= 0) {
            const Node& n = nodes[id];
            id = n.left + static_cast<NodeId>(!(row[static_cast<size_t>(n.index) * stride_col] < n.value));
        }
        return id;
    }
};

} // namespace veritas

#endif // VERITAS_FAST_EVAL_ADDTREE_HPP
