/**
 * \file addtree.hpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_ADDTREE_HPP
#define VERITAS_ADDTREE_HPP

#include "basics.hpp"
#include "interval.hpp"
#include "box.hpp"
#include "tree.hpp"

#include <memory>

namespace veritas {


/** Additive ensemble of Trees. A sum of Trees. */
template <typename TreeT>
class GAddTree { // generic AddTree
public:
    using TreeType = TreeT;
    using SplitType = typename TreeT::SplitType;
    using SplitValueT = typename TreeT::SplitValueT;
    using ValueType = typename TreeT::ValueType;
    using SplitMapT = typename TreeT::SplitMapT;
    using BoxRefT = typename TreeT::BoxRefT;

    using TreeVecT = std::vector<TreeT>;
    using const_iterator = typename TreeVecT::const_iterator;
    using iterator = typename TreeVecT::iterator;
private:
    TreeVecT trees_;

public:
    ValueType base_score; /**< Constant value added to the output of the ensemble. */
    inline GAddTree() : base_score{} {} ;
    ///** Copy trees (begin, begin+num) from given `at`. */
    //inline GAddTree(const GAddTree& at, size_t begin, size_t num)
    //    : trees_()
    //    , base_score(begin == 0 ? at.base_score : ValueType{}) {
    //    if (begin < at.size() && (begin+num) <= at.size())
    //        trees_ = std::vector(at.begin() + begin, at.begin() + begin + num);
    //    else
    //        throw std::runtime_error("out of bounds");
    //}

    /** Add a new empty tree to the ensemble. */
    inline TreeT& add_tree() {
        return trees_.emplace_back();
    }
    /** Add a tree to the ensemble. */
    inline void add_tree(TreeT&& t) {
        trees_.push_back(std::move(t));
    }
    /** Add a tree to the ensemble. */
    inline void add_tree(const TreeT& t) {
        trees_.push_back(t);
    }

    /** Get mutable reference to tree `i` */
    inline TreeT& operator[](size_t i) { return trees_.at(i); }
    /** Get const reference to tree `i` */
    inline const TreeT& operator[](size_t i) const { return trees_.at(i); }

    inline iterator begin() { return trees_.begin(); }
    inline const_iterator begin() const { return trees_.begin(); }
    inline const_iterator cbegin() const { return trees_.cbegin(); }
    inline iterator end() { return trees_.end(); }
    inline const_iterator end() const { return trees_.end(); }
    inline const_iterator cend() const { return trees_.cend(); }

    /** Number of trees. */
    inline size_t size() const { return trees_.size(); }

    size_t num_nodes() const;
    size_t num_leafs() const;

    /** Map feature -> [list of split values, sorted, unique]. */
    SplitMapT get_splits() const;
    /** Prune each tree in the ensemble. See TreeT::prune. */
    GAddTree prune(const BoxRefT& box) const;

    /**
     * Avoid negative leaf values by adding a constant positive value to
     * the leaf values, and subtracting this value from the #base_score.
     * (base_score-offset) + ({leafs} + offset)
     */
    GAddTree neutralize_negative_leaf_values() const;
    ///** Replace internal nodes at deeper depths by a leaf node with maximum
    // * leaf value in subtree */
    //GAddTree limit_depth(int max_depth) const;
    ///** Sort the trees in the ensemble by leaf-value variance. Largest
    // * variance first. */
    //GAddTree sort_by_leaf_value_variance() const;

    /** Concatenate the negated trees of `other` to this tree. */
    GAddTree concat_negated(const GAddTree& other) const;
    /** Negate the leaf values of all trees. See TreeT::negate_leaf_values. */
    GAddTree negate_leaf_values() const;

    /** Evaluate the ensemble. This is the sum of the evaluations of the
     * trees. See TreeT::eval. */
    ValueType eval(const data<SplitValueT>& row) const {
        ValueType res = base_score;
        for (size_t i = 0; i < size(); ++i)
            res += trees_[i].eval(row);
        return res;
    }

    /** Compute the intersection of the boxes of all leaf nodes. See
     * TreeT::compute_box */
    void compute_box(typename TreeT::BoxT& box,
            const std::vector<NodeId>& node_ids) const;

    bool operator==(const GAddTree& other) const {
        if (size() != other.size()) { return false; }
        if (base_score != other.base_score) { return false; }
        for (size_t i = 0; i < size(); ++i) {
            if (trees_[i] != other.trees_[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const GAddTree &other) const { return !(*this == other); }

}; // GAddTree

template <typename TreeT>
std::ostream& operator<<(std::ostream& strm, const GAddTree<TreeT>& at);

using AddTree = GAddTree<Tree>;
using AddTreeFp = GAddTree<TreeFp>;


} // namespace veritas

#endif // VERITAS_ADDTREE_HPP
