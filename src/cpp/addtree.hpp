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

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace veritas {

/*!
 * @brief  Type of AddTree
 * When a GAddTree instance is created an no type is specified, the GAddTree
 * will have the `AddTreeType::REGR` Currently AddTreeType is only used in the
 * pybind predict() function
 */
enum class AddTreeType : uint8_t {
    REGR        = 0b00000000,
    CLF         = 0b00000001,
    // (SUM implicit)
    MEAN        = 0b00000010,
    SOFTMAX     = 0b00000100,

    REGR_MEAN   = REGR | MEAN, 
    CLF_MEAN    = CLF | MEAN, 
    CLF_SOFTMAX = CLF | SOFTMAX
};

const char *
addtree_type_to_str(AddTreeType t);

AddTreeType
addtree_type_from_str(const std::string& t);

std::ostream&
operator<<(std::ostream& strm, AddTreeType t);

template <typename TreeT>
/**
 * @brief Generic additive tree ensembles.
 */
class GAddTree {
public:
    using TreeType = TreeT;
    using SplitType = typename TreeT::SplitType;
    using SplitValueT = typename TreeT::SplitValueT;
    using LeafValueType = typename TreeT::LeafValueType;
    using SplitMapT = typename TreeT::SplitMapT;
    using BoxRefT = typename TreeT::BoxRefT;

    using TreeVecT = std::vector<TreeT>;
    using const_iterator = typename TreeVecT::const_iterator;
    using iterator = typename TreeVecT::iterator;

private:
    TreeVecT trees_;
    std::vector<LeafValueType> base_scores_; /**< Constant value added to the output of the ensemble. */
    AddTreeType at_type_;

public:
    /**
     * @brief Create a new AddTree instance
     * @param nleaf_values The number of values in a single leaf
     * @param at_type Type of AddTree
     * 
     *  Create an empty AddTree. When an AddTreeType is not specified, the AddTree will have the `AddTreeType::REGR`
     *  @see `veritas::AddTreeType`
     */
    inline GAddTree(int nleaf_values, AddTreeType at_type)
        : trees_(), base_scores_(nleaf_values, {}), at_type_(at_type) {}

    ///** Copy trees (begin, begin+num) from given `at`. */
    // inline GAddTree(const GAddTree& at, size_t begin, size_t num)
    //     : trees_()
    //     , base_score(begin == 0 ? at.base_score : LeafValueType{}) {
    //     if (begin < at.size() && (begin+num) <= at.size())
    //         trees_ = std::vector(at.begin() + begin, at.begin() + begin + num);
    //     else
    //         throw std::runtime_error("out of bounds");
    // }

    /** Add a new empty tree to the ensemble. */
    inline TreeT& add_tree() {
        return trees_.emplace_back(num_leaf_values());
    }

    /** Add a tree to the ensemble. */
    inline void add_tree(TreeT&& t) {
        if (t.num_leaf_values() != num_leaf_values())
            throw std::runtime_error("num_leaf_values does not match");
        trees_.push_back(std::move(t));
    }

    /** Add a tree to the ensemble. */
    inline void add_tree(const TreeT& t) {
        if (t.num_leaf_values() != num_leaf_values())
            throw std::runtime_error("num_leaf_values does not match");
        trees_.push_back(t);
    }

    /** Add copies of the trees in `other` to this ensemble. */
    void add_trees(const GAddTree<TreeT>& other);

    /** Add multi-class copies of the trees in `other` to this ensemble. */
    void add_trees(const GAddTree<TreeT>& other, int c);

    /** Turn this ensemble in a multi-class ensemble. @see `GTree::make_multiclass`. */
    GAddTree<TreeT> make_multiclass(int c, int num_leaf_values) const;

    /** Turn this ensemble in a single-class ensemble. @see `GTree::make_singleclass`. */
    GAddTree<TreeT> make_singleclass(int c) const;

    /** Turn this multiclass ensemble in a binary ensemble.
     * @see `GTree::contrast_classes`. */
    GAddTree<TreeT> contrast_classes(int pos_c, int neg_c) const;



    /** See GTree::swap_class */
    void swap_class(int c);

    /** Get mutable reference to tree `i` */
    inline TreeT& operator[](size_t i) { return trees_.at(i); }
    /** Get const reference to tree `i` */
    inline const TreeT& operator[](size_t i) const { return trees_.at(i); }

    inline const LeafValueType& base_score(int index) const {
        return base_scores_.at(index);
    }

    inline LeafValueType& base_score(int index) {
        return base_scores_.at(index);
    }

    inline iterator begin() { return trees_.begin(); }
    inline const_iterator begin() const { return trees_.begin(); }
    inline const_iterator cbegin() const { return trees_.cbegin(); }
    inline iterator end() { return trees_.end(); }
    inline const_iterator end() const { return trees_.end(); }
    inline const_iterator cend() const { return trees_.cend(); }

    /** Number of trees. */
    inline size_t size() const { return trees_.size(); }

    /** Number of leaf values */
    inline int num_leaf_values() const {
        return static_cast<int>(base_scores_.size());
    }

    size_t num_nodes() const;
    size_t num_leafs() const;

    inline AddTreeType get_type() const { return at_type_; }

    /** Map feature -> [list of split values, sorted, unique]. */
    SplitMapT get_splits() const;

    /** Get the maximum feat id value that is used in the trees */
    FeatId get_maximum_feat_id() const;

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
    // GAddTree limit_depth(int max_depth) const;
    ///** Sort the trees in the ensemble by leaf-value variance. Largest
    // * variance first. */
    // GAddTree sort_by_leaf_value_variance() const;

    /** Concatenate the negated trees of `other` to this tree. */
    GAddTree concat_negated(const GAddTree& other) const;
    /** Negate the leaf values of all trees. See TreeT::negate_leaf_values. */
    GAddTree negate_leaf_values() const;

    /** Evaluate the ensemble. This is the sum of the evaluations of the
     * trees. See TreeT::eval. */
    inline void eval(const data<SplitValueT>& row, data<LeafValueType>& result) const {
        for (int i = 0; i < num_leaf_values(); ++i)
            result[i] = base_scores_[i];
        for (size_t m = 0; m < size(); ++m)
            trees_[m].eval(row, result);
    }

    std::enable_if_t<std::is_same_v<LeafValueType, FloatT>>
    predict(const data<SplitValueT>& row, data<LeafValueType>& result) const;

    /** Compute the intersection of the boxes of all leaf nodes. See
     * TreeT::compute_box */
    void compute_box(typename TreeT::BoxT& box, const std::vector<NodeId>& node_ids) const;

    bool operator==(const GAddTree& other) const {
        if (size() != other.size()) { return false; }
        if (base_scores_ != other.base_scores_) { return false; }
        for (size_t i = 0; i < size(); ++i)
        {
            if (trees_[i] != other.trees_[i]) { return false; }
        }
        return true;
    }

    bool operator!=(const GAddTree& other) const { return !(*this == other); }

}; // GAddTree

template <typename TreeT>
std::ostream& operator<<(std::ostream& strm, const GAddTree<TreeT>& at);

using AddTree = GAddTree<Tree>;
using AddTreeFp = GAddTree<TreeFp>;

} // namespace veritas

#endif // VERITAS_ADDTREE_HPP
