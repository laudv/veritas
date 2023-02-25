/**
 * \file tree.hpp
 *
 * The Veritas internal tree representation. Trees are binary and only support
 * less-than splits. Binary splits can be achieved (given data is in {0, 1})
 * using `LtSplit(feat_id, BOOL_SPLIT_VALUE=0.5)`.
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_TREE_HPP
#define VERITAS_TREE_HPP

#include "basics.hpp"
#include "interval.hpp"
#include "box.hpp"

#include <type_traits>
#include <vector>
#include <iostream>
#include <numeric> // std::accumulate
#include <unordered_map>

namespace veritas {

    template <typename SplitT, typename ValueT>
    class GTree; // generic tree

    namespace inner {
        template <typename ValueT>
        struct NodeLeaf { ValueT leaf_value; };

        template <typename SplitT>
        struct NodeInternal {
            NodeId left; // right = left + 1
            SplitT split;
        };

        template <typename SplitT, typename ValueT>
        struct Node {
            NodeId id;
            NodeId parent; /* root has itself as parent */
            int tree_size; /* size of tree w/ this node as root; tree_size==1 => leaf node */

            union {
                inner::NodeLeaf<ValueT> leaf;
                inner::NodeInternal<SplitT> internal;
            };

            //inline Node() : id(-1), parent(-1), tree_size(-1) {}

            /** new leaf node */
            inline Node(NodeId id, NodeId parent)
                : id(id), parent(parent), tree_size(1), leaf{} {}
            inline Node(const Node &o)
                : id(o.id), parent(o.parent), tree_size(o.tree_size), leaf{} {
                if (o.is_leaf()) leaf = o.leaf;
                else             internal = o.internal;
            }

            inline bool is_leaf() const { return tree_size == 1; }
        };

        template <typename SplitT, typename ValueT>
        struct ConstRef {
            using TreePtr = const GTree<SplitT, ValueT> *;
            using TreeRef = const GTree<SplitT, ValueT>&;
            using is_mut_type = std::false_type;
        };
        template <typename SplitT, typename ValueT>
        struct MutRef {
            using TreePtr = GTree<SplitT, ValueT> *;
            using TreeRef = GTree<SplitT, ValueT>&;
            using is_mut_type = std::true_type;
        };
    } // namespace inner




    /**
     * A reference to a node in a `veritas::Tree`.
     *
     * A node reference can be constant, disallowing mutations to the tree, or
     * it can be mutable, allowing mutations to the tree. This is controlled by
     * the template `RefT`, which is either `veritas.inner.ConstRef` or
     * `veritas.inner.MutRef`.
     *
     * You can get a NodeRef using
     * - Tree::root()
     * - Tree::operator[](NodeId)
     * - Tree::node_const(NodeId)
     * - Tree::node_mut(NodeId)
     *
     * You can convert a mut ref into a const ref using `veritas::NodeRef::to_const()`.
     */
    template <typename SplitT, typename ValueT, typename RefT /* inner::ConstRef or inner::MutRef */>
    class NodeRef {
    public:
        using SelfT = NodeRef<SplitT, ValueT, RefT>;
        using ConstSelfT = NodeRef<SplitT, ValueT, inner::ConstRef<SplitT, ValueT>>;
        using MutSelfT = NodeRef<SplitT, ValueT, inner::MutRef<SplitT, ValueT>>;
        using TreePtr = typename RefT::TreePtr;
        using TreeRef = typename RefT::TreeRef;
        using NodeT = inner::Node<SplitT, ValueT>;
        using BoxT = GBox<typename SplitT::ValueT>;
        using SplitType = SplitT;
        using ValueType = ValueT;
        using SplitMapT = std::unordered_map<FeatId, std::vector<typename SplitT::ValueT>>;

    private:
        TreePtr tree_;
        NodeId node_id_;

        inline const NodeT& node() const { return tree_->nodes_[node_id_]; };

        template <typename T=RefT>
        inline std::enable_if_t<T::is_mut_type::value, NodeT&> node()
        { return tree_->nodes_[node_id_]; }

    public:
        inline NodeRef(TreePtr tree, NodeId node_id) : tree_(tree), node_id_(node_id) {}
        inline NodeRef(TreeRef tree, NodeId node_id) : tree_(&tree), node_id_(node_id) {}
        inline NodeRef(const SelfT& o) : tree_(o.tree_), node_id_(o.node_id_) {}
        inline SelfT& operator=(const SelfT& o)
        { tree_ = o.tree_; node_id_ = o.node_id_; return *this; }

        /** Convert this to a constant reference. */
        inline ConstSelfT to_const() const { return { tree_, node_id_ }; }

        inline bool is_root() const { return node().parent == node().id; }
        inline bool is_leaf() const { return node().is_leaf(); }
        inline bool is_internal() const { return !is_leaf(); }
        inline bool is_left_child() const { return !is_root() && parent().left().id() == id(); }
        inline bool is_right_child() const { return !is_root() && parent().right().id() == id(); }

        /** Get the node id of this node. */
        inline NodeId id() const { return node_id_; }
        /** Navigate to the left child. */
        inline SelfT left() const
        {
            if (is_leaf()) throw std::runtime_error("left of leaf");
            return { tree_, node().internal.left };
        }
        /** Navigate to the right child. */
        inline SelfT right() const
        {
            if (is_leaf()) throw std::runtime_error("right of leaf");
            return { tree_, node().internal.left + 1 };
        }
        /** Navigate to the parent of this node. */
        inline SelfT parent() const
        {
            if (is_root()) throw std::runtime_error("parent of root");
            return { tree_, node().parent };
        }

        /** Number of nodes in this (sub)tree. */
        inline int tree_size() const { return node().tree_size; }

        /** Compute the depth of this node. */
        inline int depth() const
        {
            int depth = 0;
            SelfT n(*this);
            while (!n.is_root())
            {
                n = n.parent();
                depth += 1;
            }
            return depth;
        }

        /** Access the split of this internal node. */
        inline SplitType get_split() const
        {
            if (is_leaf()) throw std::runtime_error("get_split of leaf");
            return node().internal.split;
        }

        /** Access the leaf value of this leaf node. */
        inline FloatT leaf_value() const
        {
            if (is_internal()) throw std::runtime_error("leaf_value of internal");
            return node().leaf.leaf_value;
        }

        /** Set the leaf value of this leaf node. */
        template <typename T=RefT>
        inline std::enable_if_t<T::is_mut_type::value, void>
        set_leaf_value(FloatT value)
        {
            if (is_internal()) throw std::runtime_error("set_leaf_value of internal");
            node().leaf.leaf_value = value;
        }

        /** Split this leaf node. */
        template <typename T=RefT>
        inline std::enable_if_t<T::is_mut_type::value, void>
        split(SplitType split)
        {
            if (is_internal()) throw std::runtime_error("split internal");

            NodeId left_id = static_cast<NodeId>(tree_->nodes_.size());

            NodeT left(left_id,      id());
            NodeT right(left_id + 1, id());

            tree_->nodes_.push_back(left);
            tree_->nodes_.push_back(right);

            node().internal.split = split;
            node().internal.left = left_id;

            node().tree_size = 3;
            SelfT n(*this);
            while (!n.is_root())
            {
                n = n.parent();
                n.node().tree_size += 2;
            }
        }

        /**
         * Boolean split; uses SplitType with BOOL_SPLIT_VALUE
         * (only SplitType::ValueT==FloatT)
         */
        template <typename T = RefT, typename ST = SplitT>
        inline std::enable_if_t<
            T::is_mut_type::value,
            std::enable_if_t<std::is_same_v<typename ST::ValueT, FloatT>>>
        split(FeatId feat_id) {
            split({feat_id, BOOL_SPLIT_VALUE});
        } // bool split

        /** Count the number of leafs in this (sub)tree. */
        inline size_t num_leafs() const
        { return is_leaf() ? 1 : left().num_leafs() + right().num_leafs(); }

        bool operator==(const ConstSelfT& other) const
        {
            if (is_internal() && other.is_internal())
                return get_split() == other.get_split()
                    && left() == other.left()
                    && right() == other.right();
            else if (is_leaf() && other.is_leaf())
                return leaf_value() == other.leaf_value();
            return false;
        }

        /** Returns the minimum and maximum leaf value in this (sub)tree. */
        std::tuple<FloatT, FloatT> find_minmax_leaf_value() const
        {
            if (is_internal())
            {
                auto &&[lm, lM] = left().find_minmax_leaf_value();
                auto &&[rm, rM] = right().find_minmax_leaf_value();
                return {std::min(lm, rm), std::max(lM, rM)};
            }
            else return {leaf_value(), leaf_value()};
        }

        /** Returns a vector of the leaf node ids in this subtree. */
        std::vector<NodeId> get_leaf_ids() const
        {
            std::vector<NodeId> ids;
            get_leaf_ids(ids);
            return ids;
        }

        /** Appends the leaf node ids in this subtree to the given vector. */
        void get_leaf_ids(std::vector<NodeId>& ids) const
        {
            if (is_internal())
            {
                left().get_leaf_ids(ids);
                right().get_leaf_ids(ids);
            }
            else ids.push_back(id());
        }

        void collect_split_values(SplitMapT& splits) const
        {
            if (is_leaf()) return;

            const auto& split = get_split();
            auto search = splits.find(split.feat_id);
            if (search != splits.end()) { // found it!
                splits[split.feat_id].push_back(split.split_value);
            } else {
                splits.emplace(split.feat_id, typename SplitMapT::mapped_type{
                                                  split.split_value});
            }

            right().collect_split_values(splits);
            left().collect_split_values(splits);
        }

        /** Like NodeRef::compute_box(), but write to given Box */
        bool compute_box(BoxT& box) const;

        void print_node(std::ostream& strm, int depth) const;
        void to_json(std::ostream& strm, int depth) const;

        template <typename T=RefT>
        std::enable_if_t<T::is_mut_type::value, void>
        from_json(std::istream& strm);

        FloatT eval(const data& data) const;
        NodeId eval_node(const data& data) const;
    }; // NodeRef




    /**
     * A binary decision tree with less-than splits.
     */
    template <typename SplitT, typename ValueT>
    class GTree {
    public:
        using SelfT = GTree<SplitT, ValueT>;
        using ConstRef = NodeRef<SplitT, ValueT, inner::ConstRef<SplitT, ValueT>>;
        using MutRef = NodeRef<SplitT, ValueT, inner::MutRef<SplitT, ValueT>>;
        using SplitType = SplitT;
        using ValueType = ValueT;
        using BoxRefT = GBoxRef<typename SplitT::ValueT>;
        using SplitMapT = typename ConstRef::SplitMapT;

    private:
        friend ConstRef;
        friend MutRef;

        std::vector<inner::Node<SplitT, ValueT>> nodes_;

    public:
        inline GTree() { clear(); }
        /** Const NodeRef to root node */
        inline ConstRef root() const { return (*this)[0]; }
        /** Const NodeRef to root node */
        inline ConstRef root_const() const { return (*this)[0]; }
        /** Mutable NodeRef to root node */
        inline MutRef root() { return (*this)[0]; }
        /** Mutable NodeRef to root node */
        inline MutRef root_mut() { return (*this)[0]; }
        /** Reset this tree. */
        inline void clear() { nodes_.clear(); nodes_.push_back({0, 0}); }

        /** Get a const NodeRef to node with given id. */
        inline ConstRef operator[] (NodeId id) const { return { *this, id }; }
        /** Get a mutable NodeRef to node with given id. */
        inline MutRef operator[] (NodeId id) { return { *this, id }; }

        /** Bounds check the given node id. */
        inline bool is_valid_node_id(NodeId id) const
        { return id >= 0 && static_cast<size_t>(id) < nodes_.size(); }

        /** Bounds check the given node id, throw error if invalid. */
        inline void check_node_id(NodeId id) const
        { if (!is_valid_node_id(id)) throw std::runtime_error("invalid node id"); }

        /** Like Tree::operator[](NodeId), but with bounds check. */
        inline ConstRef node_const(NodeId id) const { // range checked
            if constexpr (check_sanity())
                check_node_id(id);

            return { *this, id };
        }

        /** Like Tree::operator[](NodeId), but with bounds check. */
        inline MutRef node_mut(NodeId id) { // range checked
            if constexpr (check_sanity())
                check_node_id(id);

            return { *this, id };
        }

        inline size_t num_leafs() const { return root().num_leafs(); }
        inline size_t num_nodes() const { return static_cast<size_t>(root().tree_size()); }

        inline void to_json(std::ostream& strm) const { root().to_json(strm, 0); }
        inline void from_json(std::istream& strm) { root().from_json(strm); };

        /** Prune all branches that are never taken for examples in the given box. */
        //SelfT prune(const BoxRefT& box) const;
        /** See NodeRef::find_minmax_leaf_value */
        //std::tuple<FloatT, FloatT> find_minmax_leaf_value() const { return root().find_minmax_leaf_value(); }
        /** See NodeRef::get_leaf_ids */
        std::vector<NodeId> get_leaf_ids() const { return root().get_leaf_ids(); }
        /** Limit depth and replace leaf values with max leaf value in subtree. */
        //SelfT limit_depth(int max_depth) const;
        /** Compute the variance of the leaf values */
        //FloatT leaf_value_variance() const;
        /** Construct a new tree with negated leaf values. */
        //SelfT negate_leaf_values() const;

        /** Map feature -> [list of split values, sorted, unique]. */
        SplitMapT get_splits() const {
            SplitMapT splits;
            root().collect_split_values(splits);

            // sort the split values, remove duplicates
            for (auto& n : splits) {
                auto& v = n.second;
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
            }
            return splits;
        }

        /** Evaluate this tree on an instance. */
        FloatT eval(const data& row) const { return root().eval(row); }
        /** Evaluate this tree on an instance, but return node_id of leaf
         * instead of leaf value. */
        NodeId eval_node(const data& data) const { return root().eval_node(data); }

        bool operator==(const SelfT& other) const { return root() == other.root(); }
    }; // GTree

    template <typename SplitT, typename ValueT>
    std::ostream& operator<<(std::ostream& strm, const GTree<SplitT, ValueT>& t);



    using Tree = GTree<LtSplit, FloatT>;
    using TreeFp = GTree<LtSplitFp, FpT>;

    /** Additive ensemble of Trees. A sum of Trees. */
    class AddTree {
    public:
        using const_iterator = std::vector<Tree>::const_iterator;
        using iterator = std::vector<Tree>::iterator;
        using SplitMapT = typename Tree::SplitMapT;
    private:
        std::vector<Tree> trees_;

    public:
        FloatT base_score; /**< Constant value added to the output of the ensemble. */
        inline AddTree() : base_score{0.0} {} ;
        /** Copy trees (begin, begin+num) from given `at`. */
        inline AddTree(const AddTree& at, size_t begin, size_t num)
            : trees_()
            , base_score(begin == 0 ? at.base_score : FloatT(0.0))
        {
            if (begin < at.size() && (begin+num) <= at.size())
                trees_ = std::vector(at.begin() + begin, at.begin() + begin + num);
            else
                throw std::runtime_error("out of bounds");
        }

        /** Add a new empty tree to the ensemble. */
        inline Tree& add_tree() { return trees_.emplace_back(); }
        /** Add a tree to the ensemble. */
        inline void add_tree(Tree&& t) { trees_.emplace_back(std::move(t)); }
        /** Add a tree to the ensemble. */
        inline void add_tree(const Tree& t) { trees_.push_back(t); }

        /** Get mutable reference to tree `i` */
        inline Tree& operator[](size_t i) { return trees_[i]; }
        /** Get const reference to tree `i` */
        inline const Tree& operator[](size_t i) const { return trees_[i]; }

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
        /** Prune each tree in the ensemble. See Tree::prune. */
        AddTree prune(BoxRef box) const;

        /** Avoid negative leaf values by adding a constant positive value to
         * the leaf values, and subtracting this value from the #base_score.
         * (base_score-offset) + ({leafs} + offset) */
        AddTree neutralize_negative_leaf_values() const;
        /** Replace internal nodes at deeper depths by a leaf node with maximum
         * leaf value in subtree */
        AddTree limit_depth(int max_depth) const;
        /** Sort the trees in the ensemble by leaf-value variance. Largest
         * variance first. */
        AddTree sort_by_leaf_value_variance() const;
        /** Concatenate the negated trees of `other` to this tree. */
        AddTree concat_negated(const AddTree& other) const;
        /** Negate the leaf values of all trees. See Tree::negate_leaf_values. */
        AddTree negate_leaf_values() const;

        void to_json(std::ostream& strm) const;
        void from_json(std::istream& strm);

        /** Evaluate the ensemble. This is the sum of the evaluations of the
         * trees. See Tree::eval. */
        FloatT eval(const data& row) const
        {
            auto op = [&row](FloatT v, const Tree& t) { return v + t.eval(row); };
            return std::accumulate(begin(), end(), base_score, op);
        }

        /** Compute the intersection of the boxes of all leaf nodes. See
         * Tree::compute_box */
        void compute_box(Box& box, const std::vector<NodeId>& node_ids) const;

        bool operator==(const AddTree& other) const
        {
            auto it1 = begin(), it2 = other.begin();
            for (; it1 != end() && it2 != other.end(); ++it1, ++it2)
                if (!it1->operator==(*it2)) return false;
            return base_score == other.base_score
                && it1 == end() && it2 == other.end();
        }

    }; // AddTree

    std::ostream& operator<<(std::ostream& strm, const AddTree& at);

} // namespace veritas

#endif // VERITAS_TREE_HPP
