/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_TREE_HPP
#define VERITAS_TREE_HPP

#include "domain.hpp"
#include <vector>
#include <unordered_map>
#include <iostream>
#include <numeric>

namespace veritas {

    class Tree;

    namespace inner {
        struct NodeLeaf { FloatT leaf_value; };
        struct NodeInternal {
            NodeId left; // right = left + 1;
            LtSplit split;
        };

        struct Node {
            NodeId id;
            NodeId parent; /* root has itself as parent */
            int tree_size; /* size of tree w/ this node as root; tree_size==1 => leaf node */

            union {
                inner::NodeLeaf leaf;
                inner::NodeInternal internal;
            };

            //inline Node() : id(-1), parent(-1), tree_size(-1) {}

            /** new leaf node */
            inline Node(NodeId id, NodeId parent) : id(id), parent(parent), tree_size(1), leaf{} {}
            inline Node(const Node& o) : id(o.id), parent(o.parent), tree_size(o.tree_size), leaf{}
            {
                if (o.is_leaf()) leaf = o.leaf;
                else             internal = o.internal;
            }

            inline bool is_leaf() const { return tree_size == 1; }
        };

        struct ConstRef {
            using TreePtr = const Tree *;
            using TreeRef = const Tree&;
            using is_mut_type = std::false_type;
        };
        struct MutRef {
            using TreePtr = Tree *;
            using TreeRef = Tree&;
            using is_mut_type = std::true_type;
        };
    } // namespace inner




    template <typename RefT /* inner::ConstRef or inner::MutRef */>
    class NodeRef {
    public:
        using TreePtr = typename RefT::TreePtr;
        using TreeRef = typename RefT::TreeRef;

    private:
        TreePtr tree_;
        NodeId node_id_;

        inline const inner::Node& node() const { return tree_->nodes_[node_id_]; };

        template <typename T=RefT>
        inline std::enable_if_t<T::is_mut_type::value, inner::Node&> node()
        { return tree_->nodes_[node_id_]; }

    public:
        inline NodeRef(TreePtr tree, NodeId node_id) : tree_(tree), node_id_(node_id) {}
        inline NodeRef(TreeRef tree, NodeId node_id) : tree_(&tree), node_id_(node_id) {}
        inline NodeRef(const NodeRef<RefT>& o) : tree_(o.tree_), node_id_(o.node_id_) {}
        inline NodeRef<RefT>& operator=(const NodeRef<RefT>& o)
        { tree_ = o.tree_; node_id_ = o.node_id_; return *this; }

        inline NodeRef<inner::ConstRef> to_const() const { return { tree_, node_id_ }; }

        inline bool is_root() const { return node().parent == node().id; }
        inline bool is_leaf() const { return node().is_leaf(); }
        inline bool is_internal() const { return !is_leaf(); }
        inline bool is_left_child() const { return !is_root() && parent().left().id() == id(); }
        inline bool is_right_child() const { return !is_root() && parent().right().id() == id(); }

        inline NodeId id() const { return node_id_; }
        inline NodeRef<RefT> left() const
        {
            if (is_leaf()) throw std::runtime_error("left of leaf");
            return { tree_, node().internal.left };
        }
        inline NodeRef<RefT> right() const
        {
            if (is_leaf()) throw std::runtime_error("right of leaf");
            return { tree_, node().internal.left + 1 };
        }
        inline NodeRef<RefT> parent() const
        {
            if (is_root()) throw std::runtime_error("parent of root");
            return { tree_, node().parent };
        }

        inline int tree_size() const { return node().tree_size; }

        inline int depth() const
        {
            int depth = 0;
            NodeRef n(*this);
            while (!n.is_root())
            {
                n = n.parent();
                depth += 1;
            }
            return depth;
        }

        inline LtSplit get_split() const
        {
            if (is_leaf()) throw std::runtime_error("get_split of leaf");
            return node().internal.split;
        }

        inline FloatT leaf_value() const
        {
            if (is_internal()) throw std::runtime_error("get_split of internal");
            return node().leaf.leaf_value;
        }

        template <typename T=RefT>
        inline std::enable_if_t<T::is_mut_type::value, void>
        set_leaf_value(FloatT value)
        {
            if (is_internal()) throw std::runtime_error("set_leaf_value of internal");
            node().leaf.leaf_value = value;
        }

        template <typename T=RefT>
        inline std::enable_if_t<T::is_mut_type::value, void>
        split(LtSplit split)
        {
            if (is_internal()) throw std::runtime_error("split internal");

            NodeId left_id = tree_->nodes_.size();

            inner::Node left(left_id,      id());
            inner::Node right(left_id + 1, id());

            tree_->nodes_.push_back(left);
            tree_->nodes_.push_back(right);

            node().internal.split = split;
            node().internal.left = left_id;

            node().tree_size = 3;
            NodeRef n(*this);
            while (!n.is_root())
            {
                n = n.parent();
                n.node().tree_size += 2;
            }
        }

        template <typename T=RefT>
        inline std::enable_if_t<T::is_mut_type::value, void>
        split(FeatId feat_id) { split({feat_id, BOOL_SPLIT_VALUE}); } // bool split

        inline size_t num_leafs() const
        { return is_leaf() ? 1 : left().num_leafs() + right().num_leafs(); }

        bool operator==(const NodeRef<inner::ConstRef>& other) const
        {
            if (is_internal() && other.is_internal())
                return get_split() == other.get_split()
                    && left() == other.left()
                    && right() == other.right();
            else if (is_leaf() && other.is_leaf())
                return leaf_value() == other.leaf_value();
            return false;
        }

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

        /** Get the domain restrictions on the features in this node. */
        Box compute_box() const;
        void compute_box(Box& box) const;

        void print_node(std::ostream& strm, int depth) const;
        void to_json(std::ostream& strm, int depth) const;

        template <typename T=RefT>
        std::enable_if_t<T::is_mut_type::value, void>
        from_json(std::istream& strm);

        template <typename D>
        FloatT eval(const row<D>& data) const;
    }; // NodeRef




    class Tree {
    public:
        using ConstRef = NodeRef<inner::ConstRef>;
        using MutRef = NodeRef<inner::MutRef>;

    private:
        friend ConstRef;
        friend MutRef;

        std::vector<inner::Node> nodes_;

    public:
        inline Tree() { clear(); }
        inline ConstRef root() const { return (*this)[0]; }
        inline MutRef root() { return (*this)[0]; }
        inline void clear() { nodes_.clear(); nodes_.push_back({0, 0}); }

        inline ConstRef operator[] (NodeId id) const { return { *this, id }; }
        inline MutRef operator[] (NodeId id) { return { *this, id }; }

        inline bool is_valid_node_id(NodeId id) const
        { return id >= 0 && static_cast<size_t>(id) < nodes_.size(); }

        inline void check_node_id(NodeId id) const
        { if (!is_valid_node_id(id)) throw std::runtime_error("invalid node id"); }

        inline ConstRef node_const(NodeId id) const { // range checked
            #ifndef VERITAS_SAFETY_CHECKS_DISABLED
            check_node_id(id);
            #endif

            return { *this, id };
        }

        inline MutRef node_mut(NodeId id) { // range checked
            #ifndef VERITAS_SAFETY_CHECKS_DISABLED
            check_node_id(id);
            #endif

            return { *this, id };
        }

        inline size_t num_leafs() const { return root().num_leafs(); }
        inline size_t num_nodes() const { return root().tree_size(); }

        inline void to_json(std::ostream& strm) const { root().to_json(strm, 0); }
        inline void from_json(std::istream& strm) { root().from_json(strm); };

        Tree prune(BoxRef box) const;
        std::tuple<FloatT, FloatT> find_minmax_leaf_value() const { return root().find_minmax_leaf_value(); }
        Tree limit_depth(int max_depth) const;

        template <typename D>
        FloatT eval(const row<D>& data) const { return root().eval(data); }

        bool operator==(const Tree& other) const { return root() == other.root(); }
    }; // Tree

    std::ostream& operator<<(std::ostream& strm, const Tree& t);




    class AddTree {
    public:
        using const_iterator = std::vector<Tree>::const_iterator;
        using iterator = std::vector<Tree>::iterator;
        using SplitMapT = std::unordered_map<FeatId, std::vector<FloatT>>;
    private:
        std::vector<Tree> trees_;

    public:
        FloatT base_score;
        inline AddTree() : base_score{0.0} {} ;
        inline AddTree(const AddTree& at, size_t begin, size_t num)
            : trees_()
            , base_score(begin == 0 ? at.base_score : 0.0)
        {
            if (begin < at.size() && (begin+num) < at.size())
                trees_ = std::vector(at.begin() + begin, at.begin() + begin + num);
        }

        inline Tree& add_tree() { return trees_.emplace_back(); }
        inline void add_tree(Tree&& t) { trees_.emplace_back(std::move(t)); }
        inline void add_tree(const Tree& t) { trees_.push_back(t); }

        inline Tree& operator[](size_t i) { return trees_[i]; }
        inline const Tree& operator[](size_t i) const { return trees_[i]; }

        inline iterator begin() { return trees_.begin(); }
        inline const_iterator begin() const { return trees_.begin(); }
        inline const_iterator cbegin() const { return trees_.cbegin(); }
        inline iterator end() { return trees_.end(); }
        inline const_iterator end() const { return trees_.end(); }
        inline const_iterator cend() const { return trees_.cend(); }

        inline size_t size() const { return trees_.size(); }

        size_t num_nodes() const;
        size_t num_leafs() const;

        SplitMapT get_splits() const;
        AddTree prune(BoxRef box) const;

        std::tuple<FloatT, FloatT> find_minmax_leaf_value() const;
        /** (base_score-offset) + ({leafs} + offset) */
        AddTree neutralize_negative_leaf_values() const;
        /** replace internal nodes at deeper depths by leaf node with maximum leaf value in subtree */
        AddTree limit_depth(int max_depth) const;

        void to_json(std::ostream& strm) const;
        void from_json(std::istream& strm);

        template <typename D>
        FloatT eval(const row<D>& r) const
        {
            auto op = [&r](FloatT v, const Tree& t) { return v + t.eval(r); };
            return std::accumulate(begin(), end(), 0.0, op) + base_score;
        }

        void compute_box(Box& box, const std::vector<NodeId> node_ids) const;

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
