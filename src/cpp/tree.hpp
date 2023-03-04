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

#include <stdexcept>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <variant>

namespace veritas {

template <typename SplitT, typename ValueT>
class GTree; // generic tree



namespace inner {

template <typename ValueT>
struct NodeLeaf {
    ValueT leaf_value;
    NodeLeaf() : leaf_value{} {}
};

template <typename SplitT>
struct NodeInternal {
    NodeId left; // right = left + 1
    SplitT split;

    NodeInternal(NodeId l, SplitT s) : left{l}, split{s} {}
};

template <typename SplitT, typename ValueT>
struct Node {
    using LeafT = NodeLeaf<ValueT>;
    using InternalT = NodeInternal<SplitT>;

    NodeId id;
    NodeId parent; /* root has itself as parent */
    int tree_size; /* size of tree w/ this node as root; tree_size==1 => leaf node */
    std::variant<LeafT, InternalT> inner;

    //inline Node() : id(-1), parent(-1), tree_size(-1) {}

    /** new leaf node */
    inline Node(NodeId id, NodeId parent)
        : id(id), parent(parent), tree_size(1), inner{} {}

    inline bool is_leaf() const { return tree_size == 1; }

    inline const ValueT& leaf_value() const {
        return std::get<LeafT>(inner).leaf_value;
    }

    inline ValueT& leaf_value() {
        return std::get<LeafT>(inner).leaf_value;
    }

    inline NodeId left() const {
        return std::get<InternalT>(inner).left;
    }

    inline NodeId right() const {
        return std::get<InternalT>(inner).left + 1;
    }

    inline const SplitT& split() const {
        return std::get<InternalT>(inner).split;
    }

    inline SplitT& split() {
        return std::get<InternalT>(inner).split;
    }

    inline void make_internal(SplitT split, NodeId left_id) {
        inner = InternalT(left_id, split);
    }

};

} // namespace inner






/**
 * A binary decision tree with less-than splits.
 */
template <typename SplitT, typename ValueT>
class GTree {
public:
    using SelfT = GTree<SplitT, ValueT>;
    using SplitType = SplitT;
    using SplitValueT = typename SplitT::ValueT;
    using ValueType = ValueT;
    using BoxT = GBox<SplitValueT>;
    using BoxRefT = GBoxRef<SplitValueT>;
    using SplitMapT =
        std::unordered_map<FeatId, std::vector<typename SplitT::ValueT>>;

private:
    using NodeT = inner::Node<SplitT, ValueT>;
    std::vector<NodeT> nodes_;
    
    NodeT& node(NodeId id) { return nodes_[id]; }
    const NodeT& node(NodeId id) const { return nodes_[id]; }

public:
    inline GTree() { clear(); }

    /** Reset this tree. */
    inline void clear() { nodes_.clear(); nodes_.push_back({0, {}}); }

    inline NodeId root() const { return 0; }

    /** Bounds check the given node id. */
    inline bool is_valid_node_id(NodeId id) const {
        return id >= 0 && static_cast<size_t>(id) < nodes_.size();
    }

    /** Bounds check the given node id, throw error if invalid. */
    inline void check_node_id(NodeId id) const {
        if (!is_valid_node_id(id))
            throw std::runtime_error("invalid node id");
    }

    inline bool is_root(NodeId id) const {
        auto n = node(id); return n.parent == id;
    }

    inline bool is_leaf(NodeId id) const { return node(id).is_leaf(); }

    inline bool is_internal(NodeId id) const { return !is_leaf(id); }

    inline bool is_left_child(NodeId id) const {
        return !is_root(id) && left(parent(id)) == id;
    }

    inline bool is_right_child(NodeId id) const {
        return !is_root(id) && right(parent(id)) == id;
    }

    inline NodeId left(NodeId id) const {
        if (is_leaf(id)) throw std::runtime_error("left of leaf");
        return node(id).left();
    };

    inline NodeId right(NodeId id) const {
        if (is_leaf(id)) throw std::runtime_error("right of leaf");
        return node(id).right();
    }

    inline NodeId parent(NodeId id) const {
        if (is_root(id)) throw std::runtime_error("parent of root");
        return node(id).parent;
    }

    inline NodeId operator[](const char *dirs) const { return navigate(dirs); }
    inline NodeId navigate(const char *dirs) const {
        NodeId n = root();
        for (; *dirs != '\0'; ++dirs) {
            if (*dirs == 'l')
                n = left(n);
            else if(*dirs == 'r')
                n = right(n);
            else
                throw std::invalid_argument("invalid char");
        }
        return n;
    }


    inline int tree_size(NodeId id) const { return node(id).tree_size; }

    inline int depth(NodeId id) const {
        int depth = 0;
        while (!is_root(id)) {
            id = parent(id);
            depth++;
        }
        return depth;
    }

    inline const SplitType& get_split(NodeId id) const {
        if (is_leaf(id)) throw std::runtime_error("get_split of leaf");
        return node(id).split();
    }

    inline const ValueType& leaf_value(NodeId id) const {
        if (is_internal(id)) throw std::runtime_error("leaf_value of internal");
        return node(id).leaf_value();
    }

    inline ValueType& leaf_value(NodeId id) {
        if (is_internal(id)) throw std::runtime_error("leaf_value of internal");
        return node(id).leaf_value();
    }

    inline void split(NodeId id, SplitType split) {
        if (is_internal(id)) throw std::runtime_error("split internal");

        NodeId left_id = static_cast<NodeId>(nodes_.size());

        NodeT left(left_id, id);
        NodeT right(left_id + 1, id);

        nodes_.push_back(left);
        nodes_.push_back(right);

        {
            NodeT& n = node(id);
            n.make_internal(split, left_id);
            n.tree_size = 3;
        }

        while (!is_root(id))
        {
            id = parent(id);
            node(id).tree_size += 2;
        }
    }

    inline size_t num_leaves() const { return num_leaves(root()); }
    inline size_t num_leaves(NodeId id) const {
        return is_leaf(id) ? 1
                           : num_leaves(left(id)) + num_leaves(right(id));
    }

    inline size_t num_nodes() const {
        //return static_cast<size_t>(tree_size(root()));
        return nodes_.size();
    }

    inline std::vector<NodeId> get_leaf_ids() const {
        std::vector<NodeId> ids;
        get_leaf_ids(ids);
        return ids;
    }

    inline void get_leaf_ids(std::vector<NodeId> &ids) const {
        get_leaf_ids(root(), ids);
    }

    inline void get_leaf_ids(NodeId id, std::vector<NodeId>& ids) const {
        if (is_internal(id)) {
            get_leaf_ids(left(id), ids);
            get_leaf_ids(right(id), ids);
        } else ids.push_back(id);
    }

    inline void collect_split_values(NodeId id, SplitMapT& splits) const {
        if (is_leaf(id)) return;

        SplitType split = get_split(id);
        auto search = splits.find(split.feat_id);
        if (search != splits.end()) { // found it!
            splits[split.feat_id].push_back(split.split_value);
        } else {
            splits.emplace(split.feat_id, typename SplitMapT::mapped_type{
                                              split.split_value});
        }

        collect_split_values(right(id), splits);
        collect_split_values(left(id), splits);
    }

    inline bool compute_box(NodeId id, BoxT& box) const {
        while (!is_root(id)) {
            bool isleft = is_left_child(id);
            id = parent(id);
            if (!box.refine_box(get_split(id), isleft))
                return false;
        }
        return true;
    }

    inline void print_node(std::ostream& strm, NodeId id, int depth) const {
        int i = 1;
        for (; i < depth; ++i)
            strm << "│  ";
        if (is_leaf(id)) {
            strm << (is_right_child(id) ? "└─ " : "├─ ")
                << "Leaf("
                << "id=" << id
                << ", sz=" << tree_size(id)
                << ", value=" << leaf_value(id)
                << ')' << std::endl;
        } else {
            strm << (depth==0 ? "" : "├─ ")
                << "Node("
                << "id=" << id
                << ", split=[" << get_split(id) << ']'
                << ", sz=" << tree_size(id)
                << ", left=" << left(id)
                << ", right=" << right(id)
                << ')' << std::endl;
            print_node(strm, left(id), depth+1);
            print_node(strm, right(id), depth+1);
        }
    }
    




    /** Prune all branches that are never taken for examples in the given box. */
    //SelfT prune(const BoxRefT& box) const;

    /** See NodeRef::find_minmax_leaf_value */
    std::tuple<ValueT, ValueT> find_minmax_leaf_value(NodeId id) const {
        if (is_internal(id)) {
            auto&& [rmin, rmax] = find_minmax_leaf_value(right(id));
            auto&& [lmin, lmax] = find_minmax_leaf_value(left(id));
            return { std::min(lmin, rmin), std::max(lmax, rmax) };
        } else {
            return { leaf_value(id), leaf_value(id) };
        }
    }

    inline std::tuple<ValueT, ValueT> find_minmax_leaf_value() const {
        return find_minmax_leaf_value(root());
    }

    /** Limit depth and replace leaf values with max leaf value in subtree. */
    //SelfT limit_depth(int max_depth) const;
    /** Compute the variance of the leaf values */
    //FloatT leaf_value_variance() const;
    /** Construct a new tree with negated leaf values. */
    //SelfT negate_leaf_values() const;

    /** Map feature -> [list of split values, sorted, unique]. */
    SplitMapT get_splits() const {
        SplitMapT splits;
        collect_split_values(root(), splits);

        // sort the split values, remove duplicates
        for (auto& n : splits) {
            auto& v = n.second;
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
        return splits;
    }

    /** Evaluate this tree on an instance. */
    const ValueType& eval(const data<SplitValueT>& row) const {
        return eval(root(), row);
    }
    const ValueType& eval(NodeId id, const data<SplitValueT>& row) const {
        return leaf_value(eval_node(id, row));
    }

    /**
     * Evaluate this tree on an instance, but return node_id of leaf
     * instead of leaf value.
     */
    NodeId eval_node(const data<SplitValueT>& row) const {
        return eval_node(root(), row);
    }
    NodeId eval_node(NodeId id, const data<SplitValueT>& row) const {
        if (is_leaf(id))
            return id;
        return get_split(id).test(row) ? eval_node(left(id), row)
                                       : eval_node(right(id), row);
    }

    bool subtree_equals(NodeId i, const SelfT& other, NodeId j) const {
        if (is_internal(i) && other.is_internal(j))
            return get_split(i) == other.get_split(j)
                && subtree_equals(left(i), other, other.left(j))
                && subtree_equals(right(i), other, other.right(j));
        else if (is_leaf(i) && other.is_leaf(j))
            return leaf_value(i) == other.leaf_value(j);
        return false;
    }

    bool operator==(const SelfT& other) const {
        return subtree_equals(root(), other, other.root());
    }

    bool operator!=(const SelfT &other) const { return !(*this == other); }
}; // GTree

template <typename SplitT, typename ValueT>
std::ostream& operator<<(std::ostream& strm, const GTree<SplitT, ValueT>& t) {
    t.print_node(strm, t.root(), 0);
    return strm;
}

using Tree = GTree<LtSplit, FloatT>;
using TreeFp = GTree<LtSplitFp, FloatT>;

} // namespace veritas

#endif // VERITAS_TREE_HPP
