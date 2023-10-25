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
#include <utility>
#include <vector>
#include <unordered_map>
#include <variant>

namespace veritas {

template <typename SplitT, typename ValueT>
class GTree; // generic tree



namespace inner {

struct NodeLeaf {
    int leaf_value_offset;
    NodeLeaf(int offset) : leaf_value_offset{offset} {}
};

template <typename SplitT>
struct NodeInternal {
    NodeId left; // right = left + 1
    SplitT split;

    NodeInternal(NodeId l, SplitT s) : left{l}, split{s} {}
};

template <typename SplitT>
struct Node {
    using LeafT = NodeLeaf;
    using InternalT = NodeInternal<SplitT>;

    NodeId id;
    NodeId parent; /* root has itself as parent */
    int tree_size; /* size of tree w/ this node as root; tree_size==1 => leaf node */
    std::variant<LeafT, InternalT> inner;

    //inline Node() : id(-1), parent(-1), tree_size(-1) {}

    /** new leaf node */
    inline Node(NodeId id, NodeId parent, int offset)
        : id(id)
        , parent(parent)
        , tree_size(1)
        , inner{std::in_place_type<NodeLeaf>, offset}
    {}

    inline bool is_leaf() const { return tree_size == 1; }

    inline int leaf_value_offset() const {
        return std::get<LeafT>(inner).leaf_value_offset;
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
template <typename SplitT, typename LeafValueT>
class GTree {
public:
    using SelfT = GTree<SplitT, LeafValueT>;
    using SplitType = SplitT;
    using SplitValueT = typename SplitT::ValueT;
    using IntervalType = typename SplitT::IntervalT;
    using LeafValueType = LeafValueT;
    using BoxT = GBox<SplitValueT>;
    using BoxRefT = GBoxRef<SplitValueT>;
    using SplitMapT =
        std::unordered_map<FeatId, std::vector<typename SplitT::ValueT>>;
    using LeafValueIterT = typename std::vector<LeafValueT>::iterator;
    using LeafValueConstIterT = typename std::vector<LeafValueT>::const_iterator;

private:
    using NodeT = inner::Node<SplitT>;
    std::vector<NodeT> nodes_;
    std::vector<LeafValueT> leaf_values_;
    int nleaf_values_;
    
    NodeT& node(NodeId id) { return nodes_[id]; }
    const NodeT& node(NodeId id) const { return nodes_[id]; }

public:
    inline GTree(int nleaf_values)
        : nodes_{}
        , leaf_values_{}
        , nleaf_values_{nleaf_values} { clear(); }

    /** Reset this tree. */
    inline void clear() {
        nodes_.clear();
        for (int i = 0; i < nleaf_values_; ++i)
            leaf_values_.emplace_back();
        nodes_.emplace_back(0, 0, 0);
    }

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
    inline int num_leaf_values() const { return nleaf_values_; }

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

    inline const LeafValueType& leaf_value(NodeId id, int index) const {
        if (is_internal(id)) throw std::runtime_error("leaf_value of internal");
        if (index < 0 || index >= nleaf_values_) throw std::runtime_error("invalid index");
        int offset = node(id).leaf_value_offset();
        return leaf_values_[offset + index];
    }

    inline LeafValueType& leaf_value(NodeId id, int index) {
        if (is_internal(id)) throw std::runtime_error("leaf_value of internal");
        if (index < 0 || index >= nleaf_values_) throw std::runtime_error("invalid index");
        int offset = node(id).leaf_value_offset();
        return leaf_values_[offset + index];
    }

    inline LeafValueConstIterT leaf_values_begin(NodeId id) const {
        if (is_internal(id)) throw std::runtime_error("leaf_values of internal");
        int offset = node(id).leaf_value_offset();
        return leaf_values_.cbegin() + offset;
    }

    inline LeafValueIterT leaf_values_begin(NodeId id) {
        if (is_internal(id)) throw std::runtime_error("leaf_values of internal");
        int offset = node(id).leaf_value_offset();
        return leaf_values_.begin() + offset;
    }

    inline LeafValueConstIterT leaf_values_end(NodeId id) const {
        return leaf_values_begin(id) + nleaf_values_;
    }

    inline LeafValueIterT leaf_values_end(NodeId id) {
        return leaf_values_begin(id) + nleaf_values_;
    }

    inline void split(NodeId id, SplitType split) {
        if (is_internal(id)) throw std::runtime_error("split internal");
        int left_offset = node(id).leaf_value_offset();

        // push new values for new right leaf node
        int right_offset = static_cast<int>(leaf_values_.size());
        for (int i = 0; i < nleaf_values_; ++i)
            leaf_values_.emplace_back();

        NodeId left_id = static_cast<NodeId>(nodes_.size());

        nodes_.emplace_back(left_id,   id, left_offset);
        nodes_.emplace_back(left_id+1, id, right_offset);

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
                << ", value=[";
            for (int i = 0; i < nleaf_values_; ++i)
                strm << (i==0 ? "" : ", ") << leaf_value(id, i);
            strm << "])" << std::endl;
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
    SelfT prune(const BoxRefT& box) const;

    /** See NodeRef::find_minmax_leaf_value */
    inline void find_minmax_leaf_value(
        NodeId id, std::vector<std::pair<LeafValueT, LeafValueT>> &buf) const {
        if (is_internal(id)) {
            find_minmax_leaf_value(right(id), buf);
            find_minmax_leaf_value(left(id), buf);
        } else {
            for (int c = 0; c < num_leaf_values(); ++c) {
                LeafValueT v = leaf_value(id, c);
                LeafValueT m = std::min(buf[c].first, v);
                LeafValueT M = std::max(buf[c].second, v);
                buf[c] = {m, M};
            }
        }
    }

    inline std::vector<std::pair<LeafValueT, LeafValueT>>
    find_minmax_leaf_value() const {
        std::vector<std::pair<LeafValueT, LeafValueT>> buf(num_leaf_values());
        find_minmax_leaf_value(root(), buf);
        return buf;
    }

    /** Limit depth and replace leaf values with max leaf value in subtree. */
    //SelfT limit_depth(int max_depth) const;
    /** Compute the variance of the leaf values */
    //FloatT leaf_value_variance() const;
    /** Construct a new tree with negated leaf values. */
    SelfT negate_leaf_values() const;

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

    /** Compute the maximum feat_id value used in a split. */
    FeatId get_maximum_feat_id(NodeId id) const {
        if (is_internal(id)) {
            FeatId feat_id = get_split(id).feat_id;
            return std::max({get_maximum_feat_id(left(id)),
                             get_maximum_feat_id(right(id)), feat_id});
        } else {
            return 0;
        }
    }

    /** Evaluate this tree on an instance. */
    void eval(const data<SplitValueT>& row, data<LeafValueT>& result) const {
        return eval(root(), row, result);
    }
    void eval(NodeId id, const data<SplitValueT> &row,
              data<LeafValueT> &result) const {
        NodeId leaf = eval_node(id, row);
        for (int i = 0; i < num_leaf_values(); ++i)
            result[i] += leaf_value(leaf, i); // (!) addition!
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

    /**
     * Only if this tree has `num_leaf_values() == 1`, move the leaf values to
     * position `c` in a vector of leaf values of size `num_leaf_values`.
     */
    SelfT make_multiclass(int c, int num_leaf_values) const {
        if (this->num_leaf_values() != 1)
            throw std::runtime_error("make_multiclass on multiclass tree");
        if (c >= num_leaf_values)
            throw std::runtime_error("c >= num_leaf_values");
        SelfT new_tree(num_leaf_values);
        make_multiclass(c, new_tree, root(), new_tree.root());
        return new_tree;
    }

    /**
     * Only if this tree has `num_leaf_values() == 1`, move the leaf values to
     * position `c` in a zero vector of leaf values of size `num_leaf_values`.
     */
    void make_multiclass(int c, SelfT& new_tree, NodeId n, NodeId m) const {
        if (is_internal(n)) {
            new_tree.split(m, get_split(n));
            make_multiclass(c, new_tree, left(n), new_tree.left(m));
            make_multiclass(c, new_tree, right(n), new_tree.right(m));
        } else {
            new_tree.leaf_value(m, c) = leaf_value(n, 0);
        }
    }

    /** Swap class 0 and class c */
    void swap_class(int c) {
        for (NodeId n : get_leaf_ids())
            std::swap(leaf_value(n, 0), leaf_value(n, c));
    }

    SelfT make_singleclass(int c) const {
        if (num_leaf_values() == 0)
            throw std::runtime_error("already singleclass");
        if (c >= num_leaf_values())
            throw std::runtime_error("c >= num_leaf_values");
        SelfT new_tree(1);
        make_singleclass(c, new_tree, root(), new_tree.root());
        return new_tree;
    }

    void make_singleclass(int c, SelfT& new_tree, NodeId n, NodeId m) const {
        if (is_internal(n)) {
            new_tree.split(m, get_split(n));
            make_singleclass(c, new_tree, left(n), new_tree.left(m));
            make_singleclass(c, new_tree, right(n), new_tree.right(m));
        } else {
            new_tree.leaf_value(m, 0) = leaf_value(n, c);
        }
    }

    bool is_all_zeros(int c) const { return is_all_zeros(c, root()); }
    bool is_all_zeros(int c, NodeId n) const {
        if (is_internal(n)) {
            return is_all_zeros(c, left(n))
                && is_all_zeros(c, right(n));
        } else {
            return leaf_value(n, c) == 0.0;
        }
    }

    SelfT contrast_classes(int pos_c, int neg_c) const {
        if (num_leaf_values() == 0)
            throw std::runtime_error("already singleclass");
        if (pos_c >= num_leaf_values())
            throw std::runtime_error("pos_c >= num_leaf_values");
        if (neg_c >= num_leaf_values())
            throw std::runtime_error("neg_c >= num_leaf_values");
        SelfT new_tree(1);
        contrast_classes(pos_c, neg_c, new_tree, root(), new_tree.root());
        return new_tree;
    }

    void contrast_classes(int pos_c, int neg_c, SelfT& new_tree, NodeId n, NodeId m) const {
        if (is_internal(n)) {
            new_tree.split(m, get_split(n));
            contrast_classes(pos_c, neg_c, new_tree, left(n), new_tree.left(m));
            contrast_classes(pos_c, neg_c, new_tree, right(n), new_tree.right(m));
        } else {
            new_tree.leaf_value(m, 0) = leaf_value(n, pos_c) - leaf_value(n, neg_c);
        }
    }

    bool subtree_equals(NodeId n, const SelfT& other, NodeId m) const {
        if (is_internal(n) && other.is_internal(m)) {
            return get_split(n) == other.get_split(m)
                && subtree_equals(left(n), other, other.left(m))
                && subtree_equals(right(n), other, other.right(m));
        } else if (is_leaf(n) && other.is_leaf(m)) {
            bool all_equal = true;
            for (int i = 0; i < nleaf_values_; ++i)
                all_equal &= (leaf_value(n, i) == other.leaf_value(m, i));
            return all_equal;
        }
        return false;
    }

    bool operator==(const SelfT& other) const {
        return subtree_equals(root(), other, other.root());
    }

    bool operator!=(const SelfT &other) const { return !(*this == other); }
}; // GTree

template <typename SplitT, typename LeafValueT>
std::ostream& operator<<(std::ostream& strm, const GTree<SplitT, LeafValueT>& t) {
    t.print_node(strm, t.root(), 0);
    return strm;
}

using Tree = GTree<LtSplit, FloatT>;
using TreeFp = GTree<LtSplitFp, FloatT>;

} // namespace veritas

#endif // VERITAS_TREE_HPP
