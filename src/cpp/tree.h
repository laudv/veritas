#ifndef TREECK_TREE_H
#define TREECK_TREE_H

#include <iostream>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "domain.h"

namespace treeck {

    using NodeId = int; /* Node Id type */
    using FeatId = int; /* Feat Id type */

    struct SplitBase {
        FeatId feat_id;

        SplitBase(FeatId feat_id);
    };

    struct LtSplit : public SplitBase {
        using ValueT = double;
        ValueT split_value;

        LtSplit();
        LtSplit(FeatId feat_id, ValueT split_value);
        std::tuple<RealDomain, RealDomain> get_domains() const;
        bool test(ValueT value) const;

        template<typename Archive>
        void serialize(Archive& archive);
    };

    struct EqSplit : public SplitBase {
        using ValueT = int;
        ValueT category;

        EqSplit();
        EqSplit(FeatId feat_id, ValueT category);
        bool test(ValueT value) const;

        template<typename Archive>
        void serialize(Archive& archive);
    };

    using Split = std::variant<LtSplit, EqSplit>;

    std::ostream& operator<<(std::ostream& strm, const Split& s);

    template <typename LeafT>
    class Tree;

    namespace node {

        template <typename LeafT>
        struct NodeLeaf {
            LeafT value;
        };

        struct NodeInternal {
            NodeId left; // right = left + 1
            Split split;
        };

        template <typename LeafT>
        struct Node {
            NodeId id;
            NodeId parent; /* root has itself as parent */
            int depth;
            int tree_size; /* size of tree w/ this node as root */

            union {
                NodeInternal internal;
                NodeLeaf<LeafT> leaf;
            };

            Node();
            Node(NodeId id, NodeId parent, int depth);
            bool is_leaf() const;

            template<typename Archive>
            void serialize(Archive& archive);
        };

    } /* namespace node */

    template <typename LeafT>
    class NodeRef {
        using TreeP = Tree<LeafT> *;

        TreeP tree;
        int node_id;

        const node::Node<LeafT>& node() const;
        node::Node<LeafT>& node();

    public:
        NodeRef(TreeP tree, NodeId node_id);

        bool is_root() const;
        bool is_leaf() const;
        bool is_internal() const;

        NodeId id() const;
        NodeRef<LeafT> left() const; /* internal only */
        NodeRef<LeafT> right() const; /* internal only */
        NodeRef<LeafT> parent() const;

        int tree_size() const;
        int depth() const;
        const Split& get_split() const; /* internal only */
        LeafT leaf_value() const; /* leaf only */

        void set_leaf_value(LeafT value); /* leaf only */
        void split(Split split); /* leaf only */
    };

    template <typename LeafT>
    std::ostream& operator<<(std::ostream& s, const NodeRef<LeafT>& n);

    template <typename LeafT>
    class Tree {
        friend class NodeRef<LeafT>;
        std::vector<node::Node<LeafT>> nodes_;

    public:
        Tree();
        void split(NodeId node_id, Split split);

        NodeRef<LeafT> root();
        int num_nodes() const;
        std::tuple<unsigned long long int, unsigned long long int> id() const;

        NodeRef<LeafT> operator[](NodeId index);

        template <typename Archive>
        void serialize(Archive& archive);

        std::string to_json();
        static Tree from_json(const std::string& json);
    };

    template <typename LeafT>
    std::ostream& operator<<(std::ostream& s, Tree<LeafT>& t);

} /* namespace treeck */

#endif /* TREECK_TREE_H */
