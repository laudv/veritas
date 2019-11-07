#ifndef TREECK_TREE_H
#define TREECK_TREE_H

#include <tuple>
#include <variant>
#include <vector>
#include <optional>

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

        LtSplit(FeatId feat_id, ValueT split_value);
        std::tuple<RealDomain, RealDomain> get_domains() const;

        bool test(ValueT value) const;
    };

    struct EqSplit : public SplitBase {
        using ValueT = int;

        ValueT category;

        EqSplit(FeatId feat_id, ValueT category);
        bool test(ValueT value) const;
    };

    using Split = std::variant<LtSplit, EqSplit>;

    class Tree;

    namespace node {
        struct NodeLeaf {
            double value;
            NodeLeaf(double value);
        };

        struct NodeInternal {
            NodeId left;
            /* NodeId right; */ // right = left + 1
            Split split;
        };

        struct Node {
            NodeId id;
            NodeId parent; /* root has itself as parent */
            int depth;
            int tree_size; /* size of tree w/ this node as root */

            union {
                NodeInternal internal;
                NodeLeaf leaf;
            };

            Node(NodeId id, NodeId parent, int depth);
        };
    } /* namespace node */

    class NodeRef {
        Tree *tree;
        int node_id;

        node::Node& node();
        const node::Node& node() const;

    public:
        NodeRef(Tree *tree, NodeId node_id);
        //NodeRef(const NodeRef&);
        //NodeRef(NodeRef&&);
        //NodeRef& operator=(NodeRef&);
        //NodeRef& operator=(NodeRef&&);

        bool is_root() const;
        bool is_leaf() const;
        bool is_internal() const;

        NodeId id() const;
        NodeRef left() const; /* internal only */
        NodeRef right() const; /* internal only */
        NodeRef parent() const;

        int tree_size() const;
        int depth() const;
        const Split& get_split() const; /* internal only */
        double leaf_value() const; /* leaf only */

        void set_leaf_value(double value); /* leaf only */
        void split(Split split); /* leaf only */
    };

    class Tree {
        friend NodeRef;

        std::vector<node::Node> nodes;

    public:
        Tree();
        void split(NodeId node_id, Split split);

        const NodeRef root() const;
        NodeRef root();

        const NodeRef operator[](NodeId index) const;
        NodeRef operator[](NodeId index);
    };


} /* namespace treeck */

#endif /* TREECK_TREE_H */
