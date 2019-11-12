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

    class Tree;

    namespace node {

        struct NodeLeaf {
            double value;
        };

        struct NodeInternal {
            NodeId left; // right = left + 1
            Split split;
        };

        struct Node {
            friend std::ostream& operator<<(std::ostream&, const Node&);
            friend std::istream& operator>>(std::istream&, Node&);

            NodeId id;
            NodeId parent; /* root has itself as parent */
            int depth;
            int tree_size; /* size of tree w/ this node as root */

            union {
                NodeInternal internal;
                NodeLeaf leaf;
            };

            Node();
            Node(NodeId id, NodeId parent, int depth);
            bool is_leaf() const;

            template<typename Archive>
            void serialize(Archive& archive);
        };

    } /* namespace node */

    class NodeRef {
        using TreeP = Tree *;

        TreeP tree;
        int node_id;

        const node::Node& node() const;
        node::Node& node();

    public:
        NodeRef(TreeP tree, NodeId node_id);

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

    std::ostream& operator<<(std::ostream& s, const NodeRef& n);

    class Tree {
        friend class NodeRef;
        std::vector<node::Node> nodes;

    public:
        Tree();
        void split(NodeId node_id, Split split);

        NodeRef root();
        int num_nodes() const;
        std::tuple<unsigned long long int, unsigned long long int> id() const;

        NodeRef operator[](NodeId index);

        template <typename Archive>
        void serialize(Archive& archive);

        std::string to_json();
        static Tree from_json(const std::string& json);
    };

    std::ostream& operator<<(std::ostream& s, Tree& t);

    class AddTree {
        std::vector<Tree> trees;

    public:
        double base_score;

        AddTree();

        size_t add_tree(Tree&& tree);
        size_t size() const;

        Tree& operator[](size_t index);
        const Tree& operator[](size_t index) const;

        std::string to_json();
        static AddTree from_json(const std::string& json);
    };


} /* namespace treeck */

#endif /* TREECK_TREE_H */
