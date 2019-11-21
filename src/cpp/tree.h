#ifndef TREECK_TREE_H
#define TREECK_TREE_H

#include <fstream>
#include <iostream>
#include <iostream>
#include <optional>
#include <sstream>
#include <variant>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/variant.hpp>

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

    namespace inner {

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

        template <typename TLeafT>
        struct ConstRef {
            using LeafT = TLeafT;
            using TreeP = const Tree<LeafT> *;
            using is_mut_type = std::false_type;
        };

        template <typename TLeafT>
        struct MutRef {
            using LeafT = TLeafT;
            using TreeP = Tree<LeafT> *;
            using is_mut_type = std::true_type;
        };

    } /* namespace inner */

    template <typename RefT>
    class NodeRef {
    public:
        using LeafT = typename RefT::LeafT;
        using TreeP = typename RefT::TreeP;

    private:
        TreeP tree_;
        int node_id_;

        const inner::Node<LeafT>& node() const;

        template <typename T = RefT>
        std::enable_if_t<T::is_mut_type::value, inner::Node<LeafT>&>
        node(); /* mut only */

    public:
        NodeRef(TreeP tree, NodeId node_id);

        bool is_root() const;
        bool is_leaf() const;
        bool is_internal() const;
        bool is_left_child() const;
        bool is_right_child() const;

        NodeId id() const;

        NodeRef<RefT> left() const; /* internal only */
        NodeRef<RefT> right() const; /* internal only */
        NodeRef<RefT> parent() const;

        int tree_size() const;
        int depth() const;
        const Split& get_split() const; /* internal only */
        LeafT leaf_value() const; /* leaf only */

        template <typename T = RefT>
        std::enable_if_t<T::is_mut_type::value, void>
        set_leaf_value(LeafT value); /* leaf & mut only */

        template <typename T = RefT>
        std::enable_if_t<T::is_mut_type::value, void>
        split(Split split); /* leaf & mut only */
    };

    template <typename RefT>
    std::ostream& operator<<(std::ostream& s, const NodeRef<RefT>& n);


    enum TreeVisitStatus {
        ADD_NONE = 0,
        ADD_LEFT = 1,
        ADD_RIGHT = 2,
        ADD_LEFT_AND_RIGHT = 3
    };

    template <typename LeafT>
    class Tree {
    public:
        using CRef = NodeRef<inner::ConstRef<LeafT>>;
        using MRef = NodeRef<inner::MutRef<LeafT>>;

    private:
        friend MRef;
        friend CRef;

        std::vector<inner::Node<LeafT>> nodes_;

    public:
        Tree();
        void split(NodeId node_id, Split split);

        CRef root() const;
        MRef root();

        CRef operator[](NodeId index) const;
        MRef operator[](NodeId index);

        int num_nodes() const;

        template <typename TreeVisitorT>
        void dfs(TreeVisitorT& visitor) const;

        template <typename TreeVisitorT>
        void dfs(TreeVisitorT&& visitor) const;

        template <typename Archive>
        void serialize(Archive& archive);

        std::string to_json();
        static Tree from_json(const std::string& json);
    };

    template <typename LeafT>
    std::ostream& operator<<(std::ostream& s, const Tree<LeafT>& t);


    class AddTree {
    public:
        using TreeT = Tree<double>;

    private:
        std::vector<TreeT> trees_;

    public:
        double base_score;

        AddTree();

        size_t add_tree(TreeT&& tree);
        size_t size() const;
        size_t num_nodes() const;

        TreeT& operator[](size_t index);
        const TreeT& operator[](size_t index) const;
        const std::vector<TreeT>& trees() const;

        std::string to_json();
        static AddTree from_json(const std::string& json);
        static AddTree from_json_file(const char *file);
    };

    std::ostream& operator<<(std::ostream& s, const AddTree& at);

} /* namespace treeck */

#endif /* TREECK_TREE_H */
