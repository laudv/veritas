#ifndef TREECK_TREE_H
#define TREECK_TREE_H

#include <fstream>
#include <iostream>
#include <iostream>
#include <optional>
#include <sstream>
#include <type_traits>
#include <variant>
#include <vector>
#include <tuple>
#include <unordered_map>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/variant.hpp>

#include "domain.h"

namespace treeck {

    using NodeId = int; /* Node Id type */
    using FeatId = int; /* Feat Id type */
    using DomainsT = std::unordered_map<FeatId, Domain>;

    struct SplitBase {
        FeatId feat_id;

        SplitBase(FeatId feat_id);
    };

    struct LtSplit : public SplitBase {
        using ValueT = FloatT;
        using DomainT = RealDomain;

        ValueT split_value;

        LtSplit();
        LtSplit(FeatId feat_id, ValueT split_value);
        bool test(ValueT value) const; // true goes left, false goes right

        std::tuple<DomainT, DomainT> get_domains() const;

        template<typename Archive>
        void serialize(Archive& archive);
    };
    bool operator==(const LtSplit& a, const LtSplit& b);

    //struct EqSplit : public SplitBase {
    //    using ValueT = int;
    //    ValueT category;

    //    EqSplit();
    //    EqSplit(FeatId feat_id, ValueT category);
    //    bool test(ValueT value) const; // true goes left, false goes right

    //    std::tuple<RealDomain, RealDomain> get_domains();

    //    template<typename Archive>
    //    void serialize(Archive& archive);
    //};
    //bool operator==(const EqSplit& a, const EqSplit& b);

    struct BoolSplit : public SplitBase {
        using ValueT = bool;
        using DomainT = BoolDomain;

        BoolSplit();
        BoolSplit(FeatId feat_id);
        bool test(ValueT value) const; // true goes left, false goes right

        std::tuple<DomainT, DomainT> get_domains() const;

        template<typename Archive>
        void serialize(Archive& archive);
    };
    bool operator==(const BoolSplit& a, const BoolSplit& b);

    using Split = std::variant<LtSplit, BoolSplit>;

    template <typename LtSplitF, typename BoolSplitF>
    static
    std::enable_if_t<std::is_same_v<
            std::invoke_result_t<LtSplitF, const LtSplit&>,
            std::invoke_result_t<BoolSplitF, const BoolSplit&>>,
        std::invoke_result_t<LtSplitF, const LtSplit&>>
    visit_split(LtSplitF&& f1, BoolSplitF&& f2, const Split& split)
    {
        return std::visit([f1, f2](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, LtSplit>)
                return f1(arg);
            else if constexpr (std::is_same_v<T, BoolSplit>)
                return f2(arg);
            else 
                static_assert(util::always_false<T>::value, "non-exhaustive visit_split");
        }, split);
    }

    template <typename SplitT>
    typename SplitT::DomainT
    refine_domain(const typename SplitT::DomainT& base_dom,
                  const SplitT& split, bool is_left_child);

    template <typename SplitT>
    void refine_domains(DomainsT& domains, const SplitT& split, bool is_left_child);

    void refine_domains(DomainsT& domains, const Split& split, bool is_left_child);

    std::ostream& operator<<(std::ostream& strm, const Split& s);
    bool operator==(const Split& a, const Split& b);



    template <typename SplitT, typename LeafT>
    class Tree;

    namespace inner {

        template <typename LeafT>
        struct NodeLeaf {
            LeafT value;
        };

        template <typename SplitT>
        struct NodeInternal {
            NodeId left; // right = left + 1
            SplitT split;
        };

        template <typename SplitT, typename LeafT>
        struct Node {
            NodeId id;
            NodeId parent; /* root has itself as parent */
            int tree_size; /* size of tree w/ this node as root */

            union {
                NodeInternal<SplitT> internal;
                NodeLeaf<LeafT> leaf;
            };

            Node();
            Node(const Node<SplitT, LeafT>&);
            Node(NodeId id, NodeId parent);
            bool is_leaf() const;

            template<typename Archive>
            void serialize(Archive& archive);
        };

        template <typename TSplitT, typename TLeafT>
        struct ConstRef {
            using SplitT = TSplitT;
            using LeafT = TLeafT;
            using TreeP = const Tree<SplitT, LeafT> *;
            using is_mut_type = std::false_type;
        };

        template <typename TSplitT, typename TLeafT>
        struct MutRef {
            using SplitT = TSplitT;
            using LeafT = TLeafT;
            using TreeP = Tree<SplitT, LeafT> *;
            using is_mut_type = std::true_type;
        };

    } /* namespace inner */

    template <typename RefT>
    class NodeRef {
    public:
        using SplitT = typename RefT::SplitT;
        using LeafT = typename RefT::LeafT;
        using NodeT = inner::Node<SplitT, LeafT>;
        using TreeP = typename RefT::TreeP;

    private:
        TreeP tree_;
        NodeId node_id_;

        const NodeT& node() const;

        template <typename T = RefT>
        std::enable_if_t<T::is_mut_type::value, NodeT&>
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
        const SplitT& get_split() const; /* internal only */
        LeafT leaf_value() const; /* leaf only */

        template <typename SplitT = typename RefT::SplitT>
        void get_domains(DomainsT& domains);

        template <typename T = RefT>
        std::enable_if_t<T::is_mut_type::value, void>
        set_leaf_value(LeafT value); /* leaf & mut only */

        template <typename T = RefT>
        std::enable_if_t<T::is_mut_type::value, void>
        split(SplitT split); /* leaf & mut only */

        template <typename T = RefT>
        std::enable_if_t<T::is_mut_type::value, void>
        skip_branch(); /* non-root & mut only */
    };

    template <typename RefT>
    std::ostream& operator<<(std::ostream& s, const NodeRef<RefT>& n);


    enum TreeVisitStatus {
        ADD_NONE = 0,
        ADD_LEFT = 1,
        ADD_RIGHT = 2,
        ADD_LEFT_AND_RIGHT = 3
    };

    template <typename SplitT, typename LeafT>
    class Tree {
    public:
        using CRef = NodeRef<inner::ConstRef<SplitT, LeafT>>;
        using MRef = NodeRef<inner::MutRef<SplitT, LeafT>>;

    private:
        friend MRef;
        friend CRef;

        std::vector<inner::Node<SplitT, LeafT>> nodes_;

    public:
        Tree();
        void split(NodeId node_id, SplitT split);

        CRef root() const;
        MRef root();

        CRef operator[](NodeId index) const;
        MRef operator[](NodeId index);

        int num_nodes() const;
        int num_leafs() const;

        template <typename TreeVisitorT>
        void dfs(TreeVisitorT& visitor) const;

        template <typename TreeVisitorT>
        void dfs(TreeVisitorT&& visitor) const;

        template <typename Archive>
        void serialize(Archive& archive);

        std::string to_json() const;
        static Tree from_json(const std::string& json);
    };

    template <typename SplitT, typename LeafT>
    std::ostream& operator<<(std::ostream& s, const Tree<SplitT, LeafT>& t);


    class AddTree {
    public:
        using TreeT = Tree<Split, FloatT>;
        using SplitMapT = std::unordered_map<FeatId, std::vector<FloatT>>;

    private:
        std::vector<TreeT> trees_;

    public:
        FloatT base_score;

        AddTree();

        size_t add_tree(TreeT&& tree);
        size_t size() const;
        size_t num_nodes() const;
        size_t num_leafs() const;

        TreeT& operator[](size_t index);
        const TreeT& operator[](size_t index) const;
        const std::vector<TreeT>& trees() const;

        SplitMapT get_splits() const;

        std::string to_json() const;
        static AddTree from_json(const std::string& json);
        static AddTree from_json_file(const char *file);

        template <typename Archive>
        void serialize(Archive& archive);
    };

    std::ostream& operator<<(std::ostream& s, const AddTree& at);

} /* namespace treeck */

#endif /* TREECK_TREE_H */
