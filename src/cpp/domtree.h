#ifndef TREECK_DOMTREE_H
#define TREECK_DOMTREE_H

#include <string>
#include <utility>
#include <ostream>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <type_traits>
#include <optional>

#include "domain.h"
#include "tree.h"

namespace treeck {

    struct IsReachableKey {
        int tree_index;
        NodeId node_id;

        template <typename Archive>
        void serialize(Archive& archive);
    };

    bool operator==(const IsReachableKey& a, const IsReachableKey& b);

    struct IsReachableKeyHash {
        size_t operator()(const IsReachableKey& k) const;

        template <typename Archive>
        void serialize(Archive& archive);
    };

    class IsReachable {
        std::unordered_set<IsReachableKey, IsReachableKeyHash> unreachable_;

    public:
        IsReachable();
        IsReachable(const IsReachable& other);
        IsReachable(IsReachable&& other);

        IsReachable& operator=(const IsReachable& other);
        IsReachable& operator=(IsReachable&& other);

        size_t num_unreachable() const;
        bool is_reachable(size_t tree_index, NodeId node_id) const;
        void mark_unreachable(size_t tree_index, NodeId node_id);

        void combine(const IsReachable& other);

        template <typename Archive>
        void serialize(Archive& archive);
    };

    using DomainsT = std::unordered_map<FeatId, Domain>;
    using ReachableT = std::unordered_map<NodeId, IsReachable>; // domtree_node_id -> IsReachble

    class DomTree;
    class DomTreeLeaf;


    struct Nothing {
        template <typename Archive>
        void serialize(Archive& archive);
    };

    struct DomTreeSplit {
        size_t instance_index;
        Split split;

        template <typename Archive>
        void serialize(Archive& archive);
    };

    struct DomTreeInstance {
        friend DomTree;

        size_t index;
        std::shared_ptr<AddTree> addtree;
        DomainsT root_domains;
        ReachableT is_reachables;
    };

    struct DomTreeLeafInstance {
        friend DomTree;

        std::shared_ptr<AddTree> addtree;
        DomainsT domains;
        IsReachable is_reachable;

        template <typename Archive>
        void serialize(Archive& archive);
    };

    std::ostream&
    operator<<(std::ostream& s, const Nothing& t);

    std::ostream&
    operator<<(std::ostream& s, const DomTreeSplit& t);






    class DomTree {
    public:
        using DomTreeT = Tree<DomTreeSplit, Nothing>;

    private:
        DomTreeT tree_;
        std::vector<DomTreeInstance> instances_;

    public:
        DomTree();
        const DomTreeT& tree() const;
        size_t num_instances() const;
        std::shared_ptr<AddTree> addtree(size_t instance) const;

        void add_instance(std::shared_ptr<AddTree> addtree, DomainsT&& domains);

        std::optional<Domain>
        get_root_domain(size_t instance, FeatId feat_id) const;

        DomainsT
        get_domains(size_t instance, NodeId domtree_leaf_id) const;

        DomTreeLeaf get_leaf(NodeId domtree_leaf_id) const;
        void apply_leaf(DomTreeLeaf&& leaf);

    private:
        void update_is_reachable(size_t instance, NodeId domtree_leaf_id,
                FeatId feat_id, Domain new_dom);
    };







    class DomTreeLeaf {
        NodeId domtree_leaf_id_;
        std::vector<DomTreeLeafInstance> instances_;
        std::optional<DomTreeSplit> best_split_;

        friend DomTree;

    public:
        int score;
        int balance;

        DomTreeLeaf(NodeId DomTreeLeaf,
                std::vector<DomTreeLeafInstance>&& instances_);

        //void set_addtree(size_t instance, AddTree& addtree);

        NodeId domtree_leaf_id() const;
        size_t num_instances() const;
        std::shared_ptr<AddTree> addtree(size_t instance) const;
        std::optional<DomTreeSplit> get_best_split() const;

        const DomainsT& get_domains(size_t instance) const;
        std::optional<Domain> get_domain(size_t instance, FeatId) const;

        size_t num_unreachable(size_t instance) const;
        bool is_reachable(size_t instance, size_t tree_index, NodeId) const;
        void mark_unreachable(size_t instance, size_t tree_index, NodeId);

        void find_best_split();
        bool find_best_split_for_instance(size_t instance, Split& max_split,
                int& max_score, int& min_balance);

        int count_unreachable_leafs(size_t instance,
                FeatId feat_id, Domain new_dom) const;

        std::tuple<FloatT, FloatT>
        get_tree_bounds(size_t instance, size_t tree_index);

        static DomTreeLeaf merge(const std::vector<DomTreeLeaf>& leafs);

        void to_binary(std::ostream& os) const;
        static DomTreeLeaf from_binary(std::istream& is);
    };

} /* namespace treeck */

#endif /* TREECK_DOMTREE_H */
