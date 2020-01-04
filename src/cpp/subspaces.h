#ifndef TREECK_SUBSPACES_H
#define TREECK_SUBSPACES_H

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

        bool is_reachable(size_t tree_index, NodeId node_id) const;
        void mark_unreachable(size_t tree_index, NodeId node_id);

        void combine(const IsReachable& other);

        template <typename Archive>
        void serialize(Archive& archive);
    };

    /*
     *   +------------(Sub-)space---------------+
     *   |                                      |
     *   |  +-----------+                       |
     *   |  | Attribute |                       |
     *   |  +-----------+                       |
     *   |       |                              |
     *   |       | has                          |
     *   |       v                              |
     *   |  +-----------+                       |
     *   |  | Domain    +-----> RealDomain      |
     *   |  +-----------+  |                    |
     *   |                 +--> BoolDomain      |
     *   |                                      |
     *   +--------------------------------------+
     *
     * A (sub-)space defines (additionally constrained) domains for each
     * feature. A (sub-)space can be split to produce more constrained
     * sub-spaces. This concept is captured by the `Subspace` class.
     *
     * The `Subspaces` class keeps track of all produced sub-spaces.
     *
     * The more constrained a sub-space is, the more nodes in a tree become
     * unreachable.
     */


    class Subspace;


    class Subspaces {
    public:
        using DomTreeT = Tree<FloatT>; /* we don't use the float */
        using DomainsT = std::unordered_map<FeatId, Domain>;
        using ReachableT = std::unordered_map<NodeId, IsReachable>;

    private:
        std::shared_ptr<const AddTree> addtree_;
        DomTreeT domtree_;

        DomainsT root_domains_;
        ReachableT is_reachables_;

    public:
        Subspaces(std::shared_ptr<const AddTree> addtree, DomainsT root_domains);

        const AddTree& addtree() const;
        const DomTreeT& domtree() const;

        Domain get_root_domain(FeatId) const;

        void get_leaf_domains(NodeId domtree_leaf_id, DomainsT& domains) const;
        Subspace get_subspace(NodeId domtree_leaf_id);

        void split_domtree_leaf(NodeId domtree_leaf_id);
        void split(Subspace&& leaf);

        std::string to_json() const;
        static Subspaces from_json(
                std::shared_ptr<const AddTree> addtree,
                const std::string& json);

    private:
        void update_is_reachable(IsReachable& is_reachable,
                FeatId feat_id, Domain new_dom);
        void update_is_reachable(IsReachable& is_reachable,
                size_t tree_index,
                AddTree::TreeT::CRef node,
                FeatId feat_id,
                Domain new_dom,
                bool marked);
    };


    class Subspace {
        NodeId domtree_node_id_;
        IsReachable is_reachable_;
        std::optional<Split> best_split_;

        friend Subspaces;

    public:
        int split_score;
        int split_balance;

        using SplitMapT = std::unordered_map<FeatId, std::vector<FloatT>>;

        Subspace(const Subspace& other);
        Subspace(Subspace&& other);

        Subspace(
                NodeId domtree_node_id,
                const IsReachable& is_reachable);

        Subspace(
                NodeId domtree_node_id,
                IsReachable&& is_reachable);

        Subspace& operator=(const Subspace& other);
        Subspace& operator=(Subspace&& other);

        NodeId domtree_node_id() const;
        bool is_reachable(size_t tree_index, NodeId node_id) const;
        void mark_unreachable(size_t tree_index, NodeId node_id);

        void find_best_domtree_split(const AddTree& addtree);
        Split get_best_split() const;
        std::tuple<FloatT, FloatT> get_tree_bounds(const AddTree& at, size_t tree_index);

        static Subspace merge(const std::vector<Subspace>& leafs);

        std::string to_json() const;
        static Subspace from_json(const std::string& json);

    private:
        int count_unreachable_leafs(
                const AddTree& addtree,
                FeatId feat_id,
                Domain new_dom) const;

        int count_unreachable_leafs(
                const AddTree& addtree,
                size_t tree_index,
                AddTree::TreeT::CRef node,
                FeatId feat_id,
                Domain new_dom,
                bool marked) const;
    };

} /* namespace treeck */

#endif /* TREECK_SUBSPACES_H */
