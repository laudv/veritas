#ifndef TREECK_SPLITTREE_H
#define TREECK_SPLITTREE_H

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

    struct LeafSplitInfo {
        LeafSplitInfo();

        template <typename Archive>
        void serialize(Archive& archive);
    };

    std::ostream& operator<<(std::ostream& s, LeafSplitInfo info);

    struct IsReachableKey {
        int tree_index;
        NodeId node_id;
    };

    bool operator==(const IsReachableKey& a, const IsReachableKey& b);

    struct IsReachableKeyHash {
        size_t operator()(const IsReachableKey& k) const;
    };

    class IsReachable {
        std::unordered_set<IsReachableKey, IsReachableKeyHash> unreachable_;

    public:
        IsReachable();
        IsReachable(const IsReachable& other);
        IsReachable(IsReachable&& other);

        bool is_reachable(size_t tree_index, NodeId node_id) const;
        void mark_unreachable(size_t tree_index, NodeId node_id);
    };


    class SplitTreeLeaf;


    class SplitTree {
    public:
        using DomTreeT = Tree<LeafSplitInfo>;
        using TreeIndexT = int;
        using DomainsT = std::unordered_map<FeatId, RealDomain>;
        using ReachableT = std::unordered_map<NodeId, IsReachable>;

    private:
        std::shared_ptr<const AddTree> addtree_;
        DomTreeT domtree_;
        AddTree::SplitMapT splits_;

        DomainsT root_domains_;
        ReachableT is_reachables_;

    public:
        SplitTree(std::shared_ptr<const AddTree> addtree, DomainsT root_domains);

        const AddTree& addtree() const;
        const DomTreeT& domtree() const;

        RealDomain get_root_domain(FeatId) const;

        void get_leaf_domains(NodeId domtree_node_id, DomainsT& domains) const;
        SplitTreeLeaf get_leaf(NodeId domtree_node_id);

        /*
        bool is_reachable(NodeId domtree_node_id, size_t tree_index,
                NodeId addtree_node_id) const;
        void mark_unreachable(NodeId domtree_node_id, size_t tree_index,
                NodeId addtree_node_id);

        void split(NodeId domtree_node_id);
        */

        void split_domtree_leaf(NodeId domtree_node_id);
        void split(SplitTreeLeaf&& leaf);

    private:
        void update_is_reachable(IsReachable& is_reachable,
                FeatId feat_id, RealDomain new_dom);
        void update_is_reachable(IsReachable& is_reachable,
                size_t tree_index,
                AddTree::TreeT::CRef node,
                FeatId feat_id,
                RealDomain new_dom,
                bool marked);
    };


    class SplitTreeLeaf {
        NodeId domtree_node_id_;
        IsReachable is_reachable_;
        std::optional<LtSplit> best_split_;

        friend SplitTree;

    public:
        using SplitMapT = std::unordered_map<FeatId, std::vector<double>>;

        SplitTreeLeaf(
                NodeId domtree_node_id,
                IsReachable is_reachable);

        bool is_reachable(size_t tree_index, NodeId node_id) const;
        void mark_unreachable(size_t tree_index, NodeId node_id);

        void find_best_domtree_split(const AddTree& addtree);

        template <typename Archive>
        void archive(Archive& archive);

    private:
        int count_unreachable_leafs(
                const AddTree& addtree,
                FeatId feat_id,
                RealDomain new_dom) const;

        int count_unreachable_leafs(
                const AddTree& addtree,
                size_t tree_index,
                AddTree::TreeT::CRef node,
                FeatId feat_id,
                RealDomain new_dom,
                bool marked) const;
    };

} /* namespace treeck */

#endif /* TREECK_SPLITTREE_H */
