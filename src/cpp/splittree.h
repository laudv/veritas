#ifndef TREECK_SPLITTREE_H
#define TREECK_SPLITTREE_H

#include <utility>
#include <ostream>
#include <memory>
#include <unordered_set>
#include <unordered_map>

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

        bool is_reachable(size_t tree_index, NodeId node_id) const;
        void mark_unreachable(size_t tree_index, NodeId node_id);
    };


    class SplitTree {
    public:
        using DomTreeT = Tree<LeafSplitInfo>;
        using TreeIndexT = int;

    private:
        std::shared_ptr<const AddTree> addtree_;
        DomTreeT domtree_;
        AddTree::SplitMapT splits_;

        std::unordered_map<FeatId, RealDomain> root_domains_;
        std::unordered_map<NodeId, IsReachable> reachable_;

    public:
        SplitTree(std::shared_ptr<const AddTree> addtree);

        const AddTree& addtree() const;
        const DomTreeT& domtree() const;

        RealDomain get_root_domain(FeatId) const;
        void set_root_domain(FeatId, RealDomain);
        RealDomain get_domain(NodeId domtree_node_id, FeatId) const;
        void get_domains(NodeId domtree_node_id,
                std::unordered_map<FeatId, RealDomain>& domains) const;

        bool is_reachable(NodeId domtree_node_id, size_t tree_index,
                NodeId addtree_node_id) const;
        void mark_unreachable(NodeId domtree_node_id, size_t tree_index,
                NodeId addtree_node_id);

        void split(NodeId domtree_node_id);

    private:
        int
        count_unreachable_leafs(
                NodeId domtree_node_id,
                FeatId feat_id,
                RealDomain new_dom) const;

        int
        count_unreachable_leafs(
                NodeId domtree_node_id,
                size_t tree_index,
                AddTree::TreeT::CRef node,
                FeatId feat_id,
                RealDomain new_dom,
                bool marked) const;
    };

} /* namespace treeck */

#endif /* TREECK_SPLITTREE_H */
