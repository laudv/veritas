#include <functional>
#include <exception>
#include <iostream>
#include <utility>
#include <map>
#include <limits>
#include <cmath>

#include "domain.h"
#include "util.h"
#include "tree.hpp"

#include "splittree.h"

namespace treeck {

    LeafSplitInfo::LeafSplitInfo() {}

    template <typename Archive>
    void
    LeafSplitInfo::serialize(Archive& archive) {}

    std::ostream&
    operator<<(std::ostream& s, LeafSplitInfo info) { return s; }


    bool
    operator==(const IsReachableKey& a, const IsReachableKey& b)
    {
        return a.tree_index == b.tree_index
            && a.node_id == b.node_id;
    }


    size_t
    IsReachableKeyHash::operator()(const IsReachableKey& k) const
    {
        // https://www.boost.org/doc/libs/1_34_0/doc/html/boost/hash_combine.html
        size_t seed = 0x33434282;
        seed ^= std::hash<size_t>()(k.tree_index) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<NodeId>()(k.node_id)    + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    IsReachable::IsReachable() : unreachable_() {}

    bool
    IsReachable::is_reachable(size_t tree_index, NodeId node_id) const
    {
        IsReachableKey k{static_cast<int>(tree_index), node_id};
        return unreachable_.find(k) == unreachable_.end(); // not contains
    }

    void
    IsReachable::mark_unreachable(size_t tree_index, NodeId node_id)
    {
        IsReachableKey k{static_cast<int>(tree_index), node_id};
        unreachable_.insert(k);
    }



    SplitTree::SplitTree(std::shared_ptr<const AddTree> addtree)
        : addtree_(addtree)
        , domtree_()
        , splits_()
        , root_domains_()
        , reachable_()
    {
        reachable_.emplace(domtree_.root().id(), IsReachable());
    }

    const AddTree&
    SplitTree::addtree() const { return *addtree_; }

    const SplitTree::DomTreeT&
    SplitTree::domtree() const { return domtree_; }

    RealDomain
    SplitTree::get_root_domain(FeatId feat_id) const
    {
        auto search = root_domains_.find(feat_id);
        if (search != root_domains_.end())
            return search->second;
        return RealDomain(); // unconstrained domain
    }

    void
    SplitTree::set_root_domain(FeatId feat_id, RealDomain dom)
    {
        root_domains_.insert_or_assign(feat_id, dom);
    }

    RealDomain
    SplitTree::get_domain(NodeId domtree_node_id, FeatId feat_id) const
    {
        RealDomain dom = get_root_domain(feat_id);
        DomTreeT::CRef node = domtree_[domtree_node_id];
        while (!node.is_root())
        {
            DomTreeT::CRef child_node = node;
            node = node.parent();

            LtSplit split = std::get<LtSplit>(node.get_split());
            double sval = split.split_value;

            if (split.feat_id != feat_id) continue;
            if (child_node.is_left_child())
                if (dom.hi > sval) dom.hi = sval;
            else
                if (dom.lo < sval) dom.lo = sval;
        }
        return dom;
    }

    bool
    SplitTree::is_reachable(
            NodeId domtree_node_id,
            size_t tree_index,
            NodeId addtree_node_id) const
    {
        // We're checking whether node `addtree_node_id` in
        // `addtree[tree_index]` is reachable given the domains defined by node
        // `domtree_node_id` in domtree.
        //
        // A node in an addtree tree is unreachable if
        //    (1) it or one of the nodes on the path from it to the root are
        //        explicitly marked unreachable, or
        //    (2) the domains defined by the splits in the domtree make a left
        //        or right subtree unreachable.
        //
        // To avoid having to check (2), we explicitly mark all newly
        // unreachable nodes when splitting. See SplitTree<>::split.
        // Additionally, we duplicate each explicitly marked unreachable nodes
        // in each leaf (unreachable_[domtree_leaf_node_id], i.e.,
        // unreachable_[node_id] where node_id is internal would be invalid,
        // except for the root). This way, we only need to check the reachable
        // structures for the domtree leaf node and the domtree root node.
        
        if (!domtree_[domtree_node_id].is_leaf())
            throw std::runtime_error("SplitTree::is_reachable on non-leaf domtree node");

        const auto& reachable_root = reachable_.at(domtree_.root().id());
        const auto& reachable_leaf = reachable_.at(domtree_node_id);

        return reachable_root.is_reachable(tree_index, addtree_node_id)
            && reachable_leaf.is_reachable(tree_index, addtree_node_id);
    }

    void
    SplitTree::mark_unreachable(
            NodeId domtree_node_id,
            size_t tree_index,
            NodeId addtree_node_id)
    {
        if (!domtree_[domtree_node_id].is_leaf())
            throw std::runtime_error("SplitTree::mark_unreachable on non-leaf domtree node");

        auto& reachable = reachable_.at(domtree_node_id);
        reachable.mark_unreachable(tree_index, addtree_node_id);
    }

    void
    SplitTree::split(NodeId domtree_node_id)
    {
        size_t tree_index = 0;
        const auto& reachable = reachable_.at(domtree_node_id);
        std::unordered_map<FeatId, std::map<double, std::pair<int, int>>> counts;
        std::unordered_set<FeatId> seen;

        // Count the number of leafs hidden behind each split
        for (const AddTree::TreeT& tree : addtree_->trees())
        {
            if (tree.root().is_leaf()) continue;
            tree.dfs([tree_index,
                      &seen,
                      &reachable,
                      &counts]
            (AddTree::TreeT::CRef node)
            {
                if (!reachable.is_reachable(tree_index, node.id()))
                    return TreeVisitStatus::ADD_NONE;
                if (node.is_internal())
                    return TreeVisitStatus::ADD_LEFT_AND_RIGHT;
                seen.clear();
                do
                {
                    auto parent = node.parent();
                    LtSplit split = std::get<LtSplit>(parent.get_split());
                    FeatId feat_id = split.feat_id;
                    double sval = split.split_value;

                    if (seen.find(feat_id) == seen.end()) // we haven't +1'ed this feat_id yet
                    {
                        seen.insert(feat_id);
                        if (node.is_left_child())
                            counts[feat_id][sval].first += 1;
                        else
                            counts[feat_id][sval].second += 1;
                    }

                    node = parent;
                } while (!node.is_root());
                return TreeVisitStatus::ADD_NONE;
            });

            tree_index += 1;
        }

        std::vector<std::pair<int, int>> accum;
        int max_unreachable = 0;
        FeatId max_feat_id = -1;
        double max_split_value = std::numeric_limits<double>::quiet_NaN();
        int min_left_right_diff = -1;

        for (const auto& n : counts)
        {
            FeatId feat_id = n.first;
            const std::map<double, std::pair<int, int>>& ft_cnts = n.second;
            accum.resize(ft_cnts.size());

            {
                int acc_lt = 0, acc_ge = 0;
                auto fm = ft_cnts.cbegin(); // forward map iter
                auto rm = ft_cnts.crbegin(); // reverse map iter
                auto fa = accum.begin(); // forward accum iter
                auto ra = accum.rbegin(); // reverse accum iter
                for (; fm != ft_cnts.cend(); ++fm, ++rm, ++fa, ++ra)
                {
                    acc_lt += fm->second.first;
                    fa->first = acc_lt;
                    acc_lt += fm->second.second;
                    acc_ge += rm->second.second;
                    ra->second = acc_ge;
                    acc_ge += rm->second.first;
                }
            }

            std::cout << " + feat_id=" << n.first << std::endl;
            auto fm = ft_cnts.cbegin(); // forward map iter
            for (const auto& m : accum)
                std::cout << "    - " << (fm++)->first << " => "
                    << m.first << ", " << m.second << " : "
                    << m.first + m.second
                    << std::endl;

            {
                auto fm = ft_cnts.cbegin();
                auto fa = accum.cbegin();
                for (; fm != ft_cnts.cend(); ++fm, ++fa)
                {
                    int num_unreachable = fa->first + fa->second;
                    int left_right_diff = std::abs(fa->first - fa->second);
                    if (max_unreachable <= num_unreachable)
                    {
                        if (max_unreachable < num_unreachable
                                || left_right_diff < min_left_right_diff)
                        {
                            max_unreachable = num_unreachable;
                            max_feat_id = feat_id;
                            max_split_value = fm->first;
                            min_left_right_diff = left_right_diff;

                            std::cout << "BEST! " << max_unreachable
                                << ", " << max_feat_id
                                << ", " << max_split_value
                                << ", lr " << min_left_right_diff
                                << std::endl;
                        }
                    }
                }
            }
        }

        // TODO
        // Split function up
        // loop over trees and check how many leafs are actually discarded given best split
    }


} /* namespace treeck */
