#include <functional>
#include <exception>
#include <iostream>
#include <map>
#include <limits>
#include <cmath>
#include <cstdio>

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
        splits_ = addtree_->get_splits();
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

        // TODO mark unreachable nodes
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

    void
    SplitTree::get_domains(NodeId domtree_node_id,
            std::unordered_map<FeatId, RealDomain>& domains) const
    {
        DomTreeT::CRef node = domtree_[domtree_node_id];
        while (!node.is_root())
        {
            DomTreeT::CRef child_node = node;
            node = node.parent();

            LtSplit split = std::get<LtSplit>(node.get_split());
            double sval = split.split_value;
            auto& dom = domains[split.feat_id];

            if (child_node.is_left_child())
                if (dom.hi > sval) dom.hi = sval;
            else
                if (dom.lo < sval) dom.lo = sval;
        }
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
        if (!domtree_[domtree_node_id].is_leaf())
            throw std::runtime_error("already split");

        FeatId max_feat_id = -1;
        double max_split_value = std::numeric_limits<double>::quiet_NaN();
        int max_score = 0;
        int min_balance = -1;

        std::unordered_map<FeatId, RealDomain> domains;
        get_domains(domtree_node_id, domains);

        for (auto&& [feat_id, splits] : splits_)
        for (double sval : splits)
        {
            auto domptr = domains.find(feat_id); // only splits that are still in domain
            RealDomain dom = domptr == domains.end() ? RealDomain() : domptr->second;
            if (!dom.contains_strict(sval))
                continue;

            RealDomain dom_l, dom_r;
            std::tie(dom_l, dom_r) = dom.split(sval);

            int unreachable_l = count_unreachable_leafs(domtree_node_id, feat_id, dom_l);
            int unreachable_r = count_unreachable_leafs(domtree_node_id, feat_id, dom_r);

            int score = unreachable_l + unreachable_r;
            int balance = std::abs(unreachable_l - unreachable_r);

            printf("feat_id=%d, split_value=%.10f, score=%d, balance=%d\n",
                    feat_id, sval, score, balance);

            if (score >= max_score)
            if (score > max_score || min_balance > balance)
            {
                max_feat_id = feat_id;
                max_split_value = sval;
                max_score = score;
                min_balance = balance;
            }
        }

        printf("best split: X%d <> %.10f with score %d, balance %d\n",
                max_feat_id, max_split_value, max_score, min_balance);

        // TODO perform split => update domtree, update reachabilities
    }

    int
    SplitTree::count_unreachable_leafs(
            NodeId domtree_node_id,
            FeatId feat_id,
            RealDomain new_dom) const
    {
        int unreachable = 0;
        size_t tree_index = 0;
        for (const AddTree::TreeT& tree : addtree_->trees())
        {
            unreachable += count_unreachable_leafs(domtree_node_id, tree_index,
                    tree.root(), feat_id, new_dom, false);
            ++tree_index;
        }
        return unreachable;
    }

    int
    SplitTree::count_unreachable_leafs(
            NodeId domtree_node_id,
            size_t tree_index,
            AddTree::TreeT::CRef node,
            FeatId feat_id,
            RealDomain new_dom,
            bool marked) const
    {
        if (node.is_leaf())
            return marked ? 1 : 0;
        if (!is_reachable(domtree_node_id, tree_index, node.id()))
            return 0;

        LtSplit split = std::get<LtSplit>(node.get_split());

        bool marked_l = marked; // remain marked if already marked
        bool marked_r = marked;
        if (!marked && split.feat_id == feat_id)
        {
            //       case 1       case 3          case 2
            //       [----)   |-------------)     |----)
            // ---------------------x-------------------------->
            //                 split_value
            //
            switch (new_dom.where_is_strict(split.split_value))
            {
            case WhereFlag::LEFT: // case 2: split value to the left of the domain
                marked_l = true; // left becomes unreachable -> start counting
                break;
            case WhereFlag::RIGHT: // case 1
                marked_r = true; // right becomes unreachable -> start counting
                break;
            default: // case 3: IN_DOMAIN
                break; // both branches still reachable
            }
        }

        return //((marked_l || marked_r) ? 1 : 0) + // this node's split became deterministic
            count_unreachable_leafs(domtree_node_id, tree_index, node.left(),
                    feat_id, new_dom, marked_l) +
            count_unreachable_leafs(domtree_node_id, tree_index, node.right(),
                    feat_id, new_dom, marked_r);
    }

} /* namespace treeck */
