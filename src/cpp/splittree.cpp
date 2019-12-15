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
    IsReachable::IsReachable(const IsReachable& o) : unreachable_(o.unreachable_) {}
    IsReachable::IsReachable(IsReachable&& o) : unreachable_(std::move(o.unreachable_)) {}

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



    SplitTree::SplitTree(std::shared_ptr<const AddTree> addtree, SplitTree::DomainsT domains)
        : addtree_(addtree)
        , domtree_()
        , splits_()
        , root_domains_()
        , is_reachables_()
    {
        splits_ = addtree_->get_splits();
        is_reachables_.emplace(domtree_.root().id(), IsReachable());
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

    /*
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
    */

    void
    SplitTree::get_leaf_domains(NodeId domtree_node_id,
            SplitTree::DomainsT& domains) const
    {
        DomTreeT::CRef node = domtree_[domtree_node_id];
        while (!node.is_root())
        {
            DomTreeT::CRef child_node = node;
            node = node.parent();

            LtSplit split = std::get<LtSplit>(node.get_split());
            double sval = split.split_value;

            auto domptr = domains.find(split.feat_id);
            if (domptr == domains.end())
                domains[split.feat_id] = get_root_domain(split.feat_id);
            auto& dom = domains[split.feat_id];

            if (child_node.is_left_child())
            {
                if (dom.hi > sval) dom.hi = sval;
            }
            else
            {
                if (dom.lo < sval) dom.lo = sval;
            }
        }
    }

    SplitTreeLeaf
    SplitTree::get_leaf(NodeId domtree_node_id)
    {
        auto node = domtree_[domtree_node_id];
        if (!node.is_leaf())
            throw std::runtime_error("SplitTree::get_leaf on non-leaf domtree node");

        DomainsT leaf_domains;
        get_leaf_domains(domtree_node_id, leaf_domains);

        // SplitTreeLeaf owns all its values so that we can easily transmit it
        // over the network to worker nodes. The structures should be
        // reasonably small.
        return SplitTreeLeaf(
            domtree_node_id,
            is_reachables_.at(domtree_node_id)
        );
    }

    void // shorthand method
    SplitTree::split_domtree_leaf(NodeId domtree_node_id)
    {
        split(get_leaf(domtree_node_id));
    }

    void
    SplitTree::split(SplitTreeLeaf&& leaf)
    {
        auto node = domtree_[leaf.domtree_node_id_];
        if (!node.is_leaf())
            throw std::runtime_error("SplitTree::split on non-leaf");
        if (is_reachables_.find(node.id()) == is_reachables_.end())
            throw std::runtime_error("SplitTree::split assertion error: no is_reachable for this node");
        if (!leaf.best_split_.has_value())
            leaf.find_best_domtree_split(*addtree_);

        node.split(*leaf.best_split_);

        is_reachables_.erase(node.id()); // might be old! -> mark_unreachable)
        is_reachables_.emplace(node.left().id(), leaf.is_reachable_); // copy once
        is_reachables_.emplace(node.right().id(), std::move(leaf.is_reachable_)); // reuse for right

        FeatId feat_id = leaf.best_split_->feat_id;
        RealDomain dom_l, dom_r;
        std::tie(dom_l, dom_r) = RealDomain().split(leaf.best_split_->split_value);

        auto& is_reachable_l = is_reachables_.at(node.left().id());
        update_is_reachable(is_reachable_l, feat_id, dom_l);
        auto& is_reachable_r = is_reachables_.at(node.right().id());
        update_is_reachable(is_reachable_l, feat_id, dom_r);
    }

    void
    SplitTree::update_is_reachable(IsReachable& is_reachable,
                FeatId feat_id, RealDomain new_dom)
    {
        std::cout << "update_is_reachable " << feat_id << ", " << new_dom << std::endl;

        size_t tree_index = 0;
        for (auto& tree : addtree_->trees())
        {
            update_is_reachable(is_reachable, tree_index, tree.root(), feat_id,
                    new_dom, false);
            ++tree_index;
        }
    }
    void
    SplitTree::update_is_reachable(IsReachable& is_reachable,
                size_t tree_index,
                AddTree::TreeT::CRef node,
                FeatId feat_id,
                RealDomain new_dom,
                bool marked)
    {
        if(!is_reachable.is_reachable(tree_index, node.id()))
            return;
        if (marked)
        {
            std::cout << "marking " << tree_index << '-' << node.id() << std::endl;
            is_reachable.mark_unreachable(tree_index, node.id());
        }
        if (node.is_leaf())
            return;

        LtSplit split = std::get<LtSplit>(node.get_split());
        double sval = split.split_value;

        bool marked_l = marked;
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
                marked_l = true; // left becomes unreachable -> start marking
                break;
            case WhereFlag::RIGHT: // case 1
                marked_r = true; // right becomes unreachable -> start marking
                break;
            default: // case 3: IN_DOMAIN
                break; // both branches still reachable
            }
        }

        update_is_reachable(is_reachable, tree_index, node.left(), feat_id, new_dom, marked_l);
        update_is_reachable(is_reachable, tree_index, node.right(), feat_id, new_dom, marked_r);
    }

    /*

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
        get_leaf_domains(domtree_node_id, domains);

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

        //domtree_.split(domtree_node_id
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
    */


    /* --------------------------------------------------------------------- */

    SplitTreeLeaf::SplitTreeLeaf(NodeId domtree_node_id,
            IsReachable is_reachable)
        : domtree_node_id_(domtree_node_id)
        , is_reachable_(is_reachable)
        , best_split_() {}

    bool
    SplitTreeLeaf::is_reachable(size_t tree_index, NodeId node_id) const
    {
        return is_reachable_.is_reachable(tree_index, node_id);
    }

    void
    SplitTreeLeaf::mark_unreachable(size_t tree_index, NodeId node_id)
    {
        is_reachable_.mark_unreachable(tree_index, node_id);
    }

    void
    SplitTreeLeaf::find_best_domtree_split(const AddTree& addtree)
    {
        size_t tree_index;
        std::unordered_map<FeatId, std::unordered_set<double>> duplicates;

        FeatId max_feat_id = -1;
        double max_split_value = std::numeric_limits<double>::quiet_NaN();
        int max_score = 0;
        int min_balance = -1;

        for (auto& tree : addtree.trees())
        {
            tree.dfs([this,
                      &addtree,
                      &duplicates,
                      tree_index,
                      &max_feat_id,
                      &max_split_value,
                      &max_score,
                      &min_balance]
                    (AddTree::TreeT::CRef node) {
                if (node.is_leaf() || !this->is_reachable_.is_reachable(
                            tree_index, node.id()))
                    return TreeVisitStatus::ADD_NONE;

                LtSplit split = std::get<LtSplit>(node.get_split());
                FeatId feat_id = split.feat_id;
                double sval = split.split_value;

                auto& feat_id_dups = duplicates[split.feat_id]; // auto-initialize set for feat_id
                auto p = feat_id_dups.find(split.split_value);
                if (p != feat_id_dups.end()) // already checked
                    return TreeVisitStatus::ADD_LEFT_AND_RIGHT;
                feat_id_dups.insert(split.split_value);

                // compute the number of unreachable nodes when we split the domain
                // on split.feat_id <> split.split_value

                RealDomain dom_l, dom_r;
                std::tie(dom_l, dom_r) = RealDomain().split(sval);

                int unreachable_l = this->count_unreachable_leafs(addtree, feat_id, dom_l);
                int unreachable_r = this->count_unreachable_leafs(addtree, feat_id, dom_r);
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

                return TreeVisitStatus::ADD_LEFT_AND_RIGHT;
            });
            ++tree_index;
        }

        printf("best split: X%d <> %.10f with score %d, balance %d\n",
                max_feat_id, max_split_value, max_score, min_balance);

        best_split_.emplace(max_feat_id, max_split_value);
    }

    int
    SplitTreeLeaf::count_unreachable_leafs(
            const AddTree& addtree,
            FeatId feat_id,
            RealDomain new_dom) const
    {
        int unreachable = 0;
        size_t tree_index = 0;
        for (const AddTree::TreeT& tree : addtree.trees())
        {
            unreachable += count_unreachable_leafs(addtree, tree_index,
                    tree.root(), feat_id, new_dom, false);
            ++tree_index;
        }
        return unreachable;
    }

    int
    SplitTreeLeaf::count_unreachable_leafs(
            const AddTree& addtree,
            size_t tree_index,
            AddTree::TreeT::CRef node,
            FeatId feat_id,
            RealDomain new_dom,
            bool marked) const
    {
        if (node.is_leaf())
            return marked ? 1 : 0;
        if (!is_reachable(tree_index, node.id()))
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
            count_unreachable_leafs(addtree, tree_index, node.left(),
                    feat_id, new_dom, marked_l) +
            count_unreachable_leafs(addtree, tree_index, node.right(),
                    feat_id, new_dom, marked_r);
    }

} /* namespace treeck */
