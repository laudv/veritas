#include <functional>
#include <exception>
#include <optional>
#include <sstream>
#include <iostream>
#include <map>
#include <limits>
#include <cmath>
#include <cstdio>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/optional.hpp>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "cereal/cereal.hpp"
#include "domain.h"
#include "tree.h"
#include "util.h"
#include "tree.hpp"

#include "splittree.h"

namespace treeck {

    bool
    operator==(const IsReachableKey& a, const IsReachableKey& b)
    {
        return a.tree_index == b.tree_index
            && a.node_id == b.node_id;
    }


    template <typename Archive>
    void
    IsReachableKey::serialize(Archive& archive)
    {
        archive(CEREAL_NVP(tree_index), CEREAL_NVP(node_id));
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

    template <typename Archive>
    void
    IsReachableKeyHash::serialize(Archive& archive) {}

    IsReachable::IsReachable() : unreachable_() {}
    IsReachable::IsReachable(const IsReachable& o) : unreachable_(o.unreachable_) {}
    IsReachable::IsReachable(IsReachable&& o)
        : unreachable_(std::move(o.unreachable_)) {}

    IsReachable&
    IsReachable::operator=(const IsReachable& other)
    {
        unreachable_ = other.unreachable_;
        return *this;
    }

    IsReachable&
    IsReachable::operator=(IsReachable&& other)
    {
        unreachable_ = std::move(other.unreachable_);
        return *this;
    }

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

    void
    IsReachable::combine(const IsReachable& other)
    {
        unreachable_.insert(other.unreachable_.begin(), other.unreachable_.end());
    }

    template <typename Archive>
    void
    IsReachable::serialize(Archive& archive)
    {
        archive(cereal::make_nvp("unreachable", unreachable_));
    }



    SplitTree::SplitTree(std::shared_ptr<const AddTree> addtree, SplitTree::DomainsT domains)
        : addtree_(addtree)
        , domtree_()
        , root_domains_(domains)
        , is_reachables_()
    {
        NodeId root_id = domtree_.root().id();
        is_reachables_.emplace(root_id, IsReachable());

        // For each root domain: update which nodes are accessible
        auto& is_reachable = is_reachables_.at(root_id);
        for (auto&& [feat_id, dom] : domains)
            update_is_reachable(is_reachable, feat_id, dom);
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
            FloatT sval = split.split_value;

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
            FloatT sval = split.split_value;

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

        is_reachables_.erase(node.id()); // might be old! -> mark_unreachable
        is_reachables_.emplace(node.left().id(), leaf.is_reachable_); // copy once
        is_reachables_.emplace(node.right().id(), std::move(leaf.is_reachable_)); // reuse for right

        FeatId feat_id = leaf.best_split_->feat_id;
        RealDomain dom_l, dom_r;
        std::tie(dom_l, dom_r) = RealDomain().split(leaf.best_split_->split_value);

        auto& is_reachable_l = is_reachables_.at(node.left().id());
        update_is_reachable(is_reachable_l, feat_id, dom_l);
        auto& is_reachable_r = is_reachables_.at(node.right().id());
        update_is_reachable(is_reachable_r, feat_id, dom_r);
    }

    void
    SplitTree::update_is_reachable(IsReachable& is_reachable,
                FeatId feat_id, RealDomain new_dom)
    {
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
            is_reachable.mark_unreachable(tree_index, node.id());
        if (node.is_leaf())
            return;

        LtSplit split = std::get<LtSplit>(node.get_split());
        FloatT sval = split.split_value;

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

    template<class Archive>
    void serialize(Archive& ar, RealDomain& m)
    {
        ar(cereal::make_nvp("lo", m.lo), cereal::make_nvp("hi", m.hi));
    }

    std::string
    SplitTree::to_json() const
    {
        std::stringstream ss;
        {
            cereal::JSONOutputArchive ar(ss);
            ar(cereal::make_nvp("domtree", domtree_),
               cereal::make_nvp("root_domains", root_domains_),
               cereal::make_nvp("is_reachables", is_reachables_));

        }
        return ss.str();
    }

    SplitTree
    SplitTree::from_json(
            std::shared_ptr<const AddTree> addtree,
            const std::string& json)
    {
        std::istringstream ss(json);

        DomTreeT domtree;
        DomainsT root_domains;
        ReachableT is_reachables;
        {
            cereal::JSONInputArchive ar(ss);
            ar(cereal::make_nvp("domtree", domtree),
               cereal::make_nvp("root_domains", root_domains),
               cereal::make_nvp("is_reachables", is_reachables));
        }

        SplitTree splittree(addtree, root_domains);
        std::swap(splittree.domtree_, domtree);
        std::swap(splittree.is_reachables_, is_reachables);

        return splittree;
    }




    /* --------------------------------------------------------------------- */

    SplitTreeLeaf::SplitTreeLeaf(const SplitTreeLeaf& other)
        : domtree_node_id_(other.domtree_node_id_)
        , is_reachable_(other.is_reachable_)
        , best_split_(other.best_split_)
        , split_score(other.split_score)
        , split_balance(other.split_balance)
    {}

    SplitTreeLeaf::SplitTreeLeaf(SplitTreeLeaf&& other)
        : domtree_node_id_(other.domtree_node_id_)
        , is_reachable_(std::move(other.is_reachable_))
        , best_split_(other.best_split_)
        , split_score(other.split_score)
        , split_balance(other.split_balance)
    {}

    SplitTreeLeaf::SplitTreeLeaf(NodeId domtree_node_id,
            const IsReachable& is_reachable)
        : domtree_node_id_(domtree_node_id)
        , is_reachable_(is_reachable)
        , best_split_()
        , split_score(0), split_balance(0)
    {}

    SplitTreeLeaf::SplitTreeLeaf(NodeId domtree_node_id,
            IsReachable&& is_reachable)
        : domtree_node_id_(domtree_node_id)
        , is_reachable_(std::move(is_reachable))
        , best_split_()
        , split_score(0), split_balance(0)
    {}

    SplitTreeLeaf&
    SplitTreeLeaf::operator=(const SplitTreeLeaf& other)
    {
        domtree_node_id_ = other.domtree_node_id_;
        is_reachable_ = other.is_reachable_;
        best_split_ = other.best_split_;
        split_score = other.split_score;
        split_balance = other.split_balance;
        return *this;
    }

    SplitTreeLeaf&
    SplitTreeLeaf::operator=(SplitTreeLeaf&& other)
    {
        domtree_node_id_ = other.domtree_node_id_;
        is_reachable_ = std::move(other.is_reachable_);
        best_split_ = std::move(other.best_split_);
        split_score = other.split_score;
        split_balance = other.split_balance;
        return *this;
    }

    NodeId
    SplitTreeLeaf::domtree_node_id() const
    {
        return domtree_node_id_;
    }

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
        size_t tree_index = 0;
        std::unordered_map<FeatId, std::unordered_set<FloatT>> duplicates;

        FeatId max_feat_id = -1;
        FloatT max_split_value = std::numeric_limits<FloatT>::quiet_NaN();
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
                if (node.is_leaf())
                    return ADD_NONE;

                int is_reachable_l = is_reachable_.is_reachable(tree_index, node.left().id());
                int is_reachable_r = is_reachable_.is_reachable(tree_index, node.right().id());

                if (!is_reachable_l && !is_reachable_r)
                    return ADD_NONE;
                if (!is_reachable_l)
                    return ADD_RIGHT;
                if (!is_reachable_r)
                    return ADD_LEFT;

                // only consider split if both left and right subtree are reachable

                LtSplit split = std::get<LtSplit>(node.get_split());
                FeatId feat_id = split.feat_id;
                FloatT sval = split.split_value;

                auto& feat_id_dups = duplicates[split.feat_id]; // auto-initialize set for feat_id
                auto p = feat_id_dups.find(split.split_value);
                if (p != feat_id_dups.end()) // already checked
                    return TreeVisitStatus::ADD_LEFT_AND_RIGHT;
                feat_id_dups.insert(split.split_value);

                // compute the number of unreachable nodes when we split the domain
                // on split.feat_id <> split.split_value

                RealDomain dom_l, dom_r;
                std::tie(dom_l, dom_r) = RealDomain().split(sval);

                int unreachable_l = count_unreachable_leafs(addtree, feat_id, dom_l);
                int unreachable_r = count_unreachable_leafs(addtree, feat_id, dom_r);
                int score = unreachable_l + unreachable_r;
                int balance = std::abs(unreachable_l - unreachable_r);

                //printf("tree_index=%lu feat_id=%d, split_value=%.10f, score=%d, balance=%d\n",
                //        tree_index, feat_id, sval, score, balance);

                if (score >= max_score)
                if (score > max_score || min_balance > balance)
                {
                    max_feat_id = feat_id;
                    max_split_value = sval;
                    max_score = score;
                    min_balance = balance;
                }

                return ADD_LEFT_AND_RIGHT;
            });
            ++tree_index;
        }

        printf("best split l%d: X%d <> %.10f with score %d, balance %d\n",
                domtree_node_id_, max_feat_id, max_split_value, max_score,
                min_balance);

        best_split_.emplace(max_feat_id, max_split_value);
        this->split_score = max_score;
        this->split_balance = min_balance;
    }

    LtSplit
    SplitTreeLeaf::get_best_split() const
    {
        if (!best_split_.has_value())
            throw std::runtime_error("call find_best_domtree_split first");
        return *best_split_;
    }

    std::tuple<FloatT, FloatT>
    SplitTreeLeaf::get_tree_bounds(const AddTree& at, size_t tree_index)
    {
        FloatT min =  std::numeric_limits<FloatT>::infinity();
        FloatT max = -std::numeric_limits<FloatT>::infinity();
        at[tree_index].dfs([this, tree_index, &min, &max](AddTree::TreeT::CRef node) {
            if (!is_reachable(tree_index, node.id()))
                return TreeVisitStatus::ADD_NONE;
            if (node.is_leaf())
            {
                FloatT leaf_value = node.leaf_value();
                min = std::min(min, leaf_value);
                max = std::max(max, leaf_value);
                return TreeVisitStatus::ADD_NONE;
            }
            return TreeVisitStatus::ADD_LEFT_AND_RIGHT;
        });
        return {min, max};
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

    SplitTreeLeaf
    SplitTreeLeaf::merge(const std::vector<SplitTreeLeaf>& leafs)
    {
        {
            auto it = leafs.cbegin();
            NodeId id = it->domtree_node_id_;
            ++it;
            for (; it != leafs.cend(); ++it)
                if (id != it->domtree_node_id_)
                    throw std::runtime_error("domtree_node_ids do not match");
        }

        auto it = leafs.begin();
        SplitTreeLeaf l(*it);
        ++it;
        for (; it != leafs.end(); ++it)
            l.is_reachable_.combine(it->is_reachable_);
        return l;
    }

    std::string
    SplitTreeLeaf::to_json() const
    {
        std::stringstream ss;
        {
            cereal::JSONOutputArchive ar(ss);
            ar(cereal::make_nvp("domtree_node_id", domtree_node_id_),
               cereal::make_nvp("is_reachable", is_reachable_),
               cereal::make_nvp("best_split", best_split_),
               cereal::make_nvp("split_score", split_score),
               cereal::make_nvp("split_balance", split_balance));
        }
        return ss.str();
    }

    SplitTreeLeaf
    SplitTreeLeaf::from_json(const std::string& json)
    {
        std::istringstream ss(json);

        NodeId domtree_node_id;
        IsReachable is_reachable;
        std::optional<LtSplit> best_split;
        int split_score, split_balance;
        {
            cereal::JSONInputArchive ar(ss);
            ar(cereal::make_nvp("domtree_node_id", domtree_node_id),
               cereal::make_nvp("is_reachable", is_reachable),
               cereal::make_nvp("best_split", best_split),
               cereal::make_nvp("split_score", split_score),
               cereal::make_nvp("split_balance", split_balance));
        }
        SplitTreeLeaf leaf(domtree_node_id, std::move(is_reachable));
        std::swap(leaf.best_split_, best_split);
        leaf.split_score = split_score;
        leaf.split_balance = split_balance;
        return leaf;
    }

} /* namespace treeck */
