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

#include "subspaces.h"

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

    size_t
    IsReachable::num_unreachable() const
    {
        return unreachable_.size();
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



    Subspaces::Subspaces(std::shared_ptr<const AddTree> addtree, Subspaces::DomainsT domains)
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
    Subspaces::addtree() const { return *addtree_; }

    const Subspaces::DomTreeT&
    Subspaces::domtree() const { return domtree_; }

    std::optional<Domain>
    Subspaces::get_root_domain(FeatId feat_id) const
    {
        auto search = root_domains_.find(feat_id);
        if (search != root_domains_.end())
            return search->second;
        return {}; // unconstrained domain
    }

    void
    Subspaces::get_domains(NodeId domtree_leaf_id,
            Subspaces::DomainsT& domains) const
    {
        DomTreeT::CRef node = domtree_[domtree_leaf_id];
        while (!node.is_root())
        {
            DomTreeT::CRef child_node = node;
            node = node.parent();

            visit_split(
                [this, &child_node, &domains](const LtSplit& s) {
                    RealDomain dom; // initially, is_everything == true

                    auto domptr = domains.find(s.feat_id);
                    if (domptr == domains.end()) // not in domains yet, check root domain
                    {
                        auto root_dom_opt = get_root_domain(s.feat_id);
                        if (root_dom_opt)
                            dom = util::get_or<RealDomain>(*root_dom_opt, "fid=", s.feat_id);
                    }
                    else
                    {
                        dom = util::get_or<RealDomain>(domptr->second);
                    }

                    FloatT sval = s.split_value;
                    if (child_node.is_left_child())
                    {
                        if (dom.hi > sval) dom.hi = sval;
                    }
                    else
                    {
                        if (dom.lo < sval) dom.lo = sval;
                    }

                    domains[s.feat_id] = dom;
                },
                [this, &child_node, &domains](const BoolSplit& s) {
                    BoolDomain dom; // initally both true and false

                    auto domptr = domains.find(s.feat_id);
                    if (domptr == domains.end()) // not in domains yet, check root domain
                    {
                        auto root_dom_opt = get_root_domain(s.feat_id);
                        if (root_dom_opt)
                            dom = util::get_or<BoolDomain>(*root_dom_opt, "fid=", s.feat_id);
                    }
                    else
                    {
                        dom = util::get_or<BoolDomain>(domptr->second);
                    }

                    if (child_node.is_left_child())
                        dom = std::get<0>(dom.split());
                    else
                        dom = std::get<1>(dom.split());

                    domains[s.feat_id] = dom;
                },
                node.get_split());
        }
    }

    Subspace
    Subspaces::get_subspace(NodeId domtree_leaf_id)
    {
        auto node = domtree_[domtree_leaf_id];
        if (!node.is_leaf())
            throw std::runtime_error("Subspaces::get_leaf on non-leaf domtree node");

        // Subspace owns all its values so that we can easily transmit it
        // over the network to worker nodes. The structures should be
        // reasonably small.
        return Subspace(
            domtree_leaf_id,
            is_reachables_.at(domtree_leaf_id)
        );
    }

    void // shorthand method
    Subspaces::split_domtree_leaf(NodeId domtree_leaf_id)
    {
        split(get_subspace(domtree_leaf_id));
    }

    void
    Subspaces::split(Subspace&& leaf)
    {
        auto node = domtree_[leaf.domtree_node_id_];
        if (!node.is_leaf())
            throw std::runtime_error("Subspaces::split on non-leaf");
        if (is_reachables_.find(node.id()) == is_reachables_.end())
            throw std::runtime_error("Subspaces::split assertion error: no is_reachable for this node");
        if (!leaf.best_split_)
            leaf.find_best_domtree_split(*addtree_);

        node.split(*leaf.best_split_);

        is_reachables_.erase(node.id()); // might be old! -> mark_unreachable
        is_reachables_.emplace(node.left().id(), leaf.is_reachable_); // copy once
        is_reachables_.emplace(node.right().id(), std::move(leaf.is_reachable_)); // reuse for right

        Domain dom_l, dom_r;
        FeatId feat_id;
        visit_split(
            [&dom_l, &dom_r, &feat_id](const LtSplit& s) {
                std::tie(dom_l, dom_r) = RealDomain().split(s.split_value);
                feat_id = s.feat_id;
            },
            [&dom_l, &dom_r, &feat_id](const BoolSplit& s) {
                std::tie(dom_l, dom_r) = BoolDomain().split();
                feat_id = s.feat_id;
            },
            *leaf.best_split_
        );

        auto& is_reachable_l = is_reachables_.at(node.left().id());
        update_is_reachable(is_reachable_l, feat_id, dom_l);
        auto& is_reachable_r = is_reachables_.at(node.right().id());
        update_is_reachable(is_reachable_r, feat_id, dom_r);
    }

    void
    Subspaces::update_is_reachable(IsReachable& is_reachable,
                FeatId feat_id, Domain new_dom)
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
    Subspaces::update_is_reachable(IsReachable& is_reachable,
                size_t tree_index,
                AddTree::TreeT::CRef node,
                FeatId feat_id,
                Domain new_dom,
                bool marked)
    {
        if(!is_reachable.is_reachable(tree_index, node.id()))
            return;
        if (marked)
            is_reachable.mark_unreachable(tree_index, node.id());
        if (node.is_leaf())
            return;

        bool marked_l = marked;
        bool marked_r = marked;

        visit_split(
            [feat_id, new_dom, marked, &marked_l, &marked_r]
            (const LtSplit& s) {
                if (marked || s.feat_id != feat_id) return;

                RealDomain rnew_dom = util::get_or<RealDomain>(new_dom);
                if (rnew_dom.is_everything()) // TODO remove check if slow
                    throw std::runtime_error("stupid LtSplit");

                //       case 1       case 3          case 2
                //       [----)   |-------------)     |----)
                // ---------------------x-------------------------->
                //                 split_value
                //
                switch (rnew_dom.where_is_strict(s.split_value))
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
            },
            [feat_id, new_dom, marked, &marked_l, &marked_r]
            (const BoolSplit& s) {
                if (marked || s.feat_id != feat_id) return;

                auto bnew_dom = util::get_or<BoolDomain>(new_dom);
                if (bnew_dom.is_everything())
                    throw std::runtime_error("stupid BoolSplit");

                // false goes left, true goes right
                if (bnew_dom.is_true())
                    marked_l = true;
                if (bnew_dom.is_false())
                    marked_r = true;
            },
            node.get_split());

        update_is_reachable(is_reachable, tree_index, node.left(), feat_id, new_dom, marked_l);
        update_is_reachable(is_reachable, tree_index, node.right(), feat_id, new_dom, marked_r);
    }

    template<class Archive>
    void serialize(Archive& ar, RealDomain& m)
    {
        ar(cereal::make_nvp("lo", m.lo), cereal::make_nvp("hi", m.hi));
    }

    template<class Archive>
    void serialize(Archive& ar, BoolDomain& m)
    {
        ar(cereal::make_nvp("state", m.value_));
    }

    std::string
    Subspaces::to_json() const
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

    Subspaces
    Subspaces::from_json(
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

        Subspaces subspaces(addtree, root_domains);
        std::swap(subspaces.domtree_, domtree);
        std::swap(subspaces.is_reachables_, is_reachables);

        return subspaces;
    }




    /* --------------------------------------------------------------------- */

    Subspace::Subspace(const Subspace& other)
        : domtree_node_id_(other.domtree_node_id_)
        , is_reachable_(other.is_reachable_)
        , best_split_(other.best_split_)
        , split_score(other.split_score)
        , split_balance(other.split_balance)
    {}

    Subspace::Subspace(Subspace&& other)
        : domtree_node_id_(other.domtree_node_id_)
        , is_reachable_(std::move(other.is_reachable_))
        , best_split_(other.best_split_)
        , split_score(other.split_score)
        , split_balance(other.split_balance)
    {}

    Subspace::Subspace(NodeId domtree_node_id,
            const IsReachable& is_reachable)
        : domtree_node_id_(domtree_node_id)
        , is_reachable_(is_reachable)
        , best_split_()
        , split_score(0), split_balance(0)
    {}

    Subspace::Subspace(NodeId domtree_node_id,
            IsReachable&& is_reachable)
        : domtree_node_id_(domtree_node_id)
        , is_reachable_(std::move(is_reachable))
        , best_split_()
        , split_score(0), split_balance(0)
    {}

    Subspace&
    Subspace::operator=(const Subspace& other)
    {
        domtree_node_id_ = other.domtree_node_id_;
        is_reachable_ = other.is_reachable_;
        best_split_ = other.best_split_;
        split_score = other.split_score;
        split_balance = other.split_balance;
        return *this;
    }

    Subspace&
    Subspace::operator=(Subspace&& other)
    {
        domtree_node_id_ = other.domtree_node_id_;
        is_reachable_ = std::move(other.is_reachable_);
        best_split_ = std::move(other.best_split_);
        split_score = other.split_score;
        split_balance = other.split_balance;
        return *this;
    }

    NodeId
    Subspace::domtree_node_id() const
    {
        return domtree_node_id_;
    }

    size_t
    Subspace::num_unreachable() const
    {
        return is_reachable_.num_unreachable();
    }

    bool
    Subspace::is_reachable(size_t tree_index, NodeId node_id) const
    {
        return is_reachable_.is_reachable(tree_index, node_id);
    }

    void
    Subspace::mark_unreachable(size_t tree_index, NodeId node_id)
    {
        is_reachable_.mark_unreachable(tree_index, node_id);
    }

    void
    Subspace::find_best_domtree_split(const AddTree& addtree)
    {
        size_t tree_index = 0;
        std::unordered_map<FeatId, std::unordered_set<FloatT>> duplicates;

        Split max_split = LtSplit();
        int max_score = 0;
        int min_balance = -1;

        for (auto& tree : addtree.trees())
        {
            tree.dfs([this,
                      &addtree,
                      &duplicates,
                      tree_index,
                      &max_split,
                      &max_score,
                      &min_balance]
                    (AddTree::TreeT::CRef node) {
                if (node.is_leaf())
                    return ADD_NONE;

                int is_reachable_l = is_reachable_.is_reachable(tree_index, node.left().id());
                int is_reachable_r = is_reachable_.is_reachable(tree_index, node.right().id());

                // only consider split if both left and right subtree are reachable
                if (!is_reachable_l && !is_reachable_r)
                    return ADD_NONE;
                if (!is_reachable_l)
                    return ADD_RIGHT;
                if (!is_reachable_r)
                    return ADD_LEFT;

                Domain dom_l, dom_r;
                FeatId feat_id;

                bool skip = visit_split(
                    [&dom_l, &dom_r, &feat_id, &duplicates]
                    (const LtSplit& split) -> bool {
                        feat_id = split.feat_id;

                        auto& feat_id_dups = duplicates[split.feat_id]; // auto-initialize set for feat_id
                        auto p = feat_id_dups.find(split.split_value);
                        if (p != feat_id_dups.end())
                            return true; // already checked, skip!
                        feat_id_dups.insert(split.split_value);

                        std::tie(dom_l, dom_r) = RealDomain().split(split.split_value);
                        return false;
                    },
                    [&dom_l, &dom_r, &feat_id, &duplicates]
                    (const BoolSplit& split) -> bool {
                        feat_id = split.feat_id;

                        if (duplicates.find(split.feat_id) != duplicates.end())
                            return true; // already in duplicates, skip!
                        duplicates[split.feat_id]; // use duplicates as set for bool attributes

                        // TODO remove check
                        if (duplicates.find(split.feat_id) == duplicates.end())
                            throw std::runtime_error("assertion fail");

                        std::tie(dom_l, dom_r) = BoolDomain().split();

                        return false;
                    },
                    node.get_split());

                if (skip) // we've already checked this split
                    return ADD_LEFT_AND_RIGHT;

                // compute the number of unreachable nodes when we split the domain
                // on split.feat_id <> split.split_value
                int unreachable_l = count_unreachable_leafs(addtree, feat_id, dom_l);
                int unreachable_r = count_unreachable_leafs(addtree, feat_id, dom_r);
                int score = unreachable_l + unreachable_r;
                int balance = std::abs(unreachable_l - unreachable_r);

                //std::cout
                //    << "tree_index=" << tree_index
                //    << ", feat_id=" << feat_id
                //    << ", split=" << node.get_split()
                //    << ", score=" << score
                //    << ", balance=" << balance
                //    << std::endl;

                if (score >= max_score)
                if (score > max_score || min_balance > balance)
                {
                    max_split = node.get_split();
                    max_score = score;
                    min_balance = balance;
                }

                return ADD_LEFT_AND_RIGHT;
            });
            ++tree_index;
        }

        //std::cout
        //    << "best split l" << domtree_node_id_
        //    << ", split=" << max_split
        //    << ", score=" << max_score
        //    << ", balance=" << min_balance
        //    << std::endl;

        best_split_.emplace(max_split);
        this->split_score = max_score;
        this->split_balance = min_balance;
    }


    bool
    Subspace::has_best_split() const
    {
        return best_split_.has_value();
    }

    Split
    Subspace::get_best_split() const
    {
        if (!best_split_.has_value())
            throw std::runtime_error("call find_best_domtree_split first");
        return *best_split_;
    }

    std::tuple<FloatT, FloatT>
    Subspace::get_tree_bounds(const AddTree& at, size_t tree_index)
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
    Subspace::count_unreachable_leafs(
            const AddTree& addtree,
            FeatId feat_id,
            Domain new_dom) const
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
    Subspace::count_unreachable_leafs(
            const AddTree& addtree,
            size_t tree_index,
            AddTree::TreeT::CRef node,
            FeatId feat_id,
            Domain new_dom,
            bool marked) const
    {
        if (node.is_leaf())
            return marked ? 1 : 0;
        if (!is_reachable(tree_index, node.id()))
            return 0;

        bool marked_l = marked; // remain marked if already marked
        bool marked_r = marked;

        visit_split(
            [feat_id, new_dom, marked, &marked_l, &marked_r]
            (const LtSplit& split) {
                if (marked || split.feat_id != feat_id) return;

                RealDomain rnew_dom = util::get_or<RealDomain>(new_dom,
                        " for feat_id ", feat_id, " (count_unreachable_leafs)");

                if (rnew_dom.is_everything()) // TODO remove check if slow
                    throw std::runtime_error("stupid LtSplit");

                //       case 1       case 3          case 2
                //       [----)   |-------------)     |----)
                // ---------------------x-------------------------->
                //                 split_value
                //
                switch (rnew_dom.where_is_strict(split.split_value))
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
            },
            [feat_id, new_dom, marked, &marked_l, &marked_r]
            (const BoolSplit& split) {
                if (marked || split.feat_id != feat_id) return;

                BoolDomain bnew_dom = util::get_or<BoolDomain>(new_dom,
                        " for feat_id ", feat_id, " (count_unreachable_leafs)");

                if (bnew_dom.is_everything())
                    throw std::runtime_error("stupid BoolSplit");

                // false goes left, true goes right
                if (bnew_dom.is_true())
                    marked_l = true;
                if (bnew_dom.is_false())
                    marked_r = true;
            },
            node.get_split());

        return //((marked_l || marked_r) ? 1 : 0) + // this node's split became deterministic..
            count_unreachable_leafs(addtree, tree_index, node.left(), // ..or just count leafs
                    feat_id, new_dom, marked_l) +
            count_unreachable_leafs(addtree, tree_index, node.right(),
                    feat_id, new_dom, marked_r);
    }

    Subspace
    Subspace::merge(const std::vector<Subspace>& leafs)
    {
        if (leafs.size() == 0)
            throw std::runtime_error("Subspace merge: empty leafs");
        if (leafs.size() == 1)
            return leafs.at(0);

        {
            auto it = leafs.cbegin();
            NodeId id = it->domtree_node_id_;
            ++it;
            for (; it != leafs.cend(); ++it)
                if (id != it->domtree_node_id_)
                    throw std::runtime_error("domtree_node_ids do not match");
        }

        auto it = leafs.begin();
        Subspace l(*it);
        ++it;
        for (; it != leafs.end(); ++it)
            l.is_reachable_.combine(it->is_reachable_);
        return l;
    }

    std::string
    Subspace::to_json() const
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

    Subspace
    Subspace::from_json(const std::string& json)
    {
        std::istringstream ss(json);

        NodeId domtree_node_id;
        IsReachable is_reachable;
        std::optional<Split> best_split;
        int split_score, split_balance;
        {
            cereal::JSONInputArchive ar(ss);
            ar(cereal::make_nvp("domtree_node_id", domtree_node_id),
               cereal::make_nvp("is_reachable", is_reachable),
               cereal::make_nvp("best_split", best_split),
               cereal::make_nvp("split_score", split_score),
               cereal::make_nvp("split_balance", split_balance));
        }
        Subspace leaf(domtree_node_id, std::move(is_reachable));
        std::swap(leaf.best_split_, best_split);
        leaf.split_score = split_score;
        leaf.split_balance = split_balance;
        return leaf;
    }

} /* namespace treeck */
