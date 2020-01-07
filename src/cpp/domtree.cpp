#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/optional.hpp>

#include "util.h"
#include "tree.hpp"
#include "domtree.h"

namespace treeck {

    template <typename F>
    static void
    visit_addtree(
            const AddTree& addtree,
            const IsReachable& is_reachable,
            size_t tree_index,
            AddTree::TreeT::CRef node,
            FeatId feat_id,
            Domain new_dom,
            bool marked,
            const F& f) // (tree_index, node:CRef, marked:bool) -> false break, true continue
    {
        if (!f(tree_index, node, marked))
            return;

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

        visit_addtree(addtree, is_reachable, tree_index, node.left(), feat_id,
                new_dom, marked_l, f);
        visit_addtree(addtree, is_reachable, tree_index, node.right(), feat_id,
                new_dom, marked_r, f);
    }




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

    DomTree::DomTree()
        : tree_()
        , instances_() {}

    const DomTree::DomTreeT&
    DomTree::tree() const
    {
        return tree_;
    }

    size_t
    DomTree::num_instances() const
    {
        return instances_.size();
    }

    void
    DomTree::add_instance(
            std::shared_ptr<const AddTree> addtree,
            DomainsT&& domains,
            ReachableT&& reachables)
    {
        size_t instance_index = instances_.size();
        instances_.push_back({
            instance_index,
            addtree,
            std::move(domains),
            std::move(reachables) });
    }

    std::optional<Domain>
    DomTree::get_root_domain(size_t i, FeatId feat_id) const
    {
        auto& root_domains = instances_.at(i).root_domains;
        auto search = root_domains.find(feat_id);
        if (search != root_domains.end())
            return search->second;
        return {}; // unconstrained domain
    }

    DomainsT
    DomTree::get_domains(size_t i, NodeId domtree_leaf_id) const
    {
        DomainsT domains;
        const DomTreeInstance& inst = instances_.at(i);

        // copy root domains
        domains.insert(inst.root_domains.begin(), inst.root_domains.end());

        DomTreeT::CRef node = tree_[domtree_leaf_id];
        while (!node.is_root())
        {
            DomTreeT::CRef child_node = node;
            node = node.parent();

            visit_split(
                [&child_node, &domains](const LtSplit& s) {
                    RealDomain dom; // initially, is_everything == true

                    auto domptr = domains.find(s.feat_id);
                    if (domptr != domains.end())
                        dom = util::get_or<RealDomain>(domptr->second);

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
                [&child_node, &domains](const BoolSplit& s) {
                    BoolDomain dom; // initally both true and false

                    auto domptr = domains.find(s.feat_id);
                    if (domptr != domains.end()) // not in domains yet, check root domain
                        dom = util::get_or<BoolDomain>(domptr->second);

                    if (child_node.is_left_child())
                        dom = std::get<0>(dom.split());
                    else
                        dom = std::get<1>(dom.split());

                    domains[s.feat_id] = dom;
                },
                node.get_split());
        }

        return domains;
    }

    DomTreeLeaf
    DomTree::get_leaf(NodeId domtree_leaf_id) const
    {
        auto node = tree_[domtree_leaf_id];
        if (!node.is_leaf())
            throw std::runtime_error("DomTree::get_leaf on non-leaf node");

        std::vector<DomTreeLeafInstance> leaf_instances;
        for (auto& inst : instances_)
        {
            leaf_instances.push_back({
                nullptr,
                get_domains(inst.index, domtree_leaf_id),
                inst.is_reachables.at(domtree_leaf_id)
            });
        }

        return {
            domtree_leaf_id,
            std::move(leaf_instances),
        };
    }

    void
    DomTree::return_leaf(DomTreeLeaf&& leaf)
    {
        // split if leaf has best_split
        if (leaf.best_split_)
        {

        }
    }

    void
    DomTree::update_is_reachable(size_t i, NodeId domtree_node_id,
            FeatId feat_id, Domain new_dom)
    {
        DomTreeInstance& inst = instances_.at(i);
        const AddTree& addtree = *inst.addtree;
        IsReachable& is_reachable = inst.is_reachables.at(domtree_node_id);

        auto f = [&is_reachable]( // mark all marked nodes unreachable 
                size_t tree_index,
                AddTree::TreeT::CRef node,
                bool marked)
        {
            if (!is_reachable.is_reachable(tree_index, node.id()))
                return false; // already unreachble, don't bother going deeper
            if (marked)
            {
                is_reachable.mark_unreachable(tree_index, node.id());
                return false; // newly unreachable because marked, don't bother going deeper
            }
            if (node.is_leaf())
                return false; // leaf, end of tree, stop
            return true; // continue on!
        };

        size_t tree_index = 0;
        for (auto& tree : addtree.trees())
        {
            visit_addtree(addtree, is_reachable, tree_index, tree.root(),
                    feat_id, new_dom, false, f);
            ++tree_index;
        }
    }





    DomTreeLeaf::DomTreeLeaf(NodeId domtree_leaf_id,
            std::vector<DomTreeLeafInstance>&& instances)
        : domtree_leaf_id_(domtree_leaf_id)
        , instances_(std::move(instances))
        , best_split_{} {}

    NodeId
    DomTreeLeaf::domtree_node_id() const
    {
        return domtree_leaf_id_;
    }

    size_t
    DomTreeLeaf::num_instances() const
    {
        return instances_.size();
    }

    std::optional<BestSplit>
    DomTreeLeaf::get_best_split() const
    {
        return best_split_;
    }

    const DomainsT&
    DomTreeLeaf::get_domains(size_t i) const
    {
        return instances_.at(i).domains;
    }

    std::optional<Domain>
    DomTreeLeaf::get_domain(size_t i, FeatId feat_id) const
    {
        auto& domains = get_domains(i);
        auto search = domains.find(feat_id);
        if (search != domains.end())
            return search->second;
        return {};
    }

    size_t
    DomTreeLeaf::num_unreachable(size_t i) const
    {
        return instances_.at(i).is_reachable.num_unreachable();
    }

    bool
    DomTreeLeaf::is_reachable(size_t i, size_t tree_index, NodeId node_id) const
    {
        return instances_.at(i).is_reachable.is_reachable(tree_index, node_id);
    }

    void
    DomTreeLeaf::mark_unreachable(size_t i, size_t tree_index, NodeId node_id)
    {
        instances_.at(i).is_reachable.mark_unreachable(tree_index, node_id);
    }

    void
    DomTreeLeaf::find_best_split()
    {
        // TODO find best split for all instances
    }

    void
    DomTreeLeaf::find_best_split(size_t i, Split& max_split,
            int& max_score, int& min_balance)
    {
        const AddTree& addtree = this->addtree(i);
        const IsReachable& is_reachable = instances_.at(i).is_reachable;

        size_t tree_index = 0;
        std::unordered_map<FeatId, std::unordered_set<FloatT>> duplicates;

        for (auto& tree : addtree.trees())
        {
            tree.dfs(
                    [this, &addtree, &is_reachable, &duplicates, tree_index,
                     &max_split, &max_score, &min_balance, &i]
                    (AddTree::TreeT::CRef node) {
                if (node.is_leaf())
                    return ADD_NONE;

                int is_reachable_l = is_reachable.is_reachable(tree_index, node.left().id());
                int is_reachable_r = is_reachable.is_reachable(tree_index, node.right().id());

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
                int unreachable_l = count_unreachable_leafs(i, feat_id, dom_l);
                int unreachable_r = count_unreachable_leafs(i, feat_id, dom_r);
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
    }

    std::tuple<FloatT, FloatT>
    DomTreeLeaf::get_tree_bounds(size_t i, size_t tree_index)
    {
        const AddTree& at = *instances_.at(i).addtree;

        FloatT min =  std::numeric_limits<FloatT>::infinity();
        FloatT max = -std::numeric_limits<FloatT>::infinity();
        at[tree_index].dfs(
                [this, i, tree_index, &min, &max]
                (AddTree::TreeT::CRef node) {
            if (!is_reachable(i, tree_index, node.id()))
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

    const AddTree&
    DomTreeLeaf::addtree(size_t i) const
    {
        const AddTree *ptr = instances_.at(i).addtree;
        if (ptr)
            return *ptr;
        throw std::runtime_error("DomTreeLeaf: no addtree set, use set_addtree");
    }

    int
    DomTreeLeaf::count_unreachable_leafs(size_t i, FeatId feat_id,
            Domain new_dom) const
    {
        const AddTree& addtree = this->addtree(i);
        const IsReachable& is_reachable = instances_.at(i).is_reachable;

        int unreachable = 0;
        auto f = [&unreachable, &is_reachable](
                size_t tree_index,
                AddTree::TreeT::CRef node,
                bool marked) -> bool
        {
            if (!is_reachable.is_reachable(tree_index, node.id()))
                return false; // stop
            if (marked)
                unreachable++;
            return !node.is_leaf(); // stop if node is leaf
        };

        size_t tree_index = 0;
        for (const AddTree::TreeT& tree : addtree.trees())
        {
            visit_addtree(addtree, is_reachable, tree_index, tree.root(),
                    feat_id, new_dom, false, f);
            ++tree_index;
        }

        return unreachable;
    }

} /* namespace treeck */
