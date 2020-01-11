#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/memory.hpp>
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

                bool value = bnew_dom.is_true();
                if (BoolSplit().test(value))
                    marked_r = true; // r blocked because test goes left
                else
                    marked_l = true; // l blocked because test goes right
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

    template <typename Archive>
    void
    Nothing::serialize(Archive&) {}

    template <typename Archive>
    void
    DomTreeSplit::serialize(Archive& archive)
    {
        archive(cereal::make_nvp("instance_index", instance_index),
                cereal::make_nvp("split", split));
    }

    template <typename Archive>
    void
    DomTreeLeafInstance::serialize(Archive& archive)
    {
        archive(cereal::make_nvp("addtree", addtree),
                cereal::make_nvp("domains", domains),
                cereal::make_nvp("is_reachable", is_reachable));
    }

    std::ostream&
    operator<<(std::ostream& s, const Nothing& t)
    {
        return s << "Nothing";
    }

    std::ostream&
    operator<<(std::ostream& s, const DomTreeSplit& t)
    {
        return s << "DomTreeSplit(instance " << t.instance_index << ", " << t.split << ')';
    }


    TREECK_INSTANTIATE_TREE_TEMPLATE(DomTreeSplit, Nothing);





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

    std::shared_ptr<AddTree>
    DomTree::addtree(size_t instance) const
    {
        return instances_.at(instance).addtree;
    }

    void
    DomTree::add_instance(
            std::shared_ptr<AddTree> addtree,
            DomainsT&& domains)
    {
        if (tree_.root().is_internal())
            throw std::runtime_error("DomTree::add_instance: too late to add instances now");

        size_t instance_index = instances_.size();
        instances_.push_back({
            instance_index,
            addtree,
            std::move(domains),
            {} });

        NodeId root_id = tree_.root().id();
        DomTreeInstance& inst = instances_[instance_index];

        // update reachabilities
        auto& is_reachable = inst.is_reachables[0]; // create new IsReachable for root
        for (auto&& [feat_id, dom] : inst.root_domains)
            update_is_reachable(instance_index, root_id, feat_id, dom);
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
            DomTreeSplit split = node.get_split();

            // If this split does not affect the domain of instance i, continue
            if (split.instance_index != i) continue;

            refine_domains(domains, split.split, child_node.is_left_child());
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
                inst.addtree,
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
    DomTree::apply_leaf(DomTreeLeaf&& leaf)
    {
        if (leaf.num_instances() != num_instances())
            throw std::runtime_error("DomTree::apply_leaf: incompatible leaf");

        NodeId leaf_id = leaf.domtree_leaf_id();

        // update is_reachables for each instance
        for (size_t i = 0; i < num_instances(); ++i)
        {
            IsReachable& is_reachable0 = instances_.at(i).is_reachables.at(leaf_id);
            IsReachable& is_reachable1 = leaf.instances_.at(i).is_reachable;

            is_reachable0.combine(is_reachable1);
        }

        // split if leaf has best_split
        if (leaf.best_split_)
        {
            DomTreeSplit split = *leaf.best_split_;

            auto node = tree_[leaf_id];

            if (!node.is_leaf())
                throw std::runtime_error("DomTree::apply_leaf: split on non-leaf");

            node.split(*leaf.best_split_);

            // duplicate is_reachable for the new leafs
            for (size_t i = 0; i < num_instances(); ++i)
            {
                auto& is_reachables0 = instances_.at(i).is_reachables;
                DomTreeLeafInstance& inst1 = leaf.instances_.at(i);

                is_reachables0.erase(node.id()); // might be old! -> mark_unreachable
                is_reachables0.emplace(node.left().id(), inst1.is_reachable); // copy once
                is_reachables0.emplace(node.right().id(), std::move(inst1.is_reachable)); // reuse for right
            }

            Domain dom_l, dom_r;
            FeatId feat_id;
            visit_split(
                [&dom_l, &dom_r, &feat_id](const LtSplit& s) {
                    std::tie(dom_l, dom_r) = s.get_domains();
                    feat_id = s.feat_id;
                },
                [&dom_l, &dom_r, &feat_id](const BoolSplit& s) {
                    std::tie(dom_l, dom_r) = s.get_domains();
                    feat_id = s.feat_id;
                },
                split.split
            );

            // update is_reachable for instance (addtree) where the best split was found
            // the split only affects the values of that instance (we don't
            // know about the additional constraints here).
            auto& is_reachables0 = instances_.at(split.instance_index).is_reachables;
            auto& is_reachable_l = is_reachables0.at(node.left().id());
            update_is_reachable(split.instance_index, node.left().id(), feat_id, dom_l);
            auto& is_reachable_r = is_reachables0.at(node.right().id());
            update_is_reachable(split.instance_index, node.right().id(), feat_id, dom_r);
        }

    }

    void
    DomTree::update_is_reachable(size_t i, NodeId domtree_leaf_id,
            FeatId feat_id, Domain new_dom)
    {
        DomTreeInstance& inst = instances_.at(i);
        const AddTree& addtree = *inst.addtree;
        IsReachable& is_reachable = inst.is_reachables.at(domtree_leaf_id);

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

    //void
    //DomTreeLeaf::set_addtree(size_t instance, AddTree& addtree)
    //{
    //    instances_.at(instance).addtree = &addtree;
    //}

    NodeId
    DomTreeLeaf::domtree_leaf_id() const
    {
        return domtree_leaf_id_;
    }

    size_t
    DomTreeLeaf::num_instances() const
    {
        return instances_.size();
    }

    std::shared_ptr<AddTree>
    DomTreeLeaf::addtree(size_t instance) const
    {
        return instances_.at(instance).addtree;
    }

    std::optional<DomTreeSplit>
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
        Split max_split;
        int max_score = 0, min_balance = 0;
        size_t max_i = 0;
        for (size_t i = 0; i < num_instances(); ++i)
        {
            bool has_improved = find_best_split_for_instance(i, max_split,
                    max_score, min_balance);
            if (has_improved)
                max_i = i;
        }
        if (max_score > 0)
        {
            best_split_.emplace(DomTreeSplit { max_i, max_split });
            score = max_score;
            balance = min_balance;
        }
        else
        {
            throw std::runtime_error("no DomTree split found");
        }
    }

    bool
    DomTreeLeaf::find_best_split_for_instance(
            size_t i, Split& max_split,
            int& max_score, int& min_balance)
    {
        const AddTree& addtree = *instances_.at(i).addtree;
        const IsReachable& is_reachable = instances_.at(i).is_reachable;

        size_t tree_index = 0;
        bool has_improved = false;
        std::unordered_map<FeatId, std::unordered_set<FloatT>> duplicates;

        for (auto& tree : addtree.trees())
        {
            tree.dfs(
                    [this, &addtree, &is_reachable, &duplicates, tree_index,
                     &max_split, &max_score, &min_balance, &has_improved, &i]
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
                        //if (duplicates.find(split.feat_id) == duplicates.end())
                        //    throw std::runtime_error("assertion fail");

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
                //    << "instance=" << i
                //    << ", tree_index=" << tree_index
                //    << ", split=" << node.get_split()
                //    << ", score=" << score
                //    << ", (" << unreachable_l << ", " << unreachable_r << ")"
                //    << ", balance=" << balance
                //    << std::endl;

                if (score >= max_score)
                if (score > max_score || min_balance > balance)
                {
                    max_split = node.get_split();
                    max_score = score;
                    min_balance = balance;
                    has_improved = true;
                }

                return ADD_LEFT_AND_RIGHT;
            });
            ++tree_index;
        }

        //std::cout
        //    << "best split l" << domtree_leaf_id_
        //    << ", instance=" << i
        //    << ", split=" << max_split
        //    << ", score=" << max_score
        //    << ", balance=" << min_balance
        //    << std::endl;

        return has_improved;
    }

    int
    DomTreeLeaf::count_unreachable_leafs(size_t i, FeatId feat_id,
            Domain new_dom) const
    {
        const AddTree& at = *instances_.at(i).addtree;
        const IsReachable& is_reachable = instances_.at(i).is_reachable;

        int unreachable = 0;
        auto f = [&unreachable, &is_reachable](
                size_t tree_index,
                AddTree::TreeT::CRef node,
                bool marked) -> bool
        {
            if (!is_reachable.is_reachable(tree_index, node.id()))
                return false; // stop
            if (node.is_leaf())
            {
                if (marked)
                    ++unreachable;
                return false; // stop, it's a leaf
            }
            return true; // continue
        };

        size_t tree_index = 0;
        for (const AddTree::TreeT& tree : at.trees())
        {
            visit_addtree(at, is_reachable, tree_index, tree.root(), feat_id,
                    new_dom, false, f);
            ++tree_index;
        }

        return unreachable;
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

    DomTreeLeaf
    DomTreeLeaf::merge(const std::vector<DomTreeLeaf>& leafs)
    {
        if (leafs.size() == 0)
            throw std::runtime_error("DomTreeLeaf::merge: no leafs given");
        if (leafs.size() == 1)
            return leafs.at(0);

        {
            auto it = leafs.cbegin();
            NodeId id = it->domtree_leaf_id_;
            ++it;
            for (; it != leafs.cend(); ++it)
            {
                if (id != it->domtree_leaf_id_)
                    throw std::runtime_error("domtree_leaf_id do not match");
            }
        }

        auto it = leafs.begin();
        DomTreeLeaf l(*it);
        ++it;
        for (; it != leafs.end(); ++it)
        {
            for (size_t i = 0; i < l.num_instances(); ++i)
            {
                auto& instance0 = l.instances_.at(i);
                auto& instance1 = it->instances_.at(i);

                instance0.is_reachable.combine(instance1.is_reachable);
            }
        }
        return l;
    }

    template<class Archive>
    void serialize(Archive& ar, RealDomain& m)
    {
        ar(cereal::make_nvp("lo", m.lo), cereal::make_nvp("hi", m.hi));
    }

    template<class Archive>
    void serialize(Archive& ar, BoolDomain& m)
    {
        ar(m.value_);
    }

    void
    DomTreeLeaf::to_binary(std::ostream& os) const
    {
        cereal::BinaryOutputArchive ar(os);
        ar(cereal::make_nvp("id", domtree_leaf_id_),
           cereal::make_nvp("instances", instances_),
           cereal::make_nvp("best_split", best_split_),
           cereal::make_nvp("score", score),
           cereal::make_nvp("balance", balance));
    }

    DomTreeLeaf
    DomTreeLeaf::from_binary(std::istream& is)
    {
        NodeId id;
        std::vector<DomTreeLeafInstance> instances;
        std::optional<DomTreeSplit> best_split;
        int score, balance;
        {
            cereal::BinaryInputArchive ar(is);
            ar(cereal::make_nvp("id", id),
               cereal::make_nvp("instances", instances),
               cereal::make_nvp("best_split", best_split),
               cereal::make_nvp("score", score),
               cereal::make_nvp("balance", balance));
        }
        DomTreeLeaf leaf(id, std::move(instances));
        leaf.score = score;
        leaf.balance = balance;
        leaf.best_split_.swap(best_split);
        return leaf;
    }



} /* namespace treeck */
