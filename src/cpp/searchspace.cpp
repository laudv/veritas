#include <algorithm>
#include <iostream>
#include <limits>
#include <stack>
#include <string>

#include "addtree.h"
#include "domain.h"
#include "util.h"
#include "tree.hpp"
#include "searchspace.h"

namespace treeck {

    LeafInfo::LeafInfo()
        : LeafInfo(LtSplit(), std::numeric_limits<double>::quiet_NaN()) {}

    LeafInfo::LeafInfo(Split split, double score)
        : dom_split(split)
        , score(score) {}

    template <typename Archive>
    void
    LeafInfo::serialize(Archive& archive)
    {
        archive(CEREAL_NVP(dom_split), CEREAL_NVP(score));
    }

    std::ostream&
    operator<<(std::ostream& s, LeafInfo inf)
    {
        return s << "LeafInfo(" << inf.dom_split << ", " << inf.score << ')';
    }

    namespace inner {

        static
        void
        insert_split_value(SearchSpace::SplitMap& splits, const LtSplit& split)
        {
            auto search = splits.find(split.feat_id);
            if (search != splits.end()) // found it!
                splits[split.feat_id].push_back(split.split_value);
            else
                splits.emplace(split.feat_id,  std::vector<double>{split.split_value});
        }

        static
        void
        visit_tree_nodes(const AddTree::TreeT& tree, SearchSpace::SplitMap& splits)
        {
            std::stack<AddTree::TreeT::CRef> stack;
            stack.push(tree.root());

            while (!stack.empty())
            {
                auto n = stack.top();
                stack.pop();

                if (n.is_leaf()) continue;

                Split split = n.get_split();
                std::visit(util::overloaded {
                    [&splits](LtSplit& x) { insert_split_value(splits, x);  },
                    [](EqSplit&) { throw std::runtime_error("EqSplit not supported"); },
                    [](auto& x) { static_assert(util::always_false<decltype(x)>::value, "non-exhaustive visit"); }
                }, split);

                stack.push(n.right());
                stack.push(n.left());
            }
        }

        static
        SearchSpace::SplitMap
        extract_splits(const AddTree& at)
        {
            SearchSpace::SplitMap splits;

            // collect all the split values
            for (size_t i = 0; i < at.size(); ++i)
            {
                visit_tree_nodes(at[i], splits);
            }

            // sort the split values, remove duplicates
            for (auto& n : splits)
            {
                std::vector<double>& v = n.second;
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
            }
            
            return splits;
        }

    } /* namespace inner */

    SearchSpace::SearchSpace(std::shared_ptr<const AddTree> addtree)
        : num_features_(0)
        , addtree_(addtree)
        , domtree_{}
        , splits_map_{}
        , leafs_{}
        , domains_{}
    {
        splits_map_ = inner::extract_splits(*addtree_);

        FeatId max = 0;
        for (auto& n : splits_map_)
            max = std::max(max, n.first);
        num_features_ = static_cast<size_t>(max + 1);
    }

    void
    SearchSpace::compute_best_score(NodeId domtree_leaf_id, MeasureF measure)
    {
        FeatId max_feat_id = -1;
        double max_split_value = std::numeric_limits<double>::quiet_NaN();
        double max_score = -std::numeric_limits<double>::infinity();

        get_domains(domtree_leaf_id, domains_);

        for (auto&& [feat_id, splits] : splits_map_)
        for (double split_value : splits)
        {
            // check if split in domain
            RealDomain dom = domains_[feat_id];
            if (dom.lo == split_value || !dom.contains(split_value)) continue;

            // compute score for left / right
            double score = measure(*this, domains_, LtSplit(feat_id, split_value));

            // replace details about max if better
            if (score > max_score)
            {
                //std::cout << "MAX F" << feat_id << "<" << split_value << " : " << score << " > " << max_score << std::endl;

                max_feat_id = feat_id;
                max_split_value = split_value;
                max_score = score;
            }
        }

        LeafInfo inf(LtSplit(max_feat_id, max_split_value), max_score);
        domtree_[domtree_leaf_id].set_leaf_value(inf);
    }

    size_t
    SearchSpace::num_features() const
    {
        return num_features_;
    }

    const AddTree&
    SearchSpace::addtree() const
    {
        return *addtree_;
    }

    const SearchSpace::TreeT&
    SearchSpace::domtree() const
    {
        return domtree_;
    }

    const std::vector<NodeId>&
    SearchSpace::leafs() const
    {
        return leafs_;
    }

    void
    SearchSpace::get_domains(NodeId node_id, Domains& domains)
    {
        domains.resize(num_features_);
        std::fill(domains.begin(), domains.end(), RealDomain());
        Tree<LeafInfo>::MRef leaf = domtree_[node_id];
        Tree<LeafInfo>::MRef node = leaf, prev_node = leaf;
        while (!node.is_root())
        {
            node = node.parent();

            LtSplit split = std::get<LtSplit>(node.get_split());
            double sval = split.split_value;

            auto old_dom = domains[split.feat_id];
            auto& dom = domains[split.feat_id];
            bool is_left = prev_node.id() == node.left().id();
            if (is_left && dom.hi > sval) dom.hi = sval;
            else if (dom.lo < sval)       dom.lo = sval;

            std::cout << "DOMAIN[" << split.feat_id << "]: " << old_dom << " => " << domains[split.feat_id] << std::endl;
            prev_node = node;
        }
    }
    
    void
    SearchSpace::split(MeasureF measure, StopCondF cond)
    {
        if (!domtree_.root().is_leaf()) { throw std::runtime_error("already split"); }

        leafs_.push_back(domtree_.root().id());
        compute_best_score(domtree_.root().id(), measure);

        auto cmp = [this](const NodeId& a, const NodeId& b) {
            return this->domtree_[a].leaf_value().score < this->domtree_[b].leaf_value().score;
        };

        while (!cond(*this)) // split until stop condition is met
        {
            // pop the best leaf node of the domain tree so that we can split it
            std::pop_heap(leafs_.begin(), leafs_.end(), cmp);
            NodeId node_id = leafs_.back(); leafs_.pop_back();
            auto node = domtree_[node_id];
            Split dom_split = node.leaf_value().dom_split;

            std::cout << "SPLITTING " << node_id << " " << dom_split << std::endl;

            node.split(dom_split);

            // compute scores of left and right
            compute_best_score(node.left().id(), measure);
            compute_best_score(node.right().id(), measure);

            // push children onto the heap
            leafs_.push_back(node.left().id());  std::push_heap(leafs_.begin(), leafs_.end(), cmp);
            leafs_.push_back(node.right().id()); std::push_heap(leafs_.begin(), leafs_.end(), cmp);

            std::cout << domtree_ << std::endl;

            std::cout << "leafs_:";
            for (auto x : leafs_) std::cout << " " << x << '(' << domtree_[x].leaf_value().score << ')';
            std::cout << std::endl;

            //for (size_t leafs_index = 0; leafs_index < leafs_.size(); ++leafs_index)
            //{
            //    get_domains(leafs_[leafs_index], domains);

            //    std::cout << "LEAF " << leafs_[leafs_index] << std::endl;
            //    for (auto dom : domains)
            //        std::cout << "  - " << dom << std::endl;
            //    std::cout << std::endl;

            //    for (auto&& [feat_id, splits] : splits_map_)
            //    for (double split_value : splits)
            //    {
            //        // check if split in domain
            //        RealDomain dom = domains[feat_id];
            //        if (dom.lo == split_value || !dom.contains(split_value)) continue;

            //        // compute score for left / right
            //        double score = measure(*this, domains, LtSplit(feat_id, split_value));

            //        // replace details about max if better
            //        if (score > max_score)
            //        {
            //            std::cout << "MAX F" << feat_id << "<" << split_value
            //                << " : " << score << " > " << max_score << std::endl;

            //            max_leafs_index = leafs_index;
            //            max_feat_id = feat_id;
            //            max_split_value = split_value;
            //            max_score = score;
            //        }
            //    }
            //}

            //NodeId max_leaf = leafs_[max_leafs_index];
            //std::cout << "max_leaf_id=" << max_leaf << " max_feat_id=" << max_feat_id << " max_split_value=" << max_split_value << std::endl;
            //domtree_[max_leaf].split(LtSplit(max_feat_id, max_split_value));
            //leafs_[max_leafs_index] = domtree_[max_leaf].left().id();
            //leafs_.push_back(domtree_[max_leaf].right().id());

            //std::cout << domtree_ << std::endl;
            //std::cout << "LEAFS=";
            //for (auto leaf : leafs_)
            //    std::cout << leaf << " ";
            //std::cout << std::endl;
        }

        std::cout << "stopping condition" << std::endl;
    }

    double
    UnreachableNodesMeasure::operator()(
            const SearchSpace& sp,
            SearchSpace::Domains& domains,
            LtSplit dom_split)
    {
        RealDomain dom_left, dom_right;
        RealDomain dom_parent = domains[dom_split.feat_id];
        std::tie(dom_left, dom_right) = dom_parent.split(dom_split.split_value);

        domains[dom_split.feat_id] = dom_left;
        int unreachable_left = count_unreachable_nodes(sp.addtree(), domains);
        domains[dom_split.feat_id] = dom_right;
        int unreachable_right = count_unreachable_nodes(sp.addtree(), domains);
        domains[dom_split.feat_id] = dom_parent; // ensure we undo domain changes

        double a = static_cast<double>(unreachable_left);
        double b = static_cast<double>(unreachable_right);
        double score = (a * b) / (a + b);

        //std::cout << "MEASURE: " << dom_split << ": L " << unreachable_left << ", R " << unreachable_right << " -> " << score << std::endl;

        return score;
    }

    int
    UnreachableNodesMeasure::count_unreachable_nodes(
            const AddTree& addtree,
            const SearchSpace::Domains& domains)
    {
        int unreachable = 0;
        for (const AddTree::TreeT& tree : addtree.trees())
        {
            stack.push(tree.root().id());
            while (!stack.empty())
            {
                NodeId node_id = stack.top(); stack.pop();
                auto node = tree[node_id];

                if (node.is_leaf()) continue;

                LtSplit split = std::get<LtSplit>(node.get_split());
                RealDomain dom = domains[split.feat_id];
                double sval = split.split_value;

                //       case 1       case 3          case 2
                //       [----)       |----)          |----)
                // ---------------------x-------------------------->
                //                 split_value
                if (dom.lo < sval && dom.hi < sval) // case 1
                {
                    stack.push(node.left().id()); // only left matters, skip right
                    unreachable += node.right().tree_size();
                }
                else if (dom.lo >= sval && dom.hi >= sval) // case 2
                {
                    stack.push(node.right().id()); // only right matters, skip right
                    unreachable += node.left().tree_size();
                }
                else // case 1
                {
                    stack.push(node.right().id());
                    stack.push(node.left().id());
                }
            }

        }
        return unreachable;
    }

    bool
    NumDomTreeLeafsStopCond::operator()(const SearchSpace& sp)
    {
        return sp.leafs().size() > max_num_leafs;
    }

} /* namespace treeck */
