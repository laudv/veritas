#include <algorithm>
#include <iostream>
#include <limits>
#include <stack>
#include <string>

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
        , root_domains_{}
        , domains_{}
    {
        splits_map_ = inner::extract_splits(*addtree_);

        FeatId max = 0;
        for (auto& n : splits_map_)
            max = std::max(max, n.first);
        num_features_ = static_cast<size_t>(max + 1);

        root_domains_.resize(num_features_);
    }

    SearchSpace::SearchSpace(
            std::shared_ptr<const AddTree> addtree,
            const Domains& root_domains)
        : SearchSpace(addtree)
    {
        std::copy(root_domains.begin(), root_domains.end(), root_domains_.begin());
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

    const Domains&
    SearchSpace::root_domains() const
    {
        return root_domains_;
    }

    void
    SearchSpace::get_domains(NodeId node_id, Domains& domains)
    {
        domains.resize(num_features_);
        std::copy(root_domains_.begin(), root_domains_.end(), domains.begin());
        Tree<LeafInfo>::MRef leaf = domtree_[node_id];
        Tree<LeafInfo>::MRef node = leaf;
        while (!node.is_root())
        {
            Tree<LeafInfo>::MRef child_node = node;
            node = node.parent();

            LtSplit split = std::get<LtSplit>(node.get_split());
            double sval = split.split_value;

            auto& dom = domains[split.feat_id];
            bool is_left = child_node.id() == node.left().id();
            if (is_left)
            {
                if (dom.hi > sval) dom.hi = sval;
            }
            else
            {
                if (dom.lo < sval) dom.lo = sval;
            }
        }

        std::cout << std::endl << "DOMAIN of node " << node_id << std::endl;
        for (size_t i = 0; i < domains.size(); ++i)
        {
            if (domains[i].is_everything()) continue;
            std::cout << " - [" << i << "] " << domains[i] << std::endl;
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

            node.split(dom_split);

            std::cout << "SPLITTING " << node_id << " " << dom_split << " into "
                << node.left().id() << " and " << node.right().id() << std::endl;

            // compute scores of left and right
            compute_best_score(node.left().id(), measure);
            compute_best_score(node.right().id(), measure);

            // push children onto the heap
            leafs_.push_back(node.left().id());  std::push_heap(leafs_.begin(), leafs_.end(), cmp);
            leafs_.push_back(node.right().id()); std::push_heap(leafs_.begin(), leafs_.end(), cmp);

            std::cout << domtree_ << std::endl;

            std::cout << std::endl << "leafs_:";
            for (auto x : leafs_) std::cout << " " << x << '(' << domtree_[x].leaf_value().score << ')';
            std::cout << std::endl;
        }
    }

    double
    UnreachableNodesMeasure::operator()(
            const SearchSpace& sp,
            const Domains& domains,
            LtSplit dom_split)
    {
        FeatId fid = dom_split.feat_id;
        RealDomain dom_left, dom_right;
        RealDomain dom_parent = domains[dom_split.feat_id];
        std::tie(dom_left, dom_right) = dom_parent.split(dom_split.split_value);

        int unreachable_left = count_unreachable_nodes(sp.addtree(), domains, fid, dom_left);
        int unreachable_right = count_unreachable_nodes(sp.addtree(), domains, fid, dom_right);

        double a = static_cast<double>(unreachable_left) + 1.0;
        double b = static_cast<double>(unreachable_right) + 1.0;
        //double score = (a * b) / (a + b);
        double score = a + b;

        std::cout << "MEASURE: " << dom_split
            << ": L " << unreachable_left
            << ", R " << unreachable_right
            << ", L+R " << (unreachable_left + unreachable_right)
            << " -> " << score << std::endl;

        return score;
    }

    int
    UnreachableNodesMeasure::count_unreachable_nodes(
            const AddTree& addtree,
            const Domains& domains,
            FeatId feat_id,
            RealDomain new_dom)
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
                //       [----)   |-------------)     |----)
                // ---------------------x-------------------------->
                //                 split_value
                //
                // if split.feat_id == feat_id: check new_dom (which is subdomain of dom)
                //                |--)              case 3.1
                //                |--------)        case 3.3
                //                     |--------)   case 3.3
                //                         |----)   case 3.2
                if (dom.lo < sval && dom.hi < sval) // case 1
                {
                    stack.push(node.left().id()); // only left matters, skip right
                }
                else if (dom.lo >= sval && dom.hi >= sval) // case 2
                {
                    stack.push(node.right().id()); // only right matters, skip right
                }
                else if (feat_id == split.feat_id) // case 3 and feat_id matches -> check new_dom and count!
                {
                    if (new_dom.lo < sval && new_dom.hi < sval) // case 3.1
                    {
                        stack.push(node.left().id());
                        unreachable += node.right().tree_size();
                    }
                    else if (new_dom.lo >= sval && new_dom.hi >= sval) // case 3.2
                    {
                        stack.push(node.right().id());
                        unreachable += node.left().tree_size();
                    }
                    else // case 3.3
                    {
                        stack.push(node.right().id());
                        stack.push(node.left().id());
                    }
                }
                else // case 3, but feat_id does not match, new_dom not applicable
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
