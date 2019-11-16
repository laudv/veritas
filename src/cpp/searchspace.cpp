#include <algorithm>
#include <iostream>
#include <limits>
#include <stack>
#include <string>

#include "util.h"
#include "tree.hpp"
#include "searchspace.h"

namespace treeck {

    template <typename Archive>
    void
    LeafInfo::serialize(Archive& archive)
    {
        std::string x("TODO leafInfo");
        archive(x);
    }

    std::ostream&
    operator<<(std::ostream& s, LeafInfo)
    {
        return s << "TODO leafInfo";
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
        visit_tree_nodes(AddTree::TreeT& tree, SearchSpace::SplitMap& splits)
        {
            std::stack<NodeRef<double>> stack;
            stack.push(tree.root());

            while (!stack.empty())
            {
                NodeRef n = stack.top();
                stack.pop();

                if (n.is_leaf())
                    continue;

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
        extract_splits(AddTree& at)
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

    SearchSpace::SearchSpace(std::shared_ptr<AddTree> addtree)
        : num_features_(0)
        , addtree_(addtree)
        , domtree_{}
        , splits_map_{}
        , leafs_{}
    {
        splits_map_ = inner::extract_splits(*addtree_);

        FeatId max = 0;
        for (auto& n : splits_map_)
            max = std::max(max, n.first);
        num_features_ = static_cast<size_t>(max);
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
    SearchSpace::get_domains(NodeId node_id, std::vector<RealDomain>& domains)
    {
        domains.resize(num_features_);
        std::fill(domains.begin(), domains.end(), RealDomain());
        NodeRef<LeafInfo> leaf = domtree_[node_id];

        NodeRef<LeafInfo> node = leaf, prev_node = leaf;
        while (!node.is_root())
        {
            node = node.parent();

            Split split = node.get_split();
            std::visit(util::overloaded {
                [&domains, &prev_node, &node](LtSplit& x) {
                    bool is_left = prev_node.id() == node.left().id();
                    RealDomain l, r;
                    std::tie(l, r) = domains[x.feat_id].split(x.split_value);
                    domains[x.feat_id] = is_left ? l : r;
                    std::cout << "DOMAIN[" << x.feat_id << "] = " << domains[x.feat_id] << std::endl;
                },
                [](EqSplit&) { throw std::runtime_error("EqSplit not supported"); },
                [](auto& x) { static_assert(util::always_false<decltype(x)>::value, "non-exhaustive visit"); }
            }, split);

            prev_node = node;
        }
    }
    
    void
    SearchSpace::split(MeasureF measure, StopCondF cond)
    {
        if (domtree_.num_nodes() > 1) { throw std::runtime_error("already split"); }

        leafs_.push_back(domtree_.root().id());
        std::vector<RealDomain> domains(num_features_);

        while (!cond(*this))
        {
            size_t max_leafs_index = 0;
            FeatId max_feat_id = -1;
            double max_split_value = std::numeric_limits<double>::quiet_NaN();
            double max_score = -std::numeric_limits<double>::infinity();

            for (size_t leafs_index = 0; leafs_index < leafs_.size(); ++leafs_index)
            {
                get_domains(leafs_[leafs_index], domains);

                std::cout << "LEAF " << leafs_[leafs_index] << std::endl;
                for (auto dom : domains)
                    std::cout << "  - " << dom << std::endl;
                std::cout << std::endl;

                for (auto&& [feat_id, splits] : splits_map_)
                for (double split_value : splits)
                {
                    // check if split in domain
                    RealDomain dom = domains[feat_id];
                    if (dom.lo == split_value || !dom.contains(split_value)) continue;

                    std::cout << "    feat_id=" << feat_id << ", " << "split_value=" << split_value << std::endl;

                    // compute score for left / right
                    double score = 0.0; // TODO compute score

                    // replace details about max if better
                    if (score > max_score)
                    {
                        max_leafs_index = leafs_index;
                        max_feat_id = feat_id;
                        max_split_value = split_value;
                        max_score = score;
                        std::cout << "MAX " << max_feat_id << ", " << max_split_value << std::endl;
                    }
                }
            }

            NodeId max_leaf = leafs_[max_leafs_index];
            domtree_[max_leaf].split(LtSplit(max_feat_id, max_split_value));
            leafs_[max_leafs_index] = domtree_[max_leaf].left().id();
            leafs_.push_back(domtree_[max_leaf].right().id());

            std::cout << domtree_ << std::endl;
            std::cout << "LEAFS=";
            for (auto leaf : leafs_)
                std::cout << leaf << " ";
            std::cout << std::endl;
        }

        std::cout << "stopping condition" << std::endl;
    }

    double
    NumDisabledNodesMeasure::operator()(const SearchSpace& sp)
    {
        return 1.0;
    }

    bool
    SizeOfDomTreeStopCond::operator()(const SearchSpace& sp)
    {
        return sp.domtree().num_nodes() > max_num_of_nodes;
    }

} /* namespace treeck */
