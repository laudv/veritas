#include <algorithm>
#include <string>
#include <stack>
#include <iostream>

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

    namespace inner {

        static
        void
        insert_split_value(SplitMap& splits, const LtSplit& split)
        {
            auto search = splits.find(split.feat_id);
            if (search != splits.end()) // found it!
                splits[split.feat_id].push_back(split.split_value);
            else
                splits.emplace(split.feat_id,  std::vector<double>{split.split_value});
        }

        static
        void
        visit_tree_nodes(AddTree::TreeT& tree, SplitMap& splits)
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
        SplitMap
        extract_splits(AddTree& at)
        {
            SplitMap splits;

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

                std::cout << "feat " << n.first << ": ";
                for (auto& x : v)
                    std::cout << x << ' ';
                std::cout << std::endl;
            }
            
            return splits;
        }

    } /* namespace internal */

    SearchSpace::SearchSpace(std::shared_ptr<AddTree> addtree)
        : num_features_(0)
        , addtree_(addtree)
        , domtree_{}
        , splits_{}
    {
        splits_ = inner::extract_splits(*addtree_);

        FeatId max = 0;
        for (auto& n : splits_)
            max = std::max(max, n.first);
        num_features_ = static_cast<size_t>(max);

        std::cout << "num_features_" << num_features_ << std::endl;
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
    
    template <typename Improvement, typename StopCond>
    void
    SearchSpace::split(Improvement measure, StopCond cond)
    {

    }



} /* namespace treeck */
