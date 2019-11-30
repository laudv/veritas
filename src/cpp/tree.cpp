#include "util.h"
#include "tree.hpp"

namespace treeck {

    SplitBase::SplitBase(FeatId feat_id) : feat_id(feat_id) {}

    LtSplit::LtSplit() : LtSplit(-1, 0.0) {}
    LtSplit::LtSplit(FeatId feat_id, LtSplit::ValueT split_value)
        : SplitBase(feat_id)
        , split_value(split_value) {}

    bool
    LtSplit::test(LtSplit::ValueT value) const
    {
        return value < this->split_value;
    }

    EqSplit::EqSplit() : EqSplit(-1, 0) {}
    EqSplit::EqSplit(FeatId feat_id, EqSplit::ValueT category)
        : SplitBase(feat_id)
        , category(category) {}

    bool
    EqSplit::test(EqSplit::ValueT value) const
    {
        return value == this->category;
    }

    std::ostream&
    operator<<(std::ostream& s, const Split& split)
    {
        std::visit(util::overloaded {
            [&s](const LtSplit& x) { s << "LtSplit(" << x.feat_id << ", " << x.split_value << ')'; },
            [&s](const EqSplit& x) { s << "EqSplit(" << x.feat_id << ", " << x.category << ')'; },
            [](auto& x) { static_assert(util::always_false<decltype(x)>::value, "non-exhaustive visit"); }
        }, split);
        return s;
    }



    TREECK_INSTANTIATE_TREE_TEMPLATE(double);

    AddTree::AddTree() : trees_{}, base_score(0.0)
    {
        trees_.reserve(16);
    }

    size_t
    AddTree::add_tree(AddTree::TreeT&& tree)
    {
        size_t index = trees_.size();
        trees_.push_back(std::forward<AddTree::TreeT>(tree));
        return index;
    }

    size_t
    AddTree::size() const
    {
        return trees_.size();
    }

    size_t
    AddTree::num_nodes() const
    {
        size_t c = 0;
        for (auto& tree : trees_)
            c += tree.num_nodes();
        return c;
    }

    AddTree::TreeT&
    AddTree::operator[](size_t index)
    {
        return trees_[index];
    }

    const AddTree::TreeT&
    AddTree::operator[](size_t index) const
    {
        return trees_[index];
    }

    const std::vector<AddTree::TreeT>&
    AddTree::trees() const
    {
        return trees_;
    }

    std::string
    AddTree::to_json()
    {
        std::stringstream ss;
        {
            cereal::JSONOutputArchive ar(ss);
            ar(cereal::make_nvp("base_score", base_score),
               cereal::make_nvp("trees", trees_));
        }
        return ss.str();
    }

    AddTree
    AddTree::from_json(const std::string& json)
    {
        std::istringstream ss(json);
        AddTree addtree;
        {
            cereal::JSONInputArchive ar(ss);
            ar(cereal::make_nvp("base_score", addtree.base_score),
               cereal::make_nvp("trees", addtree.trees_));
        }
        return addtree;
    }

    AddTree
    AddTree::from_json_file(const char *file)
    {
        std::ifstream t(file);
        std::stringstream buffer;
        buffer << t.rdbuf();
        std::string s(buffer.str());

        return AddTree::from_json(s);
    }

    namespace inner {
        static
        void
        insert_split_value(AddTree::SplitMapT& splits, const LtSplit& split)
        {
            auto search = splits.find(split.feat_id);
            if (search != splits.end()) // found it!
                splits[split.feat_id].push_back(split.split_value);
            else
                splits.emplace(split.feat_id,  std::vector<double>{split.split_value});
        }

        static
        void
        visit_tree_nodes(const AddTree::TreeT& tree, AddTree::SplitMapT& splits)
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
    } /* namespace inner */

    AddTree::SplitMapT
    AddTree::get_splits() const
    {
        std::unordered_map<FeatId, std::vector<double>> splits;

        // collect all the split values
        for (const TreeT& tree : trees_)
        {
            inner::visit_tree_nodes(tree, splits);
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

    std::tuple<std::vector<size_t>, std::vector<NodeId>, std::vector<FeatId>, std::vector<double>>
    AddTree::export_lists() const
    {
        std::vector<size_t> offsets;
        std::vector<NodeId> lefts;
        std::vector<FeatId> feat_ids;
        std::vector<double> values;

        for (auto& tree : trees_)
        {
            size_t offset = lefts.size();
            size_t num_nodes = tree.num_nodes();

            offsets.push_back(offset);
            lefts.resize(offset + num_nodes);
            feat_ids.resize(offset + num_nodes);
            values.resize(offset + num_nodes);

            NodeId *ls = &lefts[offset];
            FeatId *fs = &feat_ids[offset];
            double *vs = &values[offset];

            tree.dfs([ls, fs, vs](TreeT::CRef node) {
                if (node.is_leaf())
                {
                    ls[node.id()] = -1;
                    fs[node.id()] = -1;
                    vs[node.id()] = node.leaf_value();

                    return TreeVisitStatus::ADD_NONE;
                }
                else
                {
                    LtSplit split = std::get<LtSplit>(node.get_split());

                    ls[node.id()] = node.left().id();
                    fs[node.id()] = split.feat_id;
                    vs[node.id()] = split.split_value;

                    return TreeVisitStatus::ADD_LEFT_AND_RIGHT;
                }
            });
        }

        return { offsets, lefts, feat_ids, values };
    }

    std::ostream&
    operator<<(std::ostream& s, const AddTree& at)
    {
        int counter = 0;
        s << "AddTree with " << at.size() << " trees and base_score " << at.base_score << std::endl;
        for (const AddTree::TreeT& tree : at.trees())
            s  << ++counter << ". " << std::endl << tree;
        return s << "--------" << std::endl;
    }

} /* namespace treeck */
