/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "tree.hpp"

namespace veritas {

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

    std::tuple<LtSplit::DomainT, LtSplit::DomainT>
    LtSplit::get_domains() const
    {
        return RealDomain().split(split_value);
    }

    bool operator==(const LtSplit& a, const LtSplit& b)
    {
        return a.feat_id == b.feat_id && a.split_value == b.split_value;
    }

    //EqSplit::EqSplit() : EqSplit(-1, 0) {}
    //EqSplit::EqSplit(FeatId feat_id, EqSplit::ValueT category)
    //    : SplitBase(feat_id)
    //    , category(category) {}

    //bool
    //EqSplit::test(EqSplit::ValueT value) const
    //{
    //    return value == this->category;
    //}

    //bool operator==(const EqSplit& a, const EqSplit& b)
    //{
    //    return a.feat_id == b.feat_id && a.category == b.category;
    //}

    BoolSplit::BoolSplit() : SplitBase(-1) {}
    BoolSplit::BoolSplit(FeatId feat_id) : SplitBase(feat_id) {}

    bool
    BoolSplit::test(BoolSplit::ValueT value) const
    {
        return value; // true left, false right
    }

    std::tuple<BoolSplit::DomainT, BoolSplit::DomainT>
    BoolSplit::get_domains() const
    {
        return BoolDomain().split();
    }

    void
    refine_domains(DomainsT& domains, const Split& split, bool is_left_child)
    {
        visit_split(
            [&domains, is_left_child](const LtSplit& s) {
                refine_domains(domains, s, is_left_child);
            },
            [&domains, is_left_child](const BoolSplit& s) {
                refine_domains(domains, s, is_left_child);
            },
            split);
    }

    bool operator==(const BoolSplit& a, const BoolSplit& b)
    {
        return a.feat_id == b.feat_id;
    }

    std::ostream&
    operator<<(std::ostream& s, const Split& split)
    {
        visit_split(
            [&s](const LtSplit& x) { s << "LtSplit(" << x.feat_id << ", " << x.split_value << ')'; },
            [&s](const BoolSplit& x) { s << "BoolSplit(" << x.feat_id << ')'; },
            split);
        return s;
    }

    bool
    operator==(const Split& a, const Split& b)
    {
        return visit_split(
            [b](const LtSplit& x) {
                return std::holds_alternative<LtSplit>(b) && x == std::get<LtSplit>(b);
            },
            [b](const BoolSplit& x) {
                return std::holds_alternative<BoolSplit>(b) && x == std::get<BoolSplit>(b);
            }, a);
    }



    VERITAS_INSTANTIATE_TREE_TEMPLATE(Split, FloatT);


    template <> // implement NodeRef::get_domains for AddTree nodes
    template <>
    void
    NodeRef<inner::ConstRef<Split, FloatT>>::get_domains<Split>(DomainsT& domains)
    {
        auto node = *this;
        while (!node.is_root())
        {
            auto child_node = node;
            node = node.parent();
            refine_domains(domains, node.get_split(), child_node.is_left_child());
        }
    }


    AddTree::AddTree()
        : trees_{}
        , base_score(0.0)
    {
        trees_.reserve(16);
    }

    size_t
    AddTree::add_tree(AddTree::TreeT&& tree)
    {
        size_t index = trees_.size();
        trees_.push_back(std::move(tree));
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

    size_t
    AddTree::num_leafs() const
    {
        size_t c = 0;
        for (auto& tree : trees_)
            c += tree.num_leafs();
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
    AddTree::to_json() const
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

        FloatT base_score;
        std::vector<AddTree::TreeT> trees;

        {
            cereal::JSONInputArchive ar(ss);
            ar(cereal::make_nvp("base_score", base_score),
               cereal::make_nvp("trees", trees));
        }
        AddTree addtree;
        addtree.base_score = base_score;
        addtree.trees_ = std::move(trees);
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
                splits.emplace(split.feat_id, std::vector<FloatT>{split.split_value});
        }

        static
        void
        insert_split_value(AddTree::SplitMapT& splits, const BoolSplit& split)
        {
            auto search = splits.find(split.feat_id);
            if (search == splits.end()) // not in it yet, insert, otherwise, just reuse
                splits.emplace(split.feat_id, std::vector<FloatT>()); // empty vector
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

                visit_split(
                    [&splits](const LtSplit& x) { insert_split_value(splits, x); },
                    [&splits](const BoolSplit& x) { insert_split_value(splits, x); },
                    n.get_split());

                stack.push(n.right());
                stack.push(n.left());
            }
        }
    } /* namespace inner */

    AddTree::SplitMapT
    AddTree::get_splits() const
    {
        std::unordered_map<FeatId, std::vector<FloatT>> splits;

        // collect all the split values
        for (const TreeT& tree : trees_)
        {
            inner::visit_tree_nodes(tree, splits);
        }

        // sort the split values, remove duplicates
        for (auto& n : splits)
        {
            std::vector<FloatT>& v = n.second;
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }

        return splits;
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

} /* namespace veritas */
