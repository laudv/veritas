#include "new_tree.hpp"
#include <algorithm>

#include <iostream>

namespace veritas {

    inline std::ostream& operator<<(std::ostream& strm, const LtSplit& s)
    {
        strm << "LtSplit(" << s.feat_id << ", " << s.split_value << ')';
        return strm;
    }

    namespace inner {

        static void
        compute_box(
                Tree::ConstRef node,
                Box& box,
                bool from_left_child)
        {
            refine_box(box, node.get_split(), from_left_child);
            
            // repeat this for each internal node on the node-to-root path
            if (!node.is_root())
                compute_box(node.parent(), box, node.is_left_child());
        }

    } // namespace inner

    template <typename RefT>
    Box
    NodeRef<RefT>::compute_box() const
    {
        Box box;
        compute_box(box);
        return box;
    }

    template <typename RefT>
    void
    NodeRef<RefT>::compute_box(Box& box) const
    {
        if (!is_root())
            inner::compute_box(parent().to_const(), box, is_left_child());
    }

    template // manual template instantiation
    Box
    NodeRef<inner::ConstRef>::compute_box() const;

    template // manual template instantiation
    Box
    NodeRef<inner::MutRef>::compute_box() const;

    template <typename RefT>
    void
    NodeRef<RefT>::print_node(std::ostream& strm, int depth)
    {
        for (int i = 0; i < depth; ++i)
            strm << "│  ";
        if (is_leaf())
        {
            strm << (is_right_child() ? "└─ " : "├─ ")
                << "Leaf("
                << "id=" << id()
                << ", value=" << leaf_value()
                << ')' << std::endl;
        }
        else
        {
            strm << "├─ Node("
                << "id=" << id()
                << ", split=" << get_split()
                << ", left=" << left().id()
                << ", right=" << right().id()
                << ')' << std::endl;
            left().print_node(strm, depth+1);
            right().print_node(strm, depth+1);
        }
    }


    std::ostream&
    operator<<(std::ostream& strm, const Tree& t)
    {
        t.root().print_node(strm, 0);
        return strm;
    }

    size_t AddTree::num_nodes() const
    {
        size_t c = 0;
        for (const auto& t : trees_)
            c += t.num_nodes();
        return c;
    }

    size_t AddTree::num_leafs() const
    {
        size_t c = 0;
        for (const auto& t : trees_)
            c += t.num_leafs();
        return c;
    }

    namespace inner {
        static
        void
        collect_split_values(AddTree::SplitMapT& splits, const Tree::ConstRef& node)
        {
            if (node.is_leaf()) return;

            // insert split values
            const LtSplit& split = node.get_split();
            auto search = splits.find(split.feat_id);
            if (search != splits.end()) // found it!
                splits[split.feat_id].push_back(split.split_value);
            else
                splits.emplace(split.feat_id, std::vector<FloatT>{split.split_value});

            collect_split_values(splits, node.right());
            collect_split_values(splits, node.left());
        }

    } /* namespace inner */

    std::unordered_map<FeatId, std::vector<FloatT>>
    AddTree::get_splits() const
    {
        std::unordered_map<FeatId, std::vector<FloatT>> splits;

        // collect all the split values
        for (const Tree& tree : trees_)
            inner::collect_split_values(splits, tree.root());

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
    operator<<(std::ostream& strm, const AddTree& at)
    {
        return
            strm << "AddTree with " << at.size() << " trees and base_score "
                 << at.base_score << std::endl;
    }


} // namespace veritas
