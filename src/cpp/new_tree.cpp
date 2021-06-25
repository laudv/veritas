#include "new_tree.hpp"
#include <algorithm>

#include <iostream>
#include <stack>

namespace veritas {

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


    Tree
    Tree::prune(BoxRef box) const
    {
        std::stack<ConstRef, std::vector<ConstRef>> stack1;
        std::stack<MutRef, std::vector<MutRef>> stack2;

        Tree new_tree;
        stack1.push(root());
        stack2.push(new_tree.root());

        while (stack1.size() != 0)
        {
            ConstRef n1 = stack1.top();
            stack1.pop();
            MutRef n2 = stack2.top();

            if (n1.is_leaf())
            {
                stack2.pop();
                n2.set_leaf_value(n1.leaf_value());
            }
            else
            {
                Domain ldom, rdom;
                int flag = box.overlaps(n1.get_split());

                if (flag == (BoxRef::OVERLAPS_LEFT | BoxRef::OVERLAPS_RIGHT))
                {
                    stack2.pop();
                    n2.split(n1.get_split());
                    stack2.push(n2.right());
                    stack2.push(n2.left());
                }

                if ((flag & BoxRef::OVERLAPS_RIGHT) != 0)
                {
                    stack1.push(n1.right());
                }
                if ((flag & BoxRef::OVERLAPS_LEFT) != 0)
                {
                    stack1.push(n1.left());
                }
            }
        }

        return new_tree;
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

    AddTree
    AddTree::prune(BoxRef box) const
    {
        AddTree new_at;
        for (const Tree& t : *this)
            new_at.add_tree(t.prune(box));
        return new_at;
    }

    size_t
    AddTree::replace_feat_id(FeatId old_id, FeatId new_id)
    {
        size_t res = 0;
        for (Tree& t : *this)
            res += t.root().replace_feat_id(old_id, new_id);
        return res;
    }

    std::ostream&
    operator<<(std::ostream& strm, const AddTree& at)
    {
        return
            strm << "AddTree with " << at.size() << " trees and base_score "
                 << at.base_score << std::endl;
    }


} // namespace veritas
