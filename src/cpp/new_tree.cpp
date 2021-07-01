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
    NodeRef<RefT>::print_node(std::ostream& strm, int depth) const
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

    template <typename RefT>
    void
    NodeRef<RefT>::to_json(std::ostream& s, int depth) const
    {
        if (is_leaf())
        {
            s << "{\"leaf_value\": " << leaf_value() << "}";
        }
        else
        {
            s << "{\"feat_id\": " << get_split().feat_id
                << ", \"split_value\": " << get_split().split_value
                << ',' << std::endl;
            for (int i = 0; i <= depth; ++i) s << "  ";
            s << "\"lt\": ";
            left().to_json(s, depth+1);
            s << ',' << std::endl;
            for (int i = 0; i <= depth; ++i) s << "  ";
            s << "\"gteq\": ";
            right().to_json(s, depth+1);
            s << std::endl;
            for (int i = 0; i < depth; ++i) s << "  ";
            s << '}';
        }
    }

    template void NodeRef<inner::ConstRef>::to_json(std::ostream&, int) const;
    template void NodeRef<inner::MutRef>::to_json(std::ostream&, int) const;

    template <typename RefT>
    template <typename T>
    std::enable_if_t<T::is_mut_type::value, void>
    NodeRef<RefT>::from_json(std::istream& s)
    {
        std::string buf;
        FloatT leaf_value, split_value;
        FeatId feat_id;
        char c;

        while (s.get(c))
        {
            switch (c)
            {
                case ' ':
                case '\n':
                case '{':
                case ',':
                    break;
                case '"':
                case '\'':
                    buf.clear();
                    while (s.get(c))
                        if (c != '"') buf.push_back(c); else break;
                    break;
                case '}':
                    goto the_end;
                case ':':
                    if (buf == "feat_id")
                        s >> feat_id;
                    else if (buf == "split_value")
                        s >> split_value;
                    else if (buf == "leaf_value")
                    {
                        s >> leaf_value;
                        set_leaf_value(leaf_value);
                    }
                    else if (buf == "lt") // left branch
                    {
                        split({feat_id, split_value});
                        left().from_json(s);
                    }
                    else if (buf == "gteq") // expected after lt (split already applied)
                    {
                        right().from_json(s);
                    }
                    else
                        throw std::runtime_error("tree parse error: unknown key");
                    break;
                default:
                    throw std::runtime_error("tree parse error: unexpected char");
            }
        }

        the_end: return;
    }

    template void NodeRef<inner::MutRef>::from_json(std::istream&);

    template <typename RefT>
    template <typename D>
    FloatT
    NodeRef<RefT>::eval(const row<D>& row) const
    {
        if (is_leaf()) return leaf_value();
        return get_split().test(row) ? left().eval(row) : right().eval(row);
    }

    template FloatT NodeRef<inner::ConstRef>::eval(const row<row_major_data>&) const;
    template FloatT NodeRef<inner::ConstRef>::eval(const row<col_major_data>&) const;
    template FloatT NodeRef<inner::MutRef>::eval(const row<row_major_data>&) const;
    template FloatT NodeRef<inner::MutRef>::eval(const row<col_major_data>&) const;

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

    std::ostream&
    operator<<(std::ostream& strm, const AddTree& at)
    {
        return
            strm << "AddTree with " << at.size() << " trees and base_score "
                 << at.base_score;
    }

    void
    AddTree::to_json(std::ostream& s) const
    {
        s << "{\"base_score\": " << base_score
            << ", \"trees\": [" << std::endl;
        auto it = begin();
        if (it != end())
            (it++)->to_json(s);
        for (; it != end(); ++it)
        {
            s << ',' << std::endl;
            it->to_json(s);
        }

        s << "]}";
    }

    void
    AddTree::from_json(std::istream& s)
    {
        std::string buf;
        char c;
        Tree tree;

        while (s.get(c))
        {
            switch (c)
            {
                case ' ':
                case '\n':
                case ',':
                case '{':
                    break;
                case '"':
                case '\'':
                    buf.clear();
                    while (s.get(c))
                        if (c != '"') buf.push_back(c); else break;
                    break;
                case ':':
                    if (buf == "base_score")
                        s >> base_score;
                    else if (buf == "trees")
                        goto loop2;
                    else
                        throw std::runtime_error("addtree parse error: unknown key");
                    break;
                default:
                    throw std::runtime_error("addtree parse error: unexpected char");
            }
        }

        loop2: while (s.get(c))
        {
            switch (c)
            {
                case ' ':
                case ']':
                case '}':
                case '\n':
                    break;
                case '[':
                case ',':
                    tree.from_json(s);
                    add_tree(std::move(tree));
                    tree.clear();
                    break;
                default:
                    throw std::runtime_error("addtree parse error (2): unexpected char");
            }
        }
    } 

    void
    AddTree::compute_box(Box& box, const std::vector<NodeId> node_ids) const
    {
        if (size() != node_ids.size())
            throw std::runtime_error("compute_box: one node_id per tree in AddTree");

        for (size_t tree_index = 0; tree_index < size(); ++tree_index)
        {
            NodeId leaf_id = node_ids[tree_index];
            auto node = trees_[tree_index][leaf_id];
            node.compute_box(box);
        }
    }



} // namespace veritas
