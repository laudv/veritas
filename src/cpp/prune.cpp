
#include <exception>
#include <stack>
#include "prune.h"


namespace treeck {

    AddTree
    prune(const AddTree& addtree, const Domains& domains)
    {
        AddTree new_addtree;
        new_addtree.base_score = addtree.base_score;
        for (const AddTree::TreeT& tree : addtree.trees())
            new_addtree.add_tree(prune(tree, domains));
        return new_addtree;
    }

    AddTree::TreeT
    prune(const AddTree::TreeT& tree1, const Domains& domains)
    {
        AddTree::TreeT tree2;
        std::stack<AddTree::TreeT::CRef> stack1;
        std::stack<AddTree::TreeT::MRef> stack2;
        stack1.push(tree1.root());
        stack2.push(tree2.root());

        while (!stack1.empty())
        {
            if (stack2.empty()) throw std::runtime_error("stack2 empty");

            auto node1 = stack1.top(); stack1.pop();
            if (node1.is_leaf())
            {
                auto node2 = stack2.top(); stack2.pop();
                node2.set_leaf_value(node1.leaf_value());
                continue;
            }
            
            LtSplit split = std::get<LtSplit>(node1.get_split());
            RealDomain dom = domains[split.feat_id];
            
            switch (dom.where_is_strict(split.split_value)) //!!
            {
            case WhereFlag::LEFT:                       // x     (---)  [everything def. goes right]
                stack1.push(node1.right());
                break;
            case WhereFlag::RIGHT:                      // (---)     x  [everything def. goes left]
                stack1.push(node1.left());
                break;
            default:                                    // (---x---)    [could go either way]
                auto node2 = stack2.top(); stack2.pop();
                node2.split(split);
                stack1.push(node1.right());
                stack1.push(node1.left());
                stack2.push(node2.right());
                stack2.push(node2.left());
                break;
            }
        }

        //tree.dfs([&domains, &new_tree, &stack](AddTree::TreeT::CRef node) {
        //    if (stack.empty()) throw std::runtime_error("stack empty");

        //    std::cout << "STACK SIZE " << stack.size() << std::endl;
        //    if (node.is_leaf())
        //    {
        //        auto new_node = stack.top(); stack.pop();
        //        new_node.set_leaf_value(node.leaf_value());
        //        std::cout << "LEAF " << node << " <> " << new_node << std::endl;
        //        return TreeVisitStatus::ADD_NONE;
        //    }

        //    LtSplit split = std::get<LtSplit>(node.get_split());
        //    RealDomain dom = domains[split.feat_id];
        //    
        //    switch (dom.where_is(split.split_value))
        //    {
        //    case WhereFlag::LEFT:
        //        std::cout << "skipping right " << node.right().id() << std::endl;
        //        return TreeVisitStatus::ADD_LEFT;
        //    case WhereFlag::RIGHT:
        //        std::cout << "skipping left " << node.left().id() << std::endl;
        //        return TreeVisitStatus::ADD_RIGHT;
        //    default:
        //        auto new_node = stack.top(); stack.pop();
        //        new_node.split(split);
        //        stack.push(new_node.right());
        //        stack.push(new_node.left());
        //        std::cout << "INTERNAL " << node << " <> " << new_node << std::endl;
        //        return TreeVisitStatus::ADD_LEFT_AND_RIGHT;
        //    }
        //});

        return tree2;
    }
} /* namespace treeck */
