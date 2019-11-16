#include <exception>
#include <limits>
#include <stack>
#include <iostream>
#include <sstream>
#include <utility>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/variant.hpp>

#include "tree.h"

namespace treeck {

    template <typename Archive>
    void
    LtSplit::serialize(Archive& archive)
    {
        archive(CEREAL_NVP(feat_id), CEREAL_NVP(split_value));
    }

    template <typename Archive>
    void
    EqSplit::serialize(Archive& archive)
    {
        archive(CEREAL_NVP(feat_id), CEREAL_NVP(category));
    }

    namespace node {

        template <typename LeafT>
        Node<LeafT>::Node() : Node(-1, -1, -1) {}

        template <typename LeafT>
        Node<LeafT>::Node(NodeId id, NodeId parent, int depth)
            : id(id)
            , parent(parent)
            , depth(depth)
            , tree_size(1)
            , leaf{std::numeric_limits<double>::quiet_NaN()} {}

        template <typename LeafT>
        bool
        Node<LeafT>::is_leaf() const
        {
            return tree_size == 1;
        }

        template <typename LeafT>
        template <typename Archive>
        void
        Node<LeafT>::serialize(Archive& archive)
        {
            archive(
                CEREAL_NVP(id),
                CEREAL_NVP(parent),
                CEREAL_NVP(depth),
                CEREAL_NVP(tree_size));

            if (is_leaf()) // uses tree_size read above when deserializing!
            {
                archive(cereal::make_nvp("leaf_value", leaf.value));
            }
            else
            {
                archive(
                    cereal::make_nvp("left", internal.left),
                    cereal::make_nvp("split", internal.split));
            }
        }

    } /* namespace node */

    template <typename LeafT>
    NodeRef<LeafT>::NodeRef(TreeP tree, NodeId node_id)
        : tree(tree)
        , node_id(node_id) {}

    template <typename LeafT>
    const node::Node<LeafT>&
    NodeRef<LeafT>::node() const
    {
        return tree->nodes_[node_id];
    }

    template <typename LeafT>
    node::Node<LeafT>&
    NodeRef<LeafT>::node()
    {
        return tree->nodes_[node_id];
    }

    template <typename LeafT>
    bool
    NodeRef<LeafT>::is_root() const
    {
        return node().parent == node().id;
    }

    template <typename LeafT>
    bool
    NodeRef<LeafT>::is_leaf() const
    {
        return node().is_leaf();
    }

    template <typename LeafT>
    bool
    NodeRef<LeafT>::is_internal() const
    {
        return !is_leaf();
    }

    template <typename LeafT>
    NodeId
    NodeRef<LeafT>::id() const
    {
        return node().id;
    }

    template <typename LeafT>
    NodeRef<LeafT>
    NodeRef<LeafT>::left() const
    {
        if (is_leaf()) throw std::runtime_error("left of leaf");
        return NodeRef(tree, node().internal.left);
    }

    template <typename LeafT>
    NodeRef<LeafT>
    NodeRef<LeafT>::right() const
    {
        if (is_leaf()) throw std::runtime_error("right of leaf");
        return NodeRef(tree, node().internal.left + 1);
    }

    template <typename LeafT>
    NodeRef<LeafT>
    NodeRef<LeafT>::parent() const
    {
        if (is_root()) throw std::runtime_error("parent of root");
        return NodeRef(tree, node().parent);
    }

    template <typename LeafT>
    int
    NodeRef<LeafT>::tree_size() const
    {
        return node().tree_size;
    }

    template <typename LeafT>
    int
    NodeRef<LeafT>::depth() const
    {
        return node().depth;
    }

    template <typename LeafT>
    const Split&
    NodeRef<LeafT>::get_split() const
    {
        if (is_leaf()) throw std::runtime_error("split of leaf");
        return node().internal.split;
    }

    template <typename LeafT>
    LeafT
    NodeRef<LeafT>::leaf_value() const
    {
        if (is_internal()) throw std::runtime_error("leaf_value of internal");
        return node().leaf.value;
    }

    template <typename LeafT>
    void
    NodeRef<LeafT>::set_leaf_value(LeafT value)
    {
        if (is_internal()) throw std::runtime_error("set leaf_value of internal");
        node().leaf.value = value;
    }

    template <typename LeafT>
    void
    NodeRef<LeafT>::split(Split split)
    {
        if (is_internal()) throw std::runtime_error("split internal");

        NodeId left_id = tree->nodes_.size();

        node::Node<LeafT> left(left_id,      id(), depth() + 1);
        node::Node<LeafT> right(left_id + 1, id(), depth() + 1);
        
        tree->nodes_.push_back(left);
        tree->nodes_.push_back(right);

        node().internal.split = split;
        node().internal.left = left_id;

        node().tree_size = 3;
        NodeRef nf(*this);
        while (!nf.is_root())
        {
            nf = nf.parent();
            nf.node().tree_size += 2;
        }
    }

    template <typename LeafT>
    std::ostream&
    operator<<(std::ostream& s, const NodeRef<LeafT>& n)
    {
        if (n.is_leaf())
            return s << "LeafNode("
                << "id=" << n.id()
                << ", value=" << n.leaf_value()
                << ')';
        else
            return s << "InternalNode("
                << "id=" << n.id()
                << ", split=" << n.get_split()
                << ", left=" << n.left().id()
                << ", right=" << n.right().id()
                << ')';
    }


    template <typename LeafT>
    Tree<LeafT>::Tree()
    {
        nodes_.push_back(node::Node<LeafT>(0, 0, 0)); /* add a root leaf node */
    }

    template <typename LeafT>
    NodeRef<LeafT>
    Tree<LeafT>::root()
    {
        return (*this)[0];
    }

    template <typename LeafT>
    int
    Tree<LeafT>::num_nodes() const
    {
        return nodes_[0].tree_size;
    }

    template <typename LeafT>
    std::tuple<unsigned long long int, unsigned long long int>
    Tree<LeafT>::id() const
    {
        return std::make_tuple(
                reinterpret_cast<unsigned long long int>(this),
                reinterpret_cast<unsigned long long int>(nodes_.data()));
    }

    template <typename LeafT>
    NodeRef<LeafT>
    Tree<LeafT>::operator[](NodeId index)
    {
        return NodeRef(this, index);
    }

    template <typename LeafT>
    template <typename Archive>
    void
    Tree<LeafT>::serialize(Archive& archive)
    {
        archive(cereal::make_nvp("tree_nodes", this->nodes_));
    }

    template <typename LeafT>
    std::string
    Tree<LeafT>::to_json()
    {
        std::stringstream ss;
        {
            cereal::JSONOutputArchive ar(ss);
            ar(cereal::make_nvp("tree_nodes", this->nodes_)); // destructor must run!
        }
        return ss.str();
    }

    template <typename LeafT>
    Tree<LeafT>
    Tree<LeafT>::from_json(const std::string& json)
    {
        std::istringstream ss(json);
        Tree<LeafT> tree;
        {
            cereal::JSONInputArchive ar(ss);
            ar(cereal::make_nvp("tree_nodes", tree.nodes_));
        }
        return tree;
    }

    template <typename LeafT>
    std::ostream&
    operator<<(std::ostream& s, Tree<LeafT>& t)
    {
        s << "Tree(num_nodes=" << t.num_nodes() << ')' << std::endl;
        std::stack<NodeRef<LeafT>> stack;
        stack.push(t.root());
        while (!stack.empty())
        {
            NodeRef<LeafT> n = stack.top(); stack.pop();
            s << "  ";
            for (int i = 0; i < n.depth(); ++i)
                s << " | ";
            s << n << std::endl;
            if (n.is_internal())
            {
                stack.push(n.right());
                stack.push(n.left());
            }
        }
        return s;
    }

} /* namespace treeck */


#define TREECK_INSTANTIATE_TREE_TEMPLATE(T) \
    template class node::Node<T>; \
    template class NodeRef<T>; \
    template class Tree<T>; \
    template std::ostream& operator<<(std::ostream&, const NodeRef<T>&); \
    template std::ostream& operator<<(std::ostream&, Tree<T>&)
