#include <exception>
#include <limits>
#include <stack>
#include <iostream>
#include <utility>

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

    namespace inner {

        template <typename LeafT>
        Node<LeafT>::Node() : Node(-1, -1, -1) {}

        template <typename LeafT>
        Node<LeafT>::Node(NodeId id, NodeId parent, int depth)
            : id(id)
            , parent(parent)
            , depth(depth)
            , tree_size(1)
            , leaf{} {}

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

    } /* namespace inner */

    template <typename RefT>
    NodeRef<RefT>::NodeRef(TreeP tree, NodeId node_id)
        : tree_(tree)
        , node_id_(node_id) {}

    template <typename RefT>
    const inner::Node<typename NodeRef<RefT>::LeafT>&
    NodeRef<RefT>::node() const
    {
        return tree_->nodes_[node_id_];
    }

    template <typename RefT>
    template <typename T>
    std::enable_if_t<T::is_mut_type::value, inner::Node<typename RefT::LeafT>&>
    NodeRef<RefT>::node()
    {
        return tree_->nodes_[node_id_];
    }

    template <typename RefT>
    bool
    NodeRef<RefT>::is_root() const
    {
        return node().parent == node().id;
    }

    template <typename RefT>
    bool
    NodeRef<RefT>::is_leaf() const
    {
        return node().is_leaf();
    }

    template <typename RefT>
    bool
    NodeRef<RefT>::is_internal() const
    {
        return !is_leaf();
    }

    template <typename RefT>
    NodeId
    NodeRef<RefT>::id() const
    {
        return node().id;
    }

    template <typename RefT>
    NodeRef<RefT>
    NodeRef<RefT>::left() const
    {
        if (is_leaf()) throw std::runtime_error("left of leaf");
        return NodeRef(tree_, node().internal.left);
    }

    template <typename RefT>
    NodeRef<RefT>
    NodeRef<RefT>::right() const
    {
        if (is_leaf()) throw std::runtime_error("right of leaf");
        return NodeRef(tree_, node().internal.left + 1);
    }

    template <typename RefT>
    NodeRef<RefT>
    NodeRef<RefT>::parent() const
    {
        if (is_root()) throw std::runtime_error("parent of root");
        return NodeRef(tree_, node().parent);
    }

    template <typename RefT>
    int
    NodeRef<RefT>::tree_size() const
    {
        return node().tree_size;
    }

    template <typename RefT>
    int
    NodeRef<RefT>::depth() const
    {
        return node().depth;
    }

    template <typename RefT>
    const Split&
    NodeRef<RefT>::get_split() const
    {
        if (is_leaf()) throw std::runtime_error("split of leaf");
        return node().internal.split;
    }

    template <typename RefT>
    typename RefT::LeafT
    NodeRef<RefT>::leaf_value() const
    {
        if (is_internal()) throw std::runtime_error("leaf_value of internal");
        return node().leaf.value;
    }

    template <typename RefT>
    template <typename T>
    std::enable_if_t<T::is_mut_type::value, void>
    NodeRef<RefT>::set_leaf_value(LeafT value)
    {
        if (is_internal()) throw std::runtime_error("set leaf_value of internal");
        node().leaf.value = value;
    }

    template <typename RefT>
    template <typename T>
    std::enable_if_t<T::is_mut_type::value, void>
    NodeRef<RefT>::split(Split split)
    {
        if (is_internal()) throw std::runtime_error("split internal");

        NodeId left_id = tree_->nodes_.size();

        inner::Node<LeafT> left(left_id,      id(), depth() + 1);
        inner::Node<LeafT> right(left_id + 1, id(), depth() + 1);
        
        tree_->nodes_.push_back(left);
        tree_->nodes_.push_back(right);

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

    template <typename RefT>
    std::ostream&
    operator<<(std::ostream& s, const NodeRef<RefT>& n)
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
        nodes_.push_back(inner::Node<LeafT>(0, 0, 0)); /* add a root leaf node */
    }

    template <typename LeafT>
    typename Tree<LeafT>::CRef
    Tree<LeafT>::root() const
    {
        return (*this)[0];
    }

    template <typename LeafT>
    typename Tree<LeafT>::MRef
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
    typename Tree<LeafT>::CRef
    Tree<LeafT>::operator[](NodeId index) const
    {
        return typename Tree<LeafT>::CRef(this, index);
    }

    template <typename LeafT>
    typename Tree<LeafT>::MRef
    Tree<LeafT>::operator[](NodeId index)
    {
        return typename Tree<LeafT>::MRef(this, index);
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
    operator<<(std::ostream& s, const Tree<LeafT>& t)
    {
        s << "Tree(num_nodes=" << t.num_nodes() << ')' << std::endl;
        std::stack<typename Tree<LeafT>::CRef> stack;
        stack.push(t.root());
        while (!stack.empty())
        {
            auto n = stack.top(); stack.pop();
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
    template class inner::Node<T>; \
    template class NodeRef<inner::ConstRef<T>>; \
    template class NodeRef<inner::MutRef<T>>; \
    template inner::Node<T>& NodeRef<inner::MutRef<T>>::node<inner::MutRef<T>>(); \
    template void NodeRef<inner::MutRef<T>>::set_leaf_value<inner::MutRef<T>>(T); \
    template void NodeRef<inner::MutRef<T>>::split<inner::MutRef<T>>(Split); \
    template class Tree<T>; \
    template std::ostream& operator<<(std::ostream&, const NodeRef<inner::ConstRef<T>>&); \
    template std::ostream& operator<<(std::ostream&, const NodeRef<inner::MutRef<T>>&); \
    template std::ostream& operator<<(std::ostream&, const Tree<T>&)
