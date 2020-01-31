/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include <exception>
#include <limits>
#include <stack>
#include <iostream>
#include <utility>

#include "util.h"
#include "tree.h"

namespace treeck {
    template <typename Archive>
    void
    LtSplit::serialize(Archive& archive)
    {
        archive(CEREAL_NVP(feat_id), CEREAL_NVP(split_value));
    }

    //template <typename Archive>
    //void
    //EqSplit::serialize(Archive& archive)
    //{
    //    archive(CEREAL_NVP(feat_id), CEREAL_NVP(category));
    //}

    template <typename Archive>
    void
    BoolSplit::serialize(Archive& archive)
    {
        archive(CEREAL_NVP(feat_id));
    }

    template <typename SplitT>
    typename SplitT::DomainT
    refine_domain(const typename SplitT::DomainT& base_dom,
                  const SplitT& split, bool is_left_child)
    {
        if (is_left_child)
            return base_dom.intersect(std::get<0>(split.get_domains()));
        else
            return base_dom.intersect(std::get<1>(split.get_domains()));
    }

    template <typename SplitT>
    void
    refine_domains(DomainsT& domains, const SplitT& s, bool is_left_child)
    {
        typename SplitT::DomainT dom; // initially, is_everything == true

        auto domptr = domains.find(s.feat_id);
        if (domptr != domains.end())
            dom = util::get_or<typename SplitT::DomainT>(domptr->second);

        domains[s.feat_id] = refine_domain(dom, s, is_left_child);
    }

    namespace inner {

        template <typename SplitT, typename LeafT>
        Node<SplitT, LeafT>::Node() : Node(-1, -1) {}

        template <typename SplitT, typename LeafT>
        Node<SplitT, LeafT>::Node(NodeId id, NodeId parent)
            : id(id)
            , parent(parent)
            , tree_size(1)
            , leaf{} {}

        template <typename SplitT, typename LeafT>
        Node<SplitT, LeafT>::Node(const Node<SplitT, LeafT>& other)
            : id(other.id)
            , parent(other.parent)
            , tree_size(other.tree_size)
            , leaf{}
        {
            if (other.is_leaf())
                leaf = other.leaf;
            else
                internal = other.internal;
        }


        template <typename SplitT, typename LeafT>
        bool
        Node<SplitT, LeafT>::is_leaf() const
        {
            return tree_size == 1;
        }

        template <typename SplitT, typename LeafT>
        template <typename Archive>
        void
        Node<SplitT, LeafT>::serialize(Archive& archive)
        {
            archive(
                CEREAL_NVP(id),
                CEREAL_NVP(parent),
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
    const typename NodeRef<RefT>::NodeT&
    NodeRef<RefT>::node() const
    {
        return tree_->nodes_.at(node_id_);
    }

    template <typename RefT>
    template <typename T>
    std::enable_if_t<T::is_mut_type::value, typename NodeRef<RefT>::NodeT&>
    NodeRef<RefT>::node()
    {
        return tree_->nodes_.at(node_id_);
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
    bool
    NodeRef<RefT>::is_left_child() const
    {
        return !is_root() && parent().left().id() == id();
    }

    template <typename RefT>
    bool
    NodeRef<RefT>::is_right_child() const
    {
        return !is_root() && parent().right().id() == id();
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
        int depth = 0;
        NodeRef nf(*this);
        while (!nf.is_root())
        {
            nf = nf.parent();
            depth += 1;
        }
        return depth;
    }

    template <typename RefT>
    const typename NodeRef<RefT>::SplitT&
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
    NodeRef<RefT>::split(SplitT split)
    {
        if (is_internal()) throw std::runtime_error("split internal");

        NodeId left_id = tree_->nodes_.size();

        NodeT left(left_id,      id());
        NodeT right(left_id + 1, id());
        
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
    template <typename T>
    std::enable_if_t<T::is_mut_type::value, void>
    NodeRef<RefT>::skip_branch()
    {
        if (is_root()) throw std::runtime_error("skip_branch on root");

        auto& skipto_node = (is_left_child() ? parent().right() : parent().left()).node();
        auto& parent_node = parent().node();

        parent_node.tree_size = skipto_node.tree_size;
        if (skipto_node.is_leaf())
            parent_node.leaf = skipto_node.leaf;
        else
            parent_node.internal = skipto_node.internal;

        int skip_tree_size = tree_size();
        auto nf = parent();
        while (!nf.is_root())
        {
            nf = nf.parent();
            nf.node().tree_size -= (skip_tree_size + 1);
        }

        if (parent().is_internal())
        {
            parent().left().node().parent = parent().id();
            parent().right().node().parent = parent().id();
        }
        node().parent = -1;
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


    template <typename SplitT, typename LeafT>
    Tree<SplitT, LeafT>::Tree()
    {
        // NodeT must be default constructible
        nodes_.push_back({0, 0}); /* add a root leaf node */
    }

    template <typename SplitT, typename LeafT>
    typename Tree<SplitT, LeafT>::CRef
    Tree<SplitT, LeafT>::root() const
    {
        return (*this)[0];
    }

    template <typename SplitT, typename LeafT>
    typename Tree<SplitT, LeafT>::MRef
    Tree<SplitT, LeafT>::root()
    {
        return (*this)[0];
    }

    template <typename SplitT, typename LeafT>
    int
    Tree<SplitT, LeafT>::num_nodes() const
    {
        return nodes_[0].tree_size;
    }

    template <typename SplitT, typename LeafT>
    int
    Tree<SplitT, LeafT>::num_leafs() const
    {
        int num_leafs = 0;
        dfs([&num_leafs](auto node) {
            if (node.is_leaf())
            {
                num_leafs += 1;
                return TreeVisitStatus::ADD_NONE;
            }
            return TreeVisitStatus::ADD_LEFT_AND_RIGHT;
            });
        return num_leafs;
    }

    template <typename SplitT, typename LeafT>
    typename Tree<SplitT, LeafT>::CRef
    Tree<SplitT, LeafT>::operator[](NodeId index) const
    {
        return {this, index};
    }

    template <typename SplitT, typename LeafT>
    typename Tree<SplitT, LeafT>::MRef
    Tree<SplitT, LeafT>::operator[](NodeId index)
    {
        return {this, index};
    }

    template <typename SplitT, typename LeafT>
    template <typename TreeVisitorT>
    void
    Tree<SplitT, LeafT>::dfs(TreeVisitorT& visitor) const
    {
        std::stack<CRef> stack;
        stack.push(root());
        while (!stack.empty())
        {
            CRef node = stack.top(); stack.pop();
            TreeVisitStatus status = visitor(node);

            if ((status & TreeVisitStatus::ADD_RIGHT) > 0)
                stack.push(node.right());
            if ((status & TreeVisitStatus::ADD_LEFT) > 0)
                stack.push(node.left());
        }
    }

    template <typename SplitT, typename LeafT>
    template <typename TreeVisitorT>
    void
    Tree<SplitT, LeafT>::dfs(TreeVisitorT&& visitor) const
    {
        TreeVisitorT v = visitor;
        dfs(v);
    }

    template <typename SplitT, typename LeafT>
    template <typename Archive>
    void
    Tree<SplitT, LeafT>::serialize(Archive& archive)
    {
        archive(cereal::make_nvp("tree_nodes", this->nodes_));
    }

    template <typename SplitT, typename LeafT>
    std::string
    Tree<SplitT, LeafT>::to_json() const
    {
        std::stringstream ss;
        {
            cereal::JSONOutputArchive ar(ss);
            ar(cereal::make_nvp("tree_nodes", this->nodes_)); // destructor must run!
        }
        return ss.str();
    }

    template <typename SplitT, typename LeafT>
    Tree<SplitT, LeafT>
    Tree<SplitT, LeafT>::from_json(const std::string& json)
    {
        std::istringstream ss(json);
        Tree<SplitT, LeafT> tree;
        {
            cereal::JSONInputArchive ar(ss);
            ar(cereal::make_nvp("tree_nodes", tree.nodes_));
        }
        return tree;
    }

    template <typename SplitT, typename LeafT>
    std::ostream&
    operator<<(std::ostream& s, const Tree<SplitT, LeafT>& t)
    {
        s << "Tree(num_nodes=" << t.num_nodes() << ')' << std::endl;
        t.dfs([&s](typename Tree<SplitT, LeafT>::CRef n) {
            s << "  ";
            int i = 0;
            for (; i < n.depth() - 1; ++i) s << "│  ";
            for (; i < n.depth(); ++i)     s << (n.is_right_child() && n.is_leaf() ? "└─ " : "├─ ");
            s << n << std::endl;
            if (n.is_internal())
                return TreeVisitStatus::ADD_LEFT_AND_RIGHT;
            return TreeVisitStatus::ADD_NONE;
        });
        return s;
    }





    template <typename Archive>
    void
    AddTree::serialize(Archive& archive)
    {
        archive(cereal::make_nvp("base_score", base_score),
                cereal::make_nvp("trees", trees_));
    }

} /* namespace treeck */


#define TREECK_INSTANTIATE_TREE_TEMPLATE(S, T) \
    template class inner::Node<S, T>; \
    template class NodeRef<inner::ConstRef<S, T>>; \
    template class NodeRef<inner::MutRef<S, T>>; \
    template inner::Node<S, T>& NodeRef<inner::MutRef<S, T>>::node<inner::MutRef<S, T>>(); \
    template void NodeRef<inner::MutRef<S, T>>::set_leaf_value<inner::MutRef<S, T>>(T); \
    template void NodeRef<inner::MutRef<S, T>>::split<inner::MutRef<S, T>>(S); \
    template void NodeRef<inner::MutRef<S, T>>::skip_branch<inner::MutRef<S, T>>(); \
    template class Tree<S, T>; \
    template std::ostream& operator<<(std::ostream&, const NodeRef<inner::ConstRef<S, T>>&); \
    template std::ostream& operator<<(std::ostream&, const NodeRef<inner::MutRef<S, T>>&); \
    template std::ostream& operator<<(std::ostream&, const Tree<S, T>&)
