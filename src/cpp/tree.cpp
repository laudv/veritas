#include <cassert>
#include <limits>
#include <type_traits>

#include "tree.h"

namespace treeck {

    SplitBase::SplitBase(FeatId feat_id) : feat_id(feat_id) {}

    LtSplit::LtSplit(FeatId feat_id, LtSplit::ValueT split_value)
        : SplitBase(feat_id)
        , split_value(split_value) {}

    std::tuple<RealDomain, RealDomain>
    LtSplit::get_domains() const
    {
        auto dom = RealDomain();
        return dom.split(this->split_value);
    }

    bool
    LtSplit::test(LtSplit::ValueT value) const
    {
        return value < this->split_value;
    }


    EqSplit::EqSplit(FeatId feat_id, EqSplit::ValueT category)
        : SplitBase(feat_id)
        , category(category) {}

    bool
    EqSplit::test(EqSplit::ValueT value) const
    {
        return value == this->category;
    }

    node::NodeLeaf::NodeLeaf(double value) : value(value) {}

    node::Node::Node(NodeId id, NodeId parent, int depth)
        : id(id)
        , parent(parent)
        , depth(depth)
        , tree_size(1)
        , leaf{std::numeric_limits<double>::quiet_NaN()} {}

    NodeRef::NodeRef(Tree *tree, NodeId node_id)
        : tree(tree)
        , node_id(node_id) {}

    //NodeRef::NodeRef(const NodeRef& other)
    //    : tree(other.tree)
    //    , node_id(other.node_id) {}

    //NodeRef::NodeRef(NodeRef&& other)
    //    : tree(other.tree)
    //    , node_id(other.node_id) {}

    //NodeRef&
    //NodeRef::operator=(NodeRef& other)
    //{
    //    this->tree = other.tree;
    //    this->node_id = other.node_id;
    //    return *this;
    //}

    //NodeRef&
    //NodeRef::operator=(NodeRef&& other)
    //{
    //    this->tree = std::move(other.tree);
    //    this->node_id = other.node_id;
    //    return *this;
    //}

    const node::Node&
    NodeRef::node() const
    {
        return tree->nodes[node_id];
    }

    node::Node&
    NodeRef::node()
    {
        return tree->nodes[node_id];
    }

    bool
    NodeRef::is_root() const
    {
        return node().parent == node().id;
    }

    bool
    NodeRef::is_leaf() const
    {
        return node().tree_size == 1;
    }

    bool
    NodeRef::is_internal() const
    {
        return !is_leaf();
    }

    NodeId
    NodeRef::id() const
    {
        return node().id;
    }

    NodeRef
    NodeRef::left() const
    {
        assert(is_internal());
        return NodeRef(tree, node().internal.left);
    }

    NodeRef
    NodeRef::right() const
    {
        assert(is_internal());
        return NodeRef(tree, node().internal.left + 1);
    }

    NodeRef
    NodeRef::parent() const
    {
        assert(!is_root());
        return NodeRef(tree, node().parent);
    }

    int
    NodeRef::tree_size() const
    {
        return node().tree_size;
    }

    int
    NodeRef::depth() const
    {
        return node().depth;
    }

    const Split&
    NodeRef::get_split() const
    {
        assert(is_internal());
        return node().internal.split;
    }

    double
    NodeRef::leaf_value() const
    {
        assert(is_leaf());
        return node().leaf.value;
    }

    void
    NodeRef::set_leaf_value(double value)
    {
        assert(is_leaf());
        node().leaf.value = value;
    }

    void
    NodeRef::split(Split split)
    {
        assert(is_leaf());

        node::Node& n = node();
        NodeId left_id = tree->nodes.size();

        node::Node left(left_id,      n.id, n.depth + 1);
        node::Node right(left_id + 1, n.id, n.depth + 1);
        
        tree->nodes.push_back(left);
        tree->nodes.push_back(right);

        n.internal.split = split;
        n.internal.left = left_id;

        NodeRef nf(*this);
        while (!nf.is_root())
        {
            nf.node().tree_size += 2;
            nf = nf.parent();
        }
    }

    Tree::Tree()
    {
        nodes.push_back(node::Node(0, 0, 0)); /* add a root leaf node */
    }

    NodeRef
    Tree::root()
    {
        return NodeRef(this, 0);
    }
}
