#include <exception>
#include <limits>
#include <type_traits>
#include <stack>
#include <iostream>
#include <sstream>
#include <utility>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/variant.hpp>

#include "tree.h"
//#include "opaque.h"

namespace treeck {

    SplitBase::SplitBase(FeatId feat_id) : feat_id(feat_id) {}

    LtSplit::LtSplit() : LtSplit(-1, 0.0) {}
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

    template <typename Archive>
    void
    LtSplit::serialize(Archive& archive)
    {
        archive(CEREAL_NVP(feat_id), CEREAL_NVP(split_value));
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

    template <typename Archive>
    void
    EqSplit::serialize(Archive& archive)
    {
        archive(CEREAL_NVP(feat_id), CEREAL_NVP(category));
    }

    namespace util {
        template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
        template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
        template<class T> struct always_false : std::false_type {};
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

    namespace node {

        Node::Node() : Node(-1, -1, -1) {}
        Node::Node(NodeId id, NodeId parent, int depth)
            : id(id)
            , parent(parent)
            , depth(depth)
            , tree_size(1)
            , leaf{std::numeric_limits<double>::quiet_NaN()} {}

        bool
        Node::is_leaf() const
        {
            return tree_size == 1;
        }

        template <typename Archive>
        void
        Node::serialize(Archive& archive)
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

    NodeRef::NodeRef(TreeP tree, NodeId node_id)
        : tree(tree)
        , node_id(node_id) {}

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
        return node().is_leaf();
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
        if (is_leaf()) throw std::runtime_error("left of leaf");
        return NodeRef(tree, node().internal.left);
    }

    NodeRef
    NodeRef::right() const
    {
        if (is_leaf()) throw std::runtime_error("right of leaf");
        return NodeRef(tree, node().internal.left + 1);
    }

    NodeRef
    NodeRef::parent() const
    {
        if (is_root()) throw std::runtime_error("parent of root");
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
        if (is_leaf()) throw std::runtime_error("split of leaf");
        return node().internal.split;
    }

    double
    NodeRef::leaf_value() const
    {
        if (is_internal()) throw std::runtime_error("leaf_value of internal");
        return node().leaf.value;
    }

    void
    NodeRef::set_leaf_value(double value)
    {
        if (is_internal()) throw std::runtime_error("set leaf_value of internal");
        node().leaf.value = value;
    }

    void
    NodeRef::split(Split split)
    {
        if (is_internal()) throw std::runtime_error("split internal");

        NodeId left_id = tree->nodes.size();

        node::Node left(left_id,      id(), depth() + 1);
        node::Node right(left_id + 1, id(), depth() + 1);
        
        tree->nodes.push_back(left);
        tree->nodes.push_back(right);

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

    std::ostream&
    operator<<(std::ostream& s, const NodeRef& n)
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

    Tree::Tree()
    {
        nodes.push_back(node::Node(0, 0, 0)); /* add a root leaf node */
        std::cout << "hi from tree " << this << " nodes=" << nodes.data() << std::endl;
    }

    Tree::~Tree()
    {
        std::cout << "bye from tree " << this << " nodes=" << nodes.data() << std::endl;
    }

    NodeRef
    Tree::root()
    {
        return (*this)[0];
    }

    int
    Tree::num_nodes() const
    {
        return nodes[0].tree_size;
    }

    std::tuple<unsigned long long int, unsigned long long int>
    Tree::id() const
    {
        return std::make_tuple(
                reinterpret_cast<unsigned long long int>(this),
                reinterpret_cast<unsigned long long int>(nodes.data()));
    }


    NodeRef
    Tree::operator[](NodeId index)
    {
        return NodeRef(this, index);
    }

    template <typename Archive>
    void
    Tree::serialize(Archive& archive)
    {
        archive(cereal::make_nvp("tree_nodes", this->nodes));
    }

    std::string
    Tree::to_json()
    {
        std::stringstream ss;
        {
            cereal::JSONOutputArchive ar(ss);
            ar(cereal::make_nvp("tree_nodes", this->nodes)); // destructor must run!
        }
        return ss.str();
    }

    Tree
    Tree::from_json(const std::string& json)
    {
        std::istringstream ss(json);
        Tree tree;
        {
            cereal::JSONInputArchive ar(ss);
            ar(cereal::make_nvp("tree_nodes", tree.nodes));
        }
        return tree;
    }

    std::ostream&
    operator<<(std::ostream& s, Tree& t)
    {
        s << "Tree(num_nodes=" << t.num_nodes() << ')' << std::endl;
        std::stack<NodeRef> stack;
        stack.push(t.root());
        while (!stack.empty())
        {
            NodeRef n = stack.top(); stack.pop();
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

    AddTree::AddTree()
    {
        trees.reserve(16);
    }

    size_t
    AddTree::add_tree(Tree&& tree)
    {
        size_t index = trees.size();
        trees.push_back(std::forward<Tree>(tree));
        return index;
    }

    size_t
    AddTree::size() const
    {
        return trees.size();
    }

    Tree&
    AddTree::operator[](size_t index)
    {
        std::cout << "accessing tree " << &trees[index] << std::endl;
        return trees[index];
    }

    const Tree&
    AddTree::operator[](size_t index) const
    {
        return trees[index];
    }

    std::string
    AddTree::to_json()
    {
        std::stringstream ss;
        {
            cereal::JSONOutputArchive ar(ss);
            ar(cereal::make_nvp("trees", trees));
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
            ar(cereal::make_nvp("trees", addtree.trees));
        }
        return addtree;
    }


} /* namespace treeck */
