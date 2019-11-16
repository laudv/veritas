#include "tree.hpp"
#include "addtree.h"

namespace treeck {

    TREECK_INSTANTIATE_TREE_TEMPLATE(double);

    AddTree::AddTree() : trees{}, base_score(0.0)
    {
        trees.reserve(16);
    }

    size_t
    AddTree::add_tree(AddTree::TreeT&& tree)
    {
        size_t index = trees.size();
        trees.push_back(std::forward<AddTree::TreeT>(tree));
        return index;
    }

    size_t
    AddTree::size() const
    {
        return trees.size();
    }

    AddTree::TreeT&
    AddTree::operator[](size_t index)
    {
        return trees[index];
    }

    const AddTree::TreeT&
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
            ar(cereal::make_nvp("base_score", base_score),
               cereal::make_nvp("trees", trees));
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
            ar(cereal::make_nvp("base_score", addtree.base_score),
               cereal::make_nvp("trees", addtree.trees));
        }
        return addtree;
    }

} /* namespace treeck */
