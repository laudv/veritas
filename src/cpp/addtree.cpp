#include "tree.hpp"
#include "addtree.h"

namespace treeck {

    TREECK_INSTANTIATE_TREE_TEMPLATE(double);

    AddTree::AddTree() : trees_{}, base_score(0.0)
    {
        trees_.reserve(16);
    }

    size_t
    AddTree::add_tree(AddTree::TreeT&& tree)
    {
        size_t index = trees_.size();
        trees_.push_back(std::forward<AddTree::TreeT>(tree));
        return index;
    }

    size_t
    AddTree::size() const
    {
        return trees_.size();
    }

    AddTree::TreeT&
    AddTree::operator[](size_t index)
    {
        return trees_[index];
    }

    const AddTree::TreeT&
    AddTree::operator[](size_t index) const
    {
        return trees_[index];
    }

    const std::vector<AddTree::TreeT>&
    AddTree::trees() const
    {
        return trees_;
    }

    std::string
    AddTree::to_json()
    {
        std::stringstream ss;
        {
            cereal::JSONOutputArchive ar(ss);
            ar(cereal::make_nvp("base_score", base_score),
               cereal::make_nvp("trees", trees_));
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
               cereal::make_nvp("trees", addtree.trees_));
        }
        return addtree;
    }

} /* namespace treeck */
