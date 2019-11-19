#include "util.h"
#include "tree.h"

namespace treeck {

    SplitBase::SplitBase(FeatId feat_id) : feat_id(feat_id) {}

    LtSplit::LtSplit() : LtSplit(-1, 0.0) {}
    LtSplit::LtSplit(FeatId feat_id, LtSplit::ValueT split_value)
        : SplitBase(feat_id)
        , split_value(split_value) {}

    bool
    LtSplit::test(LtSplit::ValueT value) const
    {
        return value < this->split_value;
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

    AddTree
    AddTree::from_json_file(const char *file)
    {
        std::ifstream t(file);
        std::stringstream buffer;
        buffer << t.rdbuf();
        std::string s(buffer.str());

        return AddTree::from_json(s);
    }

} /* namespace treeck */
