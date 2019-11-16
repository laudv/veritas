#ifndef TREECK_SEARCHSPACE_H
#define TREECK_SEARCHSPACE_H

#include <vector>
#include <unordered_map>
#include <memory>
#include "domain.h"
#include "tree.h"
#include "addtree.h"

namespace treeck {

    struct LeafInfo {
        double x;

        template <typename Archive>
        void serialize(Archive& archive);
    };

    using SplitMap = std::unordered_map<FeatId, std::vector<double>>;
    using DomMap = std::unordered_map<FeatId, RealDomain>;
    
    class SearchSpace {
        using TreeT = Tree<LeafInfo>;

        size_t num_features_;
        std::shared_ptr<AddTree> addtree_;
        TreeT domtree_; // domain tree
        SplitMap splits_;

    public:
        SearchSpace(std::shared_ptr<AddTree> addtree);

        size_t num_features() const;
        const AddTree& addtree() const;
        const TreeT& domtree() const;

        template <typename Improvement, typename StopCond>
        void split(Improvement measure, StopCond cond);
    };

} /* namespace treeck */

#endif /* TREECK_SEARCHSPACE_H */
