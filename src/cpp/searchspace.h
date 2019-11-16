#ifndef TREECK_SEARCHSPACE_H
#define TREECK_SEARCHSPACE_H

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "domain.h"
#include "tree.h"
#include "addtree.h"

namespace treeck {

    struct LeafInfo {
        template <typename Archive>
        void serialize(Archive& archive);
    };

    std::ostream& operator<<(std::ostream& s, LeafInfo inf);

    class SearchSpace {
    public:
        using TreeT = Tree<LeafInfo>;
        using SplitMap = std::unordered_map<FeatId, std::vector<double>>;
        using MeasureF = std::function<double(const SearchSpace&)>;
        using StopCondF = std::function<bool(const SearchSpace&)>;

    private:
        size_t num_features_;
        std::shared_ptr<AddTree> addtree_;
        TreeT domtree_; // domain tree
        SplitMap splits_map_;
        std::vector<NodeId> leafs_;

    public:
        SearchSpace(std::shared_ptr<AddTree> addtree);

        size_t num_features() const;
        const AddTree& addtree() const;
        const TreeT& domtree() const;
        const std::vector<NodeId>& leafs() const;
        void get_domains(NodeId node_id, std::vector<RealDomain>& domains);

        void split(MeasureF measure, StopCondF cond);
    };

    struct NumDisabledNodesMeasure {
        double operator()(const SearchSpace& sp);
    };

    struct SizeOfDomTreeStopCond {
        int max_num_of_nodes;
        bool operator()(const SearchSpace& sp);
    };

} /* namespace treeck */

#endif /* TREECK_SEARCHSPACE_H */
