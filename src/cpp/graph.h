/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
 *
 * ----
 *
 * Reimplementation of concepts introduced by the following paper:
 *
 * https://papers.nips.cc/paper/9399-robustness-verification-of-tree-based-models
 * https://github.com/chenhongge/treeVerification
 */

#include <tuple>
#include <vector>
#include <unordered_map>

#include "domain.h"
#include "tree.h"

#ifndef TREECK_GRAPH_H
#define TREECK_GRAPH_H

namespace treeck {

    class DomainBox {
        std::vector<std::pair<FeatId, Domain>> domains_;

    public:
        DomainBox();

        Domain& operator[](FeatId feat_id);

        std::vector<std::pair<FeatId, Domain>>::const_iterator begin() const;
        std::vector<std::pair<FeatId, Domain>>::const_iterator end() const;
        std::vector<std::pair<FeatId, Domain>>::const_iterator find(FeatId feat_id) const;
        std::vector<std::pair<FeatId, Domain>>::iterator find(FeatId feat_id);

        void refine(Split split, bool is_left_child);

        void sort();
    };

    std::ostream&
    operator<<(std::ostream& s, const DomainBox& box);

    struct Vertex {
        DomainBox box;
        FloatT output;
    };

    struct IndependentSet {
        std::vector<Vertex> vertices;
    };

    class KPartiteGraph {
        std::vector<IndependentSet> sets_;

    private:
        void fill_independence_set(IndependentSet& set, AddTree::TreeT::CRef node);

    public:
        KPartiteGraph(const AddTree& addtree);

        std::vector<IndependentSet>::const_iterator begin() const;
        std::vector<IndependentSet>::const_iterator end() const;
    };

    std::ostream&
    operator<<(std::ostream& s, const KPartiteGraph& graph);

} /* namespace treeck */

#endif /* TREECK_GRAPH_H */
