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

#include <unordered_map>

#include "domain.h"
#include "tree.h"

#ifndef TREECK_GRAPH_H
#define TREECK_GRAPH_H

namespace treeck {

    class DomainBox {
        std::vector<Domain>& data_;
        size_t begin;
        size_t sz;

    private:
        void check_bounds(size_t i) const;

    public:
        DomainBox(std::vector<Domain>& data, size_t begin, size_t sz);

        const Domain& operator[](size_t i) const;
        Domain& operator[](size_t i);

        void intersect(const DomainBox& other);

    };

    struct Vertex {
        DomainBox box;
        FloatT output;
    };

    struct IndependentSet {
        std::vector<Vertex> vertices_;
    };

    class KPartiteGraph {
        size_t nfeatures_;
        size_t ninstances_;
        std::vector<IndependentSet> sets_;
        std::vector<Domain> domains_buffer_;
        std::unordered_map<FeatId, size_t> feat_id_map_;

    private:
        size_t map_feat_id(FeatId feat_id);
        DomainBox create_box();
        void fill_independence_set(IndependentSet& set, AddTree::TreeT::CRef node);

    public:
        KPartiteGraph();

        void add_instance(const AddTree& addtree);
    };

} /* namespace treeck */

#endif /* TREECK_GRAPH_H */
