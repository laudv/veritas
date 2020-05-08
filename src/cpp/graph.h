/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
 *
 * ----
 *
 * This file contains reimplemplementations of concepts introduced by the
 * following paper Chen et al. 2019:
 *
 * https://papers.nips.cc/paper/9399-robustness-verification-of-tree-based-models
 * https://github.com/chenhongge/treeVerification
 */

#include <tuple>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <functional>

#include "domain.h"
#include "tree.h"

#ifndef TREECK_GRAPH_H
#define TREECK_GRAPH_H

namespace treeck {

    class DomainBox;

    using FeatIdMapper = const std::function<int(FeatId)>&;
    using BoxFilter = const std::function<bool(const DomainBox&)>&;



    class DomainBox {
        std::vector<std::pair<int, Domain>> domains_;

    public:
        DomainBox();

        std::vector<std::pair<int, Domain>>::const_iterator begin() const;
        std::vector<std::pair<int, Domain>>::const_iterator end() const;
        std::vector<std::pair<int, Domain>>::const_iterator find(int id) const;
        std::vector<std::pair<int, Domain>>::iterator find(int id);

        void refine(Split split, bool is_left_child, FeatIdMapper fmap);

        void sort();

        bool overlaps(const DomainBox& other) const;
        DomainBox combine(const DomainBox& other) const;
    };

    std::ostream&
    operator<<(std::ostream& s, const DomainBox& box);




    struct Vertex {
        DomainBox box;
        FloatT output;
        FloatT min_output;
        FloatT max_output;

        Vertex(DomainBox box, FloatT output);

        //bool operator<(const Vertex& other) const;
        //bool operator>(const Vertex& other) const;
    };




    struct IndependentSet {
        std::vector<Vertex> vertices;
    };




    template <typename Cmp>
    class KPartiteGraphFind;

    class KPartiteGraph {
        std::vector<IndependentSet> sets_;
        template <typename Cmp> friend class KPartiteGraphFind;

    private:
        void fill_independence_set(IndependentSet& set,
                AddTree::TreeT::CRef node,
                FeatIdMapper fmap);

    public:
        KPartiteGraph();
        KPartiteGraph(const AddTree& addtree);
        KPartiteGraph(const AddTree& addtree, FeatIdMapper fmap);

        std::vector<IndependentSet>::const_iterator begin() const;
        std::vector<IndependentSet>::const_iterator end() const;

        /** remove all vertices for which the given function returns true. */
        void prune(BoxFilter filter);

        std::tuple<FloatT, FloatT> propagate_outputs();
        void merge(int K);
        void sort_asc();
        void sort_desc();

        size_t num_independent_sets() const;
        size_t num_vertices() const;
    };

    std::ostream& operator<<(std::ostream& s, const KPartiteGraph& graph);





    struct Clique {
        DomainBox box;

        FloatT output;
        FloatT output_estimate;

        int indep_set; // the index of the independent set containing the next vertex to merge.
        int vertex;    // the index of the next vertex to merge from the `indep_set`

        bool operator<(const Clique& other) const;
        bool operator>(const Clique& other) const;
    };

    std::ostream& operator<<(std::ostream&s, const Clique& c);


    /*
    template <typename Cmp>
    class KPartiteGraphFind {
        const KPartiteGraph& graph_;

        // a priority queue containing all "partial" cliques (no max-cliques) that can still be expanded.
        std::vector<Clique> pq_buf_;
        std::vector<Clique> solutions_;
        Cmp cmp_;

    public:
        size_t nsteps;
        size_t nupdate_fails;
        size_t nrejected;

    private:
        Clique pq_pop();
        void pq_push(Clique&& c);

        bool is_solution(const Clique& c) const;
        bool update_clique(Clique& c);

    public:
        KPartiteGraphFind(KPartiteGraph& graph);

        bool step();
        bool steps(int nsteps);

        FloatT current_output_estimate() const;
        const std::vector<Clique>& solutions() const;
    };

    using MaxKPartiteGraphFind = KPartiteGraphFind<std::less<Clique>>;
    using MinKPartiteGraphFind = KPartiteGraphFind<std::greater<Clique>>;
    */

} /* namespace treeck */

#endif /* TREECK_GRAPH_H */
