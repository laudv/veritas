/**
 * \file graph.hpp
 *
 * A k-partite graph derived from a tree ensemble.
 *
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_GRAPH_HPP
#define VERITAS_GRAPH_HPP

#include "tree.hpp"
#include "block_store.hpp"

#include <iomanip>
#include <chrono>
#include <sstream>

namespace veritas {

    struct Bounds {
        FloatT lo;
        FloatT up;
    };

    /**
     * K-partite graph. Edges are implicit between vertices of different
     * independent sets when their boxes overlap.
     *
     * GraphSearch uses a Graph internally. For normal use of Veritas, you do
     * not need to use this directly.
     */
    class Graph {
    public:
        struct Vertex {
            NodeId leaf_id; /**< Leaf ID corresponding to the vertex. */
            BoxRef box; /**< Box associated with the leaf. See Tree::compute_box(). */
            FloatT output; /**< The leaf value associated with the leaf. */
        };
        using IndepSet = std::vector<Vertex>; /**< independent set */
        using DomainStore = BlockStore<DomainPair>;

    private:

        DomainStore store_;

        std::vector<IndepSet> sets_; // all indep sets in graph
        Box workspace_;

        void fill_indep_set(int tree_index, IndepSet& set, Tree::ConstRef node)
        {
            if (node.is_internal())
            {
                fill_indep_set(tree_index, set, node.left());
                fill_indep_set(tree_index, set, node.right());
            }
            else // 
            {
                FloatT leaf_value = node.leaf_value();
                NodeId id = node.id();
                while (!node.is_root())
                {
                    auto child_node = node;
                    node = node.parent();
                    bool status = refine_box(workspace_, node.get_split(), child_node.is_left_child());
                    if (!status)
                    {
                        std::stringstream ss;
                        ss << "could not refine box of node " << id << " of tree " << tree_index
                           << ", refining " << workspace_ << " with split " <<
                               node.get_split() << " not possible";
                        throw std::runtime_error(ss.str());
                    }
                }
                BoxRef box = BoxRef(store_.store(workspace_, remaining_mem_capacity()));
                workspace_.clear();
                set.push_back({ id, box, leaf_value });
            }
        }

        IndepSet fill_indep_set(int tree_index, const Tree& tree)
        {
            IndepSet set;
            fill_indep_set(tree_index, set, tree.root());
            return set;
        }

        size_t remaining_mem_capacity() const
        {
            return (size_t(1024)*1024*1024) - store_.get_mem_size();
        }


    public:
        const FloatT base_score; /**< See AddTree::base_score */

        Graph() : base_score(0.0) {}
        Graph(const AddTree& at) : base_score(at.base_score)
        {
            int tree_index = 0;
            for (const Tree& tree : at)
                sets_.push_back(fill_indep_set(tree_index++, tree));
        }

        std::vector<IndepSet>::const_iterator begin() const { return sets_.begin(); }
        std::vector<IndepSet>::const_iterator end() const { return sets_.end(); }

        /**
         * Remove vertexes for which the `f` returns true.
         * Used by GraphSearch::prune().
         */
        template <typename F> /* F : (Box) -> bool */
        void prune(F f)
        {
            auto g = [f](const Vertex& v) { return f(v.box); };
            for (IndepSet& set : sets_)
                set.erase(std::remove_if(set.begin(), set.end(), g), set.end());
        }

        /**
         * Remove vertexes that do not overlap with the given box.
         *
         * If `intersect_with_box` is true, the boxes of the vertexes that
         * remain are updated to intersect with the given box.
         */
        void prune_by_box(const BoxRef& box, bool intersect_with_box)
        {
            prune([box](const BoxRef& b) { return !b.overlaps(box); });

            if (!intersect_with_box) return;

            // intersect domains in vertices with domains in box
            std::vector<IndepSet> new_sets;
            DomainStore new_store;

            for (const IndepSet& set : sets_)
            {
                IndepSet new_set;
                for (const Vertex& v : set)
                {
                    Vertex new_vertex(v);
                    // use copy_b argument set to false to avoid including irrelevant
                    // features from `box` into the new vertex's box --> !! order of arguments
                    combine_boxes(v.box, box, false, workspace_);
                    new_vertex.box = BoxRef(new_store.store(workspace_, remaining_mem_capacity()));
                    workspace_.clear();
                    new_set.push_back(std::move(new_vertex));
                }
                new_sets.push_back(std::move(new_set));
            }

            std::swap(store_, new_store);
            std::swap(sets_, new_sets);
        }

        /**
         * Simple output bound:
         *  - lo = sum{min{v.output : v in indep_set} : indep_set in graph}
         *  - hi = sum{max{v.output : v in indep_set} : indep_set in graph}
         */
        Bounds basic_bound() const
        {
            FloatT min_bound = base_score, max_bound = base_score;
            for (const auto& set : sets_)
            {
                FloatT min = +FLOATT_INF;
                FloatT max = -FLOATT_INF;
                for (const auto& v : set)
                {
                    min = std::min(min, v.output);
                    max = std::max(max, v.output);
                }
                min_bound += min;
                max_bound += max;
            }
            return {min_bound, max_bound};
        }

        /**
         * Looking at sets `start_indep_set` .. end, what is the maximum
         * possible sum of vertex output values given the box?
         *
         * return = sum{max{v.output : v in indep_set} : indep set in graph[start_indep_set..]}
         */
        FloatT basic_remaining_upbound(size_t start_indep_set, const BoxRef& box) const
        {
            FloatT res = 0.0;
            for (size_t indep_set = start_indep_set; indep_set < sets_.size(); ++indep_set)
            {
                FloatT max = -FLOATT_INF;
                for (const Vertex& v : sets_[indep_set])
                    if (v.box.overlaps(box))
                        max = std::max(max, v.output);
                res += max;
            }
            return res;
        }

        /** Get the vertexes in an independent set. */
        const IndepSet& get_vertices(size_t indep_set) const { return sets_.at(indep_set); }

        size_t num_independent_sets() const { return sets_.size(); }
        size_t num_vertices() const /** Total number of vertices. */
        {
            size_t count = 0;
            for (const IndepSet& set : sets_) count += set.size();
            return count;
        }

        /**
         * Merge algorithm due to 
         *
         *  > Hongge Chen, Huan Zhang, Duane Boning, and Cho-Jui Hsieh "Robust
         *  > Decision Trees Against Adversarial Examples", ICML 2019
         */
        bool merge(int K, float max_time)
        {
            using ms = std::chrono::milliseconds;
            using clock = std::chrono::steady_clock;

            max_time *= 1000; /* milliseconds */
            clock::time_point begin = clock::now();
            std::vector<IndepSet> new_sets;

            for (auto it = sets_.cbegin(); it != sets_.cend(); )
            {
                IndepSet set0(*it++);
                IndepSet set1;

                for (int k = 1; k < K && it != sets_.cend(); ++k, ++it)
                {
                    //std::cout << "merge " << set0.size() << " x " << it->size() << std::endl;
                    for (const auto& v0 : set0)
                    {
                        for (const auto& v1 : *it)
                        {
                            clock::time_point end = clock::now();
                            float dur = std::chrono::duration_cast<ms>(end - begin).count();
                            if (dur > max_time) { return false; }

                            if (v0.box.overlaps(v1.box))
                            {
                                //BoxRef box = store.combine_and_push(v0.box, v1.box, true);
                                combine_boxes(v0.box, v1.box, true, workspace_);
                                BoxRef box = BoxRef(store_.store(workspace_, remaining_mem_capacity()));
                                workspace_.clear();
                                FloatT output = v0.output + v1.output;
                                set1.push_back({-1, box, output}); // we loose leaf node ids
                            }
                        }
                    }

                    set0.clear();
                    std::swap(set0, set1);
                }

                std::cout << "merge new_set of size " << set0.size() << std::endl;
                new_sets.push_back(std::move(set0));
            }

            std::swap(new_sets, sets_);
            return true;
        }

    }; // class Graph

    std::ostream&
    operator<<(std::ostream& s, const Graph& graph)
    {
        std::ios_base::fmtflags flgs(std::cout.flags());

        if (graph.num_independent_sets() == 0)
            return s << "Graph { }";

        s << "Graph {" << std::endl;
        for (auto& set : graph)
        {
            s << "  IndependentSet {" << std::endl;;
            for (auto& vertex : set)
            {
                s
                    << "    v("
                    << std::fixed
                    << std::setprecision(3)
                    << vertex.output
                    << ") ";
                s << vertex.box << std::endl;
            }
            s << "  }" << std::endl;
        }
        s << "}";
        std::cout.flags(flgs);
        return s;
    }

} // namespace veritas


#endif // VERITAS_GRAPH_HPP
