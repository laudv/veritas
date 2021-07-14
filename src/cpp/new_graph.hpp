/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_GRAPH_HPP
#define VERITAS_GRAPH_HPP

#include "new_tree.hpp"
#include "block_store.hpp"

#include <iomanip>
#include <chrono>

namespace veritas {

    struct MinMax {
        FloatT min;
        FloatT max;
    };

    /**
     * K-partite graph. Edges are implicit between vertices of different
     * independent sets when their boxes overlap
     */
    class Graph {

        struct Vertex { BoxRef box; FloatT output; };
        using IndepSet = std::vector<Vertex>; // independent set
        using DomainStore = BlockStore<DomainPair>;

        std::vector<IndepSet> sets_; // all indep sets in graph
        Box workspace_;

        void fill_indep_set(IndepSet& set, Tree::ConstRef node)
        {
            if (node.is_internal())
            {
                fill_indep_set(set, node.left());
                fill_indep_set(set, node.right());
            }
            else // 
            {
                FloatT leaf_value = node.leaf_value();
                while (!node.is_root())
                {
                    auto child_node = node;
                    node = node.parent();
                    refine_box(workspace_, node.get_split(), child_node.is_left_child());
                }
                BoxRef box = store.store(workspace_, remaining_mem_capacity());
                workspace_.clear();
                set.push_back({ box, leaf_value });
            }
        }

        IndepSet fill_indep_set(const Tree& tree)
        {
            IndepSet set;
            fill_indep_set(set, tree.root());
            return set;
        }

        size_t remaining_mem_capacity() const
        {
            return (size_t(1024)*1024*1024) - store.get_mem_size();
        }


    public:
        DomainStore store;

        Graph() {}
        Graph(const AddTree& at)
        {
            // pseudo vertex with empty box for the base_score
            sets_.push_back({ { BoxRef::null_box(), at.base_score } });
            for (const Tree& tree : at)
                sets_.push_back(fill_indep_set(tree));
        }

        std::vector<IndepSet>::const_iterator begin() const { return sets_.begin(); }
        std::vector<IndepSet>::const_iterator end() const { return sets_.end(); }

        template <typename F> /* F : (Box) -> bool */
        void prune(F f)
        {
            auto g = [f](const Vertex& v) { return f(v.box); };
            for (IndepSet& set : sets_)
                set.erase(std::remove_if(set.begin(), set.end(), g), set.end());
        }

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
                    new_vertex.box = new_store.store(workspace_, remaining_mem_capacity());
                    workspace_.clear();
                    new_set.push_back(std::move(new_vertex));
                }
                new_sets.push_back(std::move(new_set));
            }

            std::swap(store, new_store);
            std::swap(sets_, new_sets);
        }

        MinMax basic_bound() const
        {
            FloatT min_bound = 0.0, max_bound = 0.0;
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

        size_t num_independent_sets() const { return sets_.size(); }
        size_t num_vertices() const
        {
            size_t count = 0;
            for (const IndepSet& set : sets_) count += set.size();
            return count;
        }

        void add_with_negated_leaf_values(const Graph& rhs)
        {
            // CAREFUL: rhs.sets_ might be equal to this->sets_ -> don't use iterators
            size_t index = sets_.size();
            for (size_t i = 0; i < rhs.sets_.size(); ++i)
                sets_.push_back(rhs.sets_[i]); // copy!

            for (; index < sets_.size(); ++index)
                for (Vertex& v : sets_[index])
                    v = {v.box, -v.output};
        }

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
                                BoxRef box = store.store(workspace_, remaining_mem_capacity());
                                workspace_.clear();
                                FloatT output = v0.output + v1.output;
                                set1.push_back({box, output});
                            }
                        }
                    }

                    set0.clear();
                    std::swap(set0, set1);
                }

                //std::cout << "merge new_set of size " << set0.size() << std::endl;
                new_sets.push_back(std::move(set0));
            }

            std::swap(new_sets, sets_);
            return true;
        }
    };

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
