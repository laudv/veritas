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

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

#include "graph.h"

namespace treeck {

    DomainBox::DomainBox() : domains_() { }

    std::vector<std::pair<int, Domain>>::const_iterator
    DomainBox::begin() const
    {
        return domains_.begin();
    }

    std::vector<std::pair<int, Domain>>::const_iterator
    DomainBox::end() const
    {
        return domains_.end();
    }

    std::vector<std::pair<int, Domain>>::const_iterator
    DomainBox::find(int id) const
    {
        return std::find_if(domains_.cbegin(), domains_.cend(), 
                [id](const std::pair<int, Domain>& arg) {
            return arg.first == id;
        });
    }

    std::vector<std::pair<int, Domain>>::iterator
    DomainBox::find(int id)
    {
        return std::find_if(domains_.begin(), domains_.end(), 
                [id](std::pair<int, Domain>& arg) {
            return arg.first == id;
        });
    }

    void
    DomainBox::refine(Split split, bool is_left_child, FeatIdMapper fmap)
    {
        visit_split(
                [this, &fmap, is_left_child](const LtSplit& s) {
                    int id = fmap(s.feat_id);
                    auto p = find(id);
                    if (p == end()) {
                        domains_.push_back({id, refine_domain({}, s, is_left_child)});
                    } else {
                        RealDomain dom = util::get_or<RealDomain>(p->second);
                        p->second = refine_domain(dom, s, is_left_child);
                    }
                },
                [this, &fmap, is_left_child](const BoolSplit& s) {
                    int id = fmap(s.feat_id);
                    auto p = find(id);
                    if (p == end()) {
                        domains_.push_back({id, refine_domain({}, s, is_left_child)});
                    } else {
                        BoolDomain dom = util::get_or<BoolDomain>(p->second);
                        p->second = refine_domain(dom, s, is_left_child);
                    }
               },
               split);
    }

    void
    DomainBox::sort()
    {
        std::sort(domains_.begin(), domains_.end(), [](auto& p, auto& q) {
            return p.first < q.first;
        });
    }

    bool
    DomainBox::overlaps(const DomainBox& other) const
    {
        //std::cout << "OVERLAPS" << std::endl;
        //std::cout << "  " << *this << std::endl;
        //std::cout << "  " << other << std::endl;

        auto it0 = domains_.begin();
        auto it1 = other.domains_.begin();
        
        // assume sorted
        while (it0 != domains_.end() && it1 != other.domains_.end())
        {
            //std::cout << "checking " << it0->first << ", " << it1->first << std::endl;
            if (it0->first == it1->first)
            {
                bool overlaps = visit_domain(
                    [it1](const RealDomain& dom0) {
                        auto dom1 = util::get_or<RealDomain>(it1->second);
                        return dom0.overlaps(dom1);
                    },
                    [it1](const BoolDomain& dom0) {
                        auto dom1 = util::get_or<BoolDomain>(it1->second);
                        return (dom0.value_ & dom1.value_) != 0;
                    },
                    it0->second);

                ++it0; ++it1;

                if (!overlaps)
                    return false;
            }
            else if (it0->first < it1->first) ++it0;
            else ++it1;
        }

        return true;
    }

    DomainBox
    DomainBox::combine(const DomainBox& other) const
    {
        DomainBox box;

        auto it0 = domains_.begin();
        auto it1 = other.domains_.begin();

        // assume sorted
        while (it0 != domains_.end() && it1 != other.domains_.end())
        {
            if (it0->first == it1->first)
            {
                visit_domain(
                    [&box, it1](const RealDomain& dom0) {
                        auto dom1 = util::get_or<RealDomain>(it1->second);
                        box.domains_.push_back({it1->first, dom0.intersect(dom1)});
                    },
                    [&box, it1](const BoolDomain& dom0) {
                        auto dom1 = util::get_or<BoolDomain>(it1->second);
                        box.domains_.push_back({it1->first, dom0.intersect(dom1)});
                    },
                    it0->second);

                ++it0; ++it1;
            }
            else if (it0->first < it1->first)
            {
                box.domains_.push_back(*it0); // copy
                ++it0;
            }
            else
            {
                box.domains_.push_back(*it1); // copy
                ++it1;
            }
        }

        // we're here because one (or both) of the iterators reached their end, but which one?
        auto end = domains_.end();
        if (it1 != other.domains_.end()) // it0 must be at its end
        {
            it0 = it1;
            end = other.domains_.end();
        }

        for (; it0 != end; ++it0)
        {
            box.domains_.push_back(*it0); // copy
        }

        return box;
    }

    std::ostream&
    operator<<(std::ostream& s, const DomainBox& box)
    {
        s << "DBox { ";
        for (auto&& [id, dom] : box)
            s << id << ":" << dom << " ";
        s << '}';
        return s;
    }

    Vertex::Vertex(DomainBox box, FloatT output)
        : box(box)
        , output(output)
        , min_bound(output)
        , max_bound(output) { }

    //bool
    //Vertex::operator<(const Vertex& other) const
    //{
    //    return min_output < other.min_output;
    //}

    //bool
    //Vertex::operator>(const Vertex& other) const
    //{
    //    return max_output > other.max_output;
    //}

    // - KPartiteGraph ---------------------------------------------------------

    KPartiteGraph::KPartiteGraph()
    {
        //sets_.push_back({
        //    std::vector<Vertex>{{{}, 0.0}} // one dummy vertex
        //});
    }

    KPartiteGraph::KPartiteGraph(const AddTree& addtree)
        : KPartiteGraph(addtree, [](FeatId fid) { return fid; })
    { }

    KPartiteGraph::KPartiteGraph(const AddTree& addtree, FeatIdMapper fmap)
    {
        if (addtree.base_score != 0.0)
        {
            //std::cout << "adding base_score set" << std::endl;
            IndependentSet set;
            set.vertices.push_back({{}, addtree.base_score});
            sets_.push_back(set);
        }

        for (const AddTree::TreeT& tree : addtree.trees())
        {
            IndependentSet set;
            fill_independence_set(set, tree.root(), fmap);

            sets_.push_back(set);
        }
    }


    std::vector<IndependentSet>::const_iterator
    KPartiteGraph::begin() const
    {
        return sets_.cbegin();
    }

    std::vector<IndependentSet>::const_iterator
    KPartiteGraph::end() const
    {
        return sets_.cend();
    }

    void
    KPartiteGraph::fill_independence_set(IndependentSet& set, AddTree::TreeT::CRef node,
            FeatIdMapper fmap)
    {
        if (node.is_internal())
        {
            fill_independence_set(set, node.left(), fmap);
            fill_independence_set(set, node.right(), fmap);
        }
        else
        {
            FloatT leaf_value = node.leaf_value();
            DomainBox box;

            while (!node.is_root())
            {
                auto child_node = node;
                bool is_left = child_node.is_left_child();
                node = node.parent();
                box.refine(node.get_split(), child_node.is_left_child(), fmap);
            }
            box.sort();
            set.vertices.push_back({box, leaf_value});
        }
    }

    void
    KPartiteGraph::prune(BoxFilter filter)
    {
        auto f = [filter](const Vertex& v) { return !filter(v.box); }; // keep if true, remove if false

        for (auto it = sets_.begin(); it != sets_.end(); ++it)
        {
            auto& v = it->vertices;
            v.erase(std::remove_if(v.begin(), v.end(), f), v.end());
        }
    }

    std::tuple<FloatT, FloatT>
    KPartiteGraph::propagate_outputs()
    {
        if (sets_.empty())
            return {0.0, 0.0};

        // dynamic programming algorithm from paper Chen et al. 2019
        for (auto it1 = sets_.rbegin() + 1; it1 != sets_.rend(); ++it1)
        {
            auto it0 = it1 - 1;
            for (auto& v1 : it1->vertices)
            {
                FloatT min0 = +std::numeric_limits<FloatT>::infinity();
                FloatT max0 = -std::numeric_limits<FloatT>::infinity();
                for (const auto& v0 : it0->vertices)
                {
                    if (v0.box.overlaps(v1.box))
                    {
                        min0 = std::min(min0, v0.min_bound);
                        max0 = std::max(max0, v0.max_bound);
                    }
                }
                v1.min_bound = min0 + v1.output;
                v1.max_bound = max0 + v1.output;
            }
        }

        // output the min and max
        FloatT min0 = +std::numeric_limits<FloatT>::infinity();
        FloatT max0 = -std::numeric_limits<FloatT>::infinity();
        for (const auto& v0 : sets_.front().vertices)
        {
            min0 = std::min(min0, v0.min_bound);
            max0 = std::max(max0, v0.max_bound);
        }

        return {min0, max0};
    }

    void
    KPartiteGraph::merge(int K)
    {
        std::vector<IndependentSet> new_sets;

        for (auto it = sets_.cbegin(); it != sets_.cend(); )
        {
            IndependentSet set0(*it++);
            IndependentSet set1;

            for (int k = 1; k < K && it != sets_.cend(); ++k, ++it)
            {
                for (const auto& v0 : set0.vertices)
                {
                    for (const auto& v1 : it->vertices)
                    {
                        if (v0.box.overlaps(v1.box))
                        {
                            auto box = v0.box.combine(v1.box);
                            FloatT output = v0.output + v1.output;
                            set1.vertices.push_back({box, output});
                        }
                    }
                }

                set0.vertices.clear();
                std::swap(set0, set1);
            }

            new_sets.push_back(std::move(set0));
        }

        std::swap(new_sets, sets_);
    }

    void
    KPartiteGraph::sort_asc()
    {
        for (auto& set : sets_)
        {
            std::sort(set.vertices.begin(), set.vertices.end(),
                [](const Vertex& a, const Vertex& b){
                    return a.output < b.output;
            });
        }
    }

    void
    KPartiteGraph::sort_desc()
    {
        for (auto& set : sets_)
        {
            std::sort(set.vertices.begin(), set.vertices.end(),
                [](const Vertex& a, const Vertex& b){
                    return a.output > b.output;
            });
        }
    }

    void
    KPartiteGraph::sort_bound_asc()
    {
        for (auto& set : sets_)
        {
            std::sort(set.vertices.begin(), set.vertices.end(),
                [](const Vertex& a, const Vertex& b){
                    return a.min_bound < b.min_bound;
            });
        }
    }

    void
    KPartiteGraph::sort_bound_desc()
    {
        for (auto& set : sets_)
        {
            std::sort(set.vertices.begin(), set.vertices.end(),
                [](const Vertex& a, const Vertex& b){
                    return a.max_bound > b.max_bound;
            });
        }
    }

    size_t
    KPartiteGraph::num_independent_sets() const
    {
        return sets_.size();
    }

    size_t
    KPartiteGraph::num_vertices() const
    {
        size_t result = 0;
        for (const auto& set : sets_)
            result += set.vertices.size();
        return result;
    }

    size_t
    KPartiteGraph::num_vertices_in_set(int indep_set) const
    {
        const auto &set = sets_.at(indep_set);
        return set.vertices.size();
    }

    std::ostream&
    operator<<(std::ostream& s, const KPartiteGraph& graph)
    {
        std::ios_base::fmtflags flgs(std::cout.flags());

        if (graph.num_independent_sets() == 0)
            return s << "KPartiteGraph { }";

        s << "KPartiteGraph {" << std::endl;
        for (auto& set : graph)
        {
            s << "  IndependentSet {" << std::endl;;
            for (auto& vertex : set.vertices)
            {
                s
                    << "    v("
                    << std::fixed
                    << std::setprecision(3)
                    << vertex.output
                    << "," << vertex.min_bound
                    << "," << vertex.max_bound
                    << ") ";
                s << vertex.box << std::endl;
            }
            s << "  }" << std::endl;
        }
        s << "}";
        std::cout.flags(flgs);
        return s;
    }

    // - KPartiteGraphOptimize ------------------------------------------------
    
    template <typename T> static T& get0(two_of<T>& t) { return std::get<0>(t); }
    template <typename T> static const T& get0(const two_of<T>& t) { return std::get<0>(t); }
    template <typename T> static T& get1(two_of<T>& t) { return std::get<1>(t); }
    template <typename T> static const T& get1(const two_of<T>& t) { return std::get<1>(t); }

    /*
    bool
    Clique::operator<(const Clique& other) const
    {
        return output_estimate < other.output_estimate;
    }

    bool
    Clique::operator>(const Clique& other) const
    {
        return output_estimate > other.output_estimate;
    }
    */

    bool
    CliqueMaxDiffPqCmp::operator()(const Clique& a, const Clique& b) const
    {
        int depth_a = get0(a.instance).indep_set + get1(a.instance).indep_set;
        int depth_b = get0(b.instance).indep_set + get1(b.instance).indep_set;
        
        FloatT diff_a = get1(a.instance).output_bound - get0(a.instance).output_bound;
        FloatT diff_b = get1(b.instance).output_bound - get0(b.instance).output_bound;

        // -- depth first with heuristic per indep.set
        //if (depth_a != depth_b)
        //    return depth_a < depth_b;

        //return diff_a < diff_b;

        // -- favor deeper stuff more, but keep 'error' within precision
        //FloatT precision = 0.0;

        //FloatT advantage_a = static_cast<FloatT>(depth_a > depth_b) * precision;
        //FloatT advantage_b = static_cast<FloatT>(depth_a < depth_b) * precision;

        //FloatT weight_a = diff_a + advantage_a;
        //FloatT weight_b = diff_b + advantage_b;

        // -- favor deeper stuff at most X%, step by step
        FloatT percentage = 0.0;
        
        FloatT advantage_a = 1.0 + depth_a * percentage;
        FloatT advantage_b = 1.0 + depth_b * percentage;

        FloatT weight_a = diff_a * advantage_a;
        FloatT weight_b = diff_b * advantage_b;

        return weight_a < weight_b;
    }

    std::ostream&
    operator<<(std::ostream& s, const CliqueInstance& ci)
    {
        return s
            << "    output=" << ci.output << ", bound=" << ci.output_bound << ", " << std::endl
            << "    indep_set=" << ci.indep_set << ", vertex=" << ci.vertex;
    }

    std::ostream&
    operator<<(std::ostream& s, const Clique& c)
    {
        FloatT diff = get1(c.instance).output_bound - get0(c.instance).output_bound;
        return s 
            << "Clique { " << std::endl
            << "  box=" << c.box << std::endl
            << "  instance0:" << std::endl << get0(c.instance) << std::endl
            << "  instance1:" << std::endl << get1(c.instance) << std::endl
            << "  bound_diff=" << diff << std::endl
            << '}';
    }

    std::ostream& operator<<(std::ostream& s, const Solution& sol)
    {
        return s
            << "Solution {" << std::endl
            << "  box=" << sol.box << std::endl
            << "  output0=" << sol.output0 << ", output1=" << sol.output1
            << " (diff=" << (sol.output1-sol.output0) << ')' << std::endl
            << '}';

    }

    static KPartiteGraph DUMMY_GRAPH{};

    KPartiteGraphOptimize::KPartiteGraphOptimize(KPartiteGraph& g0)
        : KPartiteGraphOptimize(g0, DUMMY_GRAPH) { }

    KPartiteGraphOptimize::KPartiteGraphOptimize(bool, KPartiteGraph& g1)
        : KPartiteGraphOptimize(DUMMY_GRAPH, g1) { }

    KPartiteGraphOptimize::KPartiteGraphOptimize(KPartiteGraph& g0, KPartiteGraph& g1)
        : graph_{g0, g1} // minimize g0, maximize g1
        , cliques_()
        , cmp_()
        , solutions()
        , nsteps{0, 0}
        , nupdate_fails{0}
        , nrejected{0}
        , nbox_filter_calls{0}
    {
        auto&& [output_bound0, max0] = g0.propagate_outputs(); // min output bound of first clique
        auto&& [min1, output_bound1] = g1.propagate_outputs(); // max output bound of first clique

        g0.sort_bound_asc(); // choose vertex with smaller `output` first
        g1.sort_bound_desc(); // choose vertex with larger `output` first

        bool unsat = std::isinf(output_bound0) || std::isinf(output_bound1); // some empty indep set
        if (!unsat && (g0.num_independent_sets() > 0 || g1.num_independent_sets() > 0))
        {
            cliques_.push_back({
                {}, // empty domain, ie no restrictions
                {
                    {0.0, output_bound0, 0, 0}, // output, bound, indep_set, vertex
                    {0.0, output_bound1, 0, 0}
                }
            });
        }
    }

    Clique 
    KPartiteGraphOptimize::pq_pop()
    {
        std::pop_heap(cliques_.begin(), cliques_.end(), cmp_);
        Clique c = std::move(cliques_.back());
        cliques_.pop_back();
        return c;
    }

    void
    KPartiteGraphOptimize::pq_push(Clique&& c)
    {
        cliques_.push_back(std::move(c));
        std::push_heap(cliques_.begin(), cliques_.end(), cmp_);
    }

    bool
    KPartiteGraphOptimize::is_solution(const Clique& c) const
    {
        return is_instance_solution<0>(c) && is_instance_solution<1>(c);
    }

    template <size_t instance>
    bool
    KPartiteGraphOptimize::is_instance_solution(const Clique& c) const
    {
        return std::get<instance>(c.instance).indep_set ==
            std::get<instance>(graph_).sets_.size();
    }

    template <size_t instance>
    bool
    KPartiteGraphOptimize::update_clique(Clique& c)
    {
        // Things to do:
        // 1. find next vertex in `indep_set`
        // 2. update max_bound (assume vertices in indep_set sorted)
        CliqueInstance& ci = std::get<instance>(c.instance);
        const KPartiteGraph& graph = std::get<instance>(graph_);

        const auto& set = graph.sets_[ci.indep_set].vertices;
        //std::cout << "UPDATE " << instance << ": " << ci.vertex << " -> " << set.size() << std::endl;
        for (int i = ci.vertex; i < set.size(); ++i) // (!) including ci.vertex!
        {
            const Vertex& v = set[i];
            //std::cout << "CHECK BOXES OVERLAP " << instance << " i=" << i << std::endl;
            //std::cout << "  " << c.box << std::endl;
            //std::cout << "  " << v.box << " -> " << c.box.overlaps(v.box) << std::endl << std::endl;
            if (c.box.overlaps(v.box))
            {
                // this is the next vertex to merge with in `indep_set`
                ci.vertex = i;

                // reuse dynamic programming value (propagate_outputs) to update bound
                FloatT prev_bound;
                if constexpr (instance==0)
                    prev_bound = v.min_bound;  // minimize instance 0
                else prev_bound = v.max_bound; // maximize instance 1

                // update bound
                ci.output_bound = prev_bound + ci.output;

                return true; // update successful!
            }
            else ++nupdate_fails;
        }
        return false; // out of compatible vertices in `indep_set`
    }

    template <size_t instance, typename BF, typename OF>
    void
    KPartiteGraphOptimize::step_instance(Clique c, BF box_filter, OF output_filter)
    {
        // invariant: Clique `c` can be extended
        //
        // Things to do (contd.):
        // 3. construct the new clique by merging another vertex from the selected graph
        // 4. [update new] check whether we can extend further later (all cliques in
        //    `cliques_` must be extendible)
        //    possible states are (solution=all trees contributed a value
        //                         extendible=valid `indep_set` & `vertex` values exist
        //      - extendible    extendible
        //      - solution      extendible
        //      - extendible    solution
        //    if the clique is not in any of the above states, do not add to `cliques_`
        // 5. [update old] update active instance of given clique `c`, add again if still extendible

        const KPartiteGraph& graph = std::get<instance>(graph_);
        CliqueInstance& ci = std::get<instance>(c.instance);

        const Vertex& v = graph.sets_[ci.indep_set].vertices.at(ci.vertex);
        Clique new_c = {
            c.box.combine(v.box), // box of the new clique
            c.instance // copy
        };

        CliqueInstance& new_ci = std::get<instance>(new_c.instance);
        new_ci.output += v.output; // output of clique is previous output + output of newly merged vertex
        new_ci.indep_set += 1;
        new_ci.vertex = 0; // start from beginning in new `indep_set` (= tree)
        ci.vertex += 1; // (!) mark the merged vertex as 'used' in old clique, so we dont merge it again later

        bool is_solution0 = is_instance_solution<0>(new_c);
        bool is_solution1 = is_instance_solution<1>(new_c);

        // check if this newly created clique is a solution
        if (is_solution0 && is_solution1)
        {
            bool is_valid_box = box_filter(new_c.box);
            bool is_valid_output = output_filter(
                    get0(new_c.instance).output,
                    get1(new_c.instance).output);

            ++nbox_filter_calls;

            if (is_valid_box && is_valid_output)
            {
                //std::cout << "SOLUTION (" << solutions.size() << "): " << new_c << std::endl;
                // push back to queue so it is 'extracted' when it actually is the optimal solution
                pq_push(std::move(new_c));
            }
            else
            {
                //std::cout << "discarding invalid solution" << std::endl;
                ++nrejected;
            }
        }
        else
        {
            // update both instances of `new_c`, the new box can change
            // `output_bound`s and `vertex`s values for both instances!
            // (update_clique updates `output_bound` and `vertex`)
            bool is_valid0 = is_solution0 || update_clique<0>(new_c);
            bool is_valid1 = is_solution1 || update_clique<1>(new_c);
            bool is_valid_box = box_filter(new_c.box);
            bool is_valid_output = output_filter(
                    get0(new_c.instance).output_bound,
                    get1(new_c.instance).output_bound);

            ++nbox_filter_calls;

            // there is a valid extension of this clique and it is not yet a solution -> push to `cliques_`
            if (is_valid0 && is_valid1 && is_valid_box && is_valid_output)
            {
                //std::cout << "push new: " << new_c << std::endl;
                pq_push(std::move(new_c));
            }
            else
            {
                //std::cout << "reject: " << is_valid0 << is_valid1
                //    << is_valid_box << is_valid_output << std::endl;
                ++nrejected;
            }
        }

        // push old clique if it still has a valid extension
        // we only need to update the current instance: box did not chane so
        // other instance cannot be affected
        if (update_clique<instance>(c))
        {
            // box did not change, so we don't have to check again (calling Z3 is expensive)
            if (output_filter(get0(c.instance).output_bound, get1(c.instance).output_bound))
            {
                //std::cout << "push old again: " << c << std::endl;
                pq_push(std::move(c));
            }
            else
            {
                //std::cout << "rejected old because of output " << c << std::endl;
                ++nrejected;
            }
        }
        else
        {
            //std::cout << "done with old: " << c << std::endl;
        }

        std::get<instance>(nsteps)++;


        //// Things to do (contd.):
        //// 2. find how we can update the given clique (update_clique)
        //// 2.1. if no update possible, then return false
        //// 2.2. if update possible, push self to pq, and goto 3
        //// 3. extend clique, and push to pq

        //// can we extend the given clique?
        //if (!update_clique<instance>(c))
        //    return false;


        //const KPartiteGraph& graph = std::get<instance>(graph_);
        //const CliqueInstance& ci = std::get<instance>(c.instance);

        //const Vertex& v = graph.sets_[ci.indep_set].vertices[ci.vertex];
        //Clique new_c = {
        //    c.box.combine(v.box), // box of the new clique
        //    c.instance // copy 
        //};

        //// update graph vertex info of the updated instance
        //CliqueInstance& new_ci = std::get<instance>(new_c.instance);
        //new_ci.output += v.output;
        //new_ci.output_bound = 

        //// push new clique
        //pq_push(std::move(new_c));

        //// push `old` clique again, maybe there is another interesting extension
        //pq_push(std::move(c));

        //return true;

        //// Things to do:
        //// 2. construct new clique
        //// 2.1. find new output_bound
        //// 2.2. determine index of next vertex in next indep_set
        //// 3. update the parent "clique"
        //// 3.1. if no more expansions possible, remove from pq
        //// 3.2. otherwise: update next vertex index
        //// 3.3. and update output_bound

        //short indep_set = std::get<instance>(c.indep_set);
        //int vertex = std::get<instance>(c.vertex);
        //const Vertex& v = std::get<instance>(graph_).sets_[indep_set].vertices[vertex];
        //FloatT output = std::get<instance>(c.output);
        //FloatT output_bound = std::get<instance>(c.output_bound);

        //// 1. construct new clique
        //two_of<FloatT> new_output = c.output;
        //std::get<instance>(new_output) += v.output;

        //two_of<short> new_indep_set = c.indep_set;
        //std::get<instance>(new_indep_set) += 1;

        //two_of<int> new_vertex = c.vertex;
        //std::get<instance>(new_vertex) = -1;

        //Clique new_c = {
        //    c.box.combine(v.box), // the new box
        //    new_output, // output of clique
        //    c.output_bound, // to be updated by `update_clique`
        //    new_indep_set, // we move one tree/indep.set further
        //    new_vertex // next vertex to merge, to be updated by `update_clique` (must be a valid index)
        //};

        //// 3.1 update the "parent" clique, if no more update, don't add to cliques_ again
        //if (update_clique<instance>(c))
        //{
        //    std::cout << "previous " << c << std::endl;
        //    pq_push(std::move(c));
        //}

        //// 2.1 check if new clique is a solution, if not, set `vertex` and `output_bound` values
        //if (is_solution(new_c))
        //{
        //    std::cout << "solution " << new_c << std::endl;
        //    //solutions_.push_back(std::move(new_c));
        //}
        //else if (update_clique<instance>(new_c))
        //{
        //    std::cout << "update " << new_c << std::endl;
        //    pq_push(std::move(new_c));
        //}

        ////std::cout << "STEP " << nsteps << " UPDATE " << old_est << " -> " << current_output_estimate()
        ////    << " (#pq=" << pq_buf_.size()
        ////    << ", #sol=" << solutions_.size() << ')'
        ////    << std::endl;

        //std::get<instance>(nsteps)++;
    }

    template <typename BF, typename OF>
    bool
    KPartiteGraphOptimize::step_aux(BF box_filter, OF output_filter)
    {
        if (cliques_.empty())
            return false;

        // Things to do:
        // 1. check whether the top of the pq is a solution
        // 2. determine which graph to use to extend the best clique
        // --> goto step_instance

        //std::cout << "ncliques=" << cliques_.size()
        //    << ", nsteps=" << get0(nsteps) << ":" << get1(nsteps)
        //    << ", nupdate_fails=" << nupdate_fails
        //    << ", nbox_filter_calls=" << nbox_filter_calls
        //    << std::endl;

        Clique c = pq_pop();

        bool is_solution0 = is_instance_solution<0>(c);
        bool is_solution1 = is_instance_solution<1>(c);

        //std::cout << "best clique " << c << std::endl;

        // move result to solutions if this is a solution
        if (is_solution0 && is_solution1)
        {
            //std::cout << "SOLUTION added "
            //    << get0(c.instance).output << ", "
            //    << get1(c.instance).output << std::endl;
            solutions.push_back({
                std::move(c.box),
                get0(c.instance).output,
                get1(c.instance).output
            });
        }
        // if not a solution, extend from graph0 first, then graph1, then graph0 again...
        else if (!is_solution0 && (get0(c.instance).indep_set <= get1(c.instance).indep_set
                    || is_solution1))
        {
            //std::cout << "step(): extend graph0" << std::endl;
            step_instance<0>(std::move(c), box_filter, output_filter);
        }
        else if (!is_solution1)
        {
            //std::cout << "step(): extend graph1" << std::endl;
            step_instance<1>(std::move(c), box_filter, output_filter);
        }
        else // there's a solution in `cliques_` -> shouldn't happen
        {
            throw std::runtime_error("invalid clique in cliques_: cannot be extended");
        }

        return true;
    }

    bool
    KPartiteGraphOptimize::step()
    {
        // accept everything
        return step_aux(
                [](const DomainBox& box) { return true; },
                [](FloatT output0, FloatT output1) { return true; });
    }

    bool
    KPartiteGraphOptimize::step(BoxFilter bf)
    {
        return step_aux(
                [&bf](const DomainBox& box) { return bf(box); },
                [](FloatT output0, FloatT output1) { return true; }); // accept all outputs
    }

    bool
    KPartiteGraphOptimize::step(BoxFilter bf, FloatT max_output0, FloatT min_output1)
    {
        return step_aux(
                [&bf](const DomainBox& box) { return bf(box); },
                [max_output0, min_output1](FloatT output0, FloatT output1) {
                    return output0 <= max_output0 && output1 >= min_output1;
                });
    }

    bool
    KPartiteGraphOptimize::step(BoxFilter bf, FloatT min_output_difference)
    {
        return step_aux(
                [&bf](const DomainBox& box) { return bf(box); },
                [min_output_difference](FloatT output0, FloatT output1) {
                    return (output1 - output0) > min_output_difference;
                });
    }

    bool
    KPartiteGraphOptimize::steps(int K)
    {
        for (int i=0; i<K; ++i)
            if (!step()) return false;
        return true;
    }

    bool
    KPartiteGraphOptimize::steps(int K, BoxFilter bf)
    {
        for (int i=0; i<K; ++i)
            if (!step(bf)) return false;
        return true;
    }

    bool
    KPartiteGraphOptimize::steps(int K, BoxFilter bf, FloatT max_output0, FloatT min_output1)
    {
        for (int i=0; i<K; ++i)
            if (!step(bf, max_output0, min_output1)) return false;
        //std::cout << "front: " << cliques_.front() << std::endl;
        return true;
    }

    bool
    KPartiteGraphOptimize::steps(int K, BoxFilter bf, FloatT min_output_difference)
    {
        for (int i=0; i<K; ++i)
        {
            if (!step(bf, min_output_difference)) return false;
        }
        return true;
    }

    two_of<FloatT>
    KPartiteGraphOptimize::current_bounds() const
    {
        if (cliques_.empty())
        {
            return {
                std::numeric_limits<FloatT>::infinity(),
                -std::numeric_limits<FloatT>::infinity()
            };
        }
        const Clique& c = cliques_.front();
        return {
            get0(c.instance).output_bound,
            get1(c.instance).output_bound
        };
    }

    size_t
    KPartiteGraphOptimize::num_candidate_cliques() const
    {
        return cliques_.size();
    }
    
    /*

    template <typename Cmp>
    KPartiteGraphFind<Cmp>::KPartiteGraphFind(KPartiteGraph& graph)
        : graph_(graph), nsteps(0), nupdate_fails(0), nrejected(0)
    {
        if constexpr (std::is_same_v<MaxKPartiteGraphFind, KPartiteGraphFind<Cmp>>)
            graph.sort_desc(); // try vertices with greater output values first
        else if constexpr (std::is_same_v<MinKPartiteGraphFind, KPartiteGraphFind<Cmp>>)
            graph.sort_asc(); // try vertices with smaller output values first
        else
            static_assert(util::always_false<Cmp>::value, "invalid Cmp type"); 

        auto&& [min, max] = graph.propagate_outputs();

        FloatT output_estimate = 0.0;
        if constexpr (std::is_same_v<MaxKPartiteGraphFind, KPartiteGraphFind<Cmp>>)
            output_estimate = max;
        else
            output_estimate = min;

        //if (graph.num_independent_sets() > 0)
        //{
        //    for (const auto& v0 : graph.sets_.back().vertices)
        //    {
        //        AnnotatedVertex v = {v0, merge_set, 0};
        //        if (is_expandable(v))
        //            pq_.push(std::move(v));
        //    }
        //}

        if (graph.num_independent_sets() > 0)
        {
            int indep_set = 0; // join into the first tree
            int vertex = 0; // the first vertex of the first indep.set

            // push a dummy AnnotatedVertex
            pq_push({
                {}, // empty domain
                0.0, // output
                output_estimate,
                indep_set,
                vertex
            });
        }

        std::cout << "THE GRAPH " << graph << std::endl;
    }

    template <typename Cmp>
    Clique
    KPartiteGraphFind<Cmp>::pq_pop()
    {
        std::pop_heap(pq_buf_.begin(), pq_buf_.end(), cmp_);
        Clique c = std::move(pq_buf_.back());
        pq_buf_.pop_back();
        return c;
    }

    template <typename Cmp>
    void
    KPartiteGraphFind<Cmp>::pq_push(Clique&& c)
    {
        pq_buf_.push_back(std::move(c));
        std::push_heap(pq_buf_.begin(), pq_buf_.end(), cmp_);
    }

    template <typename Cmp>
    bool
    KPartiteGraphFind<Cmp>::is_solution(const Clique& c) const
    {
        return c.indep_set == graph_.sets_.size();
    }

    template <typename Cmp>
    bool
    KPartiteGraphFind<Cmp>::update_clique(Clique& c)
    {
        // Things to do:
        // 1. find next vertex in `indep_set`
        // 2. update max_output (assume vertices in indep_set sorted)

        const auto& set = graph_.sets_[c.indep_set].vertices;
        for (int i = c.vertex + 1; i < set.size(); ++i)
        {
            const Vertex& v = set[i];
            if (c.box.overlaps(v.box))
            {
                c.vertex = i;
                if constexpr (std::is_same_v<MaxKPartiteGraphFind, KPartiteGraphFind<Cmp>>)
                {
                    //std::cout << "output update MAX: "
                    //    << c.output_estimate
                    //    << " -> "
                    //    << v.max_output + c.output
                    //    << std::endl;
                    c.output_estimate = v.max_output + c.output;
                }
                else
                {
                    //std::cout << "output update MIN: "
                    //    << c.output_estimate
                    //    << " -> "
                    //    << v.min_output + c.output
                    //    << std::endl;
                    c.output_estimate = v.min_output + c.output;
                }
                return true; // update successful!
            }
            else ++nupdate_fails;
        }
        return false; // out of compatible vertices in `c.indep_set`
    }

    template <typename Cmp>
    bool
    KPartiteGraphFind<Cmp>::step()
    {
        if (pq_buf_.empty())
            return false;

        // Things to do:
        // 1. construct new clique
        // 1.1. find new output_estimate
        // 1.2. determine index of next vertex in next indep_set
        // 2. update the parent "clique"
        // 2.1. if no more expansions possible, remove from pq
        // 2.2. otherwise: update next vertex index
        // 2.3. and update output_estimate
        
        FloatT old_est = current_output_estimate();

        Clique c = pq_pop();
        const Vertex& v = graph_.sets_[c.indep_set].vertices[c.vertex];

        // 1. construct new clique
        Clique new_c = {
            c.box.combine(v.box), // the new box
            c.output + v.output, // output of clique
            c.output_estimate, // to be updated by `update_clique`
            c.indep_set + 1, // we move one tree/indep.set further
            -1 // next vertex to merge, to be updated by `update_clique` (must be a valid index)
        };

        if (update_clique(c))
        {
            //std::cout << "previous " << c << std::endl;
            pq_push(std::move(c));
        }

        if (is_solution(new_c))
        {
            //std::cout << "solution " << new_c << std::endl;
            solutions_.push_back(std::move(new_c));
        }
        else if (update_clique(new_c))
        {
            //std::cout << "update " << new_c << std::endl;
            pq_push(std::move(new_c));
        }

        std::cout << "STEP " << nsteps << " UPDATE " << old_est << " -> " << current_output_estimate()
            << " (#pq=" << pq_buf_.size()
            << ", #sol=" << solutions_.size() << ')'
            << std::endl;

        ++nsteps;
        return true;
    }

    template <typename Cmp>
    bool
    KPartiteGraphFind<Cmp>::steps(int nsteps)
    {
        for (int i =0; i < nsteps; ++i)
            if (!step()) return false;
        return true;
    }

    template <typename Cmp>
    FloatT
    KPartiteGraphFind<Cmp>::current_output_estimate() const
    {
        if (pq_buf_.empty())
            return std::numeric_limits<FloatT>::quiet_NaN();
        return pq_buf_.front().output_estimate;
    }

    template <typename Cmp>
    const std::vector<Clique>&
    KPartiteGraphFind<Cmp>::solutions() const
    {
        return solutions_;
    }

    // manual template instantiations
    template class KPartiteGraphFind<std::greater<Clique>>;
    template class KPartiteGraphFind<std::less<Clique>>;

    */




} /* namespace treeck */
