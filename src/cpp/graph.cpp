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

        auto end = domains_.end();
        if (it1 != domains_.end()) // it0 must be end
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
        s << "DomainBox { ";
        for (auto&& [id, dom] : box)
            s << id << "->" << dom << " ";
        s << '}';
        return s;
    }

    Vertex::Vertex(DomainBox box, FloatT output)
        : box(box)
        , output(output)
        , min_output(output)
        , max_output(output) { }

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
        sets_.push_back({
            std::vector<Vertex>{{{}, 0.0}} // one dummy vertex
        });
    }

    KPartiteGraph::KPartiteGraph(const AddTree& addtree)
        : KPartiteGraph(addtree, [](FeatId fid) { return fid; })
    { }

    KPartiteGraph::KPartiteGraph(const AddTree& addtree, FeatIdMapper fmap)
    {
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
        auto f = [filter](const Vertex& v) { return filter(v.box); };

        for (auto it = sets_.begin(); it != sets_.end(); ++it)
        {
            auto& v = it->vertices;
            v.erase(std::remove_if(v.begin(), v.end(), f), v.end());
        }
    }

    std::tuple<FloatT, FloatT>
    KPartiteGraph::propagate_outputs()
    {
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
                        min0 = std::min(min0, v0.min_output);
                        max0 = std::max(max0, v0.max_output);
                    }
                }
                v1.min_output = min0 + v1.output;
                v1.max_output = max0 + v1.output;
            }
        }

        // output the min and max
        FloatT min0 = +std::numeric_limits<FloatT>::infinity();
        FloatT max0 = -std::numeric_limits<FloatT>::infinity();
        for (const auto& v0 : sets_.front().vertices)
        {
            min0 = std::min(min0, v0.min_output);
            max0 = std::max(max0, v0.max_output);
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

    std::ostream&
    operator<<(std::ostream& s, const KPartiteGraph& graph)
    {
        s << "KPartiteGraph {" << std::endl;
        for (auto& set : graph)
        {
            s << "  IndependentSet {" << std::endl;;
            for (auto& vertex : set.vertices)
            {
                s
                    << "    vertex("
                    << std::fixed
                    << std::setprecision(2)
                    << vertex.output
                    << ", " << vertex.min_output
                    << ", " << vertex.max_output
                    << ") "
                    << vertex.box << std::endl;
            }
            s << "  }" << std::endl;
        }
        s << "}";
        return s;
    }

    // - KPartiteGraphFind ----------------------------------------------------
    
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

    std::ostream&
    operator<<(std::ostream&s, const Clique& c)
    {
        std::cout
            << "Clique { " << std::endl
            << "    box=" << c.box << std::endl
            << "    output=" << c.output << std::endl
            << "    output_estimate=" << c.output_estimate << std::endl
            << "    indep_set=" << c.indep_set << std::endl
            << "    vertex=" << c.vertex << std::endl
            << " }";
        return s;
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
