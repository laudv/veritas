/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

#include "graph.h"

namespace treeck {

    DomainBox::DomainBox() : domains_() { }

    std::vector<std::pair<FeatId, Domain>>::const_iterator
    DomainBox::begin() const
    {
        return domains_.begin();
    }

    std::vector<std::pair<FeatId, Domain>>::const_iterator
    DomainBox::end() const
    {
        return domains_.end();
    }

    std::vector<std::pair<FeatId, Domain>>::const_iterator
    DomainBox::find(FeatId feat_id) const
    {
        return std::find_if(domains_.cbegin(), domains_.cend(), 
                [feat_id](const std::pair<FeatId, Domain>& arg) {
            return arg.first == feat_id;
        });
    }

    std::vector<std::pair<FeatId, Domain>>::iterator
    DomainBox::find(FeatId feat_id)
    {
        return std::find_if(domains_.begin(), domains_.end(), 
                [feat_id](std::pair<FeatId, Domain>& arg) {
            return arg.first == feat_id;
        });
    }

    void
    DomainBox::refine(Split split, bool is_left_child)
    {
        visit_split(
                [this, is_left_child](const LtSplit& s) {
                    auto p = find(s.feat_id);
                    if (p == end()) {
                        domains_.push_back({s.feat_id, refine_domain({}, s, is_left_child)});
                    } else {
                        RealDomain dom = util::get_or<RealDomain>(p->second);
                        p->second = refine_domain(dom, s, is_left_child);
                    }
                },
                [this, is_left_child](const BoolSplit& s) {
                    auto p = find(s.feat_id);
                    if (p == end()) {
                        domains_.push_back({s.feat_id, refine_domain({}, s, is_left_child)});
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
        for (auto&& [feat_id, dom] : box)
            s << feat_id << "->" << dom << " ";
        s << '}';
        return s;
    }

    Vertex::Vertex(DomainBox box, FloatT output)
        : box(box)
        , output(output)
        , min_output(output)
        , max_output(output) { }

    // -------------------------------------------------------------------------

    KPartiteGraph::KPartiteGraph(const AddTree& addtree)
    {
        for (const AddTree::TreeT& tree : addtree.trees())
        {
            IndependentSet set;
            fill_independence_set(set, tree.root());

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
    KPartiteGraph::fill_independence_set(IndependentSet& set, AddTree::TreeT::CRef node)
    {
        if (node.is_internal())
        {
            fill_independence_set(set, node.left());
            fill_independence_set(set, node.right());
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
                box.refine(node.get_split(), child_node.is_left_child());
            }
            box.sort();
            set.vertices.push_back({box, leaf_value});
        }
    }

    void
    KPartiteGraph::propage_outputs()
    {
        // dynamic programming algorithm from paper Chen et al. 2019
        for (auto it1 = sets_.begin() + 1; it1 != sets_.end(); ++it1)
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

                std::cout << "MIN: " << v1.min_output << " -> " << min0 + v1.output << std::endl;
                std::cout << "MAX: " << v1.max_output << " -> " << max0 + v1.output << std::endl;

                v1.min_output = min0 + v1.output;
                v1.max_output = max0 + v1.output;
            }
        }
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
                    << std::fixed << std::setw(6)
                    << std::setprecision(3)
                    << vertex.output
                    << ") "
                    << vertex.box << std::endl;
            }
            s << "  }" << std::endl;
        }
        s << "}";
        return s;
    }
}
