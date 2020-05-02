/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include <algorithm>
#include <iostream>
#include <iomanip>
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

    std::ostream&
    operator<<(std::ostream& s, const DomainBox& box)
    {
        s << "DomainBox { ";
        for (auto&& [feat_id, dom] : box)
            s << feat_id << "->" << dom << " ";
        s << '}';
        return s;
    }

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
