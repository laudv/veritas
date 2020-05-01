/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include <iostream>
#include <vector>
#include "graph.h"

namespace treeck {

    DomainBox::DomainBox(std::vector<Domain>& data, size_t begin, size_t sz)
        : data_(data)
        , begin(begin)
        , sz(sz) { }

    void
    DomainBox::check_bounds(size_t i) const
    {
#ifndef DISABLE_BOUNDS_CHECKS
        if (i >= sz)
            throw std::out_of_range("DomainBox out of bounds");
#endif
    }

    const Domain&
    DomainBox::operator[](size_t i) const
    {
        check_bounds(i);
        return data_[begin + i];
    }

    Domain&
    DomainBox::operator[](size_t i)
    {
        check_bounds(i);
        return data_[begin + i];
    }

    void
    DomainBox::intersect(const DomainBox& other)
    {
        check_bounds(other.sz - 1);
        other.check_bounds(sz - 1);

        auto it1 = data_.begin() + begin;
        auto it2 = other.data_.cbegin() + other.begin;
        auto stop1 = it1 + sz;

        for(; it1 != stop1; ++it1, ++it2)
        {

        }
    }


    // -------------------------------------------------------------------------

    KPartiteGraph::KPartiteGraph()
    {
        std::cout << "hi from here" << std::endl;
    }

    size_t
    KPartiteGraph::map_feat_id(FeatId feat_id)
    {
        auto ptr = feat_id_map_.find(feat_id);
        if (ptr != feat_id_map_.end())
        {
            return ptr->second;
        }
        else
        {
            size_t mapped_value = feat_id_map_.size();
            feat_id_map_.insert({feat_id, mapped_value});
            return mapped_value;
        }
    }

    DomainBox
    KPartiteGraph::create_box()
    {
        size_t begin = domains_buffer_.size();
        size_t sz = nfeatures_;
        domains_buffer_.resize(begin + sz);

        return DomainBox(domains_buffer_, begin, sz);
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
            std::cout << "add node " << node.id() << std::endl;
            auto box = create_box(); // TODO probleem! we kennen nfeatures_ nog niet
            while (!node.is_root())
            {

            }
            //set.vertices_.push_back({});
        }
    }

    void
    KPartiteGraph::add_instance(const AddTree& addtree)
    {
        for (const AddTree::TreeT& tree : addtree.trees())
        {
            std::cout << tree << std::endl;
            IndependentSet set;
            fill_independence_set(set, tree.root());
        }
    }
}
