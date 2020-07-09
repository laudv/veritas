/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
 *
 * ----
 *
 * This file contains reimplemplementations of concepts introduced by the
 * following paper Chen et al. 2019
 *  - KPartiteGraph::merge (the core algorithm of the paper)
 *  - KPartiteGraph::propagate_outputs (dynamic programming output estimation agorithm)
 *
 * https://papers.nips.cc/paper/9399-robustness-verification-of-tree-based-models
 * https://github.com/chenhongge/treeVerification
*/

#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

#include "graph.h"

namespace treeck {

    FeatInfo::FeatInfo() : max_id_(-1), id_boundary_(0) {}

    FeatInfo::FeatInfo(
            const AddTree& at0,
            const AddTree& at1,
            const std::unordered_set<FeatId>& matches,
            bool match_is_reuse)
        : FeatInfo()
    {
        auto splits0 = at0.get_splits();
        auto splits1 = at1.get_splits();

        for (auto&& [feat_id, split_values] : splits0)
            feat_ids0_.push_back(feat_id);
        for (auto&& [feat_id, split_values] : splits1)
            feat_ids1_.push_back(feat_id);

        std::sort(feat_ids0_.begin(), feat_ids0_.end());
        std::sort(feat_ids1_.begin(), feat_ids1_.end());

        for (FeatId feat_id : feat_ids0_)
        {
            int key = feat_id;
            key2id_.emplace(key, ++max_id_);
        }

        id_boundary_ = max_id_+1;

        for (FeatId feat_id : feat_ids1_)
        {
            int key = ~static_cast<int>(feat_id);
            auto in_matches = matches.find(feat_id) != matches.end();
            if (in_matches == match_is_reuse)
            {
                auto lookup = key2id_.find(static_cast<int>(feat_id)); // feat_id == key of instance0
                if (lookup != key2id_.end())
                    key2id_.emplace(key, lookup->second);
                // if at0 and at1 are different, then it is possible that we haven't seen feat_id yet
                // if so, create a new id
                else
                    key2id_.emplace(key, ++max_id_);
            }
            else
            {
                key2id_.emplace(key, ++max_id_);
            }
        }

        // check types
        is_real_.resize(max_id_ + 1, false);
        for (FeatId feat_id : feat_ids0_)
        {
            auto split_values = splits0.find(feat_id);
            if (split_values == splits0.end())
                throw std::runtime_error("FeatInfo: invalid state 1");

            int key = feat_id;
            auto lookup = key2id_.find(key);
            if (lookup == key2id_.end())
                throw std::runtime_error("FeatInfo: invalid state 2");

            int id = lookup->second;
            if (split_values->second.size() != 0) // at least one split value => real split
                is_real_[id] = true;
        }
        for (FeatId feat_id : feat_ids1_)
        {
            auto split_values = splits1.find(feat_id);
            if (split_values == splits1.end())
                throw std::runtime_error("FeatInfo: invalid state 3");

            int key = ~static_cast<int>(feat_id);
            auto lookup = key2id_.find(key);
            if (lookup == key2id_.end())
                throw std::runtime_error("FeatInfo: invalid state 4");

            int id = lookup->second;
            if (split_values->second.size() != 0) // at least one split value => real split
                is_real_[id] = true;
        }
    }

    int
    FeatInfo::get_max_id() const
    {
        return max_id_;
    }

    size_t
    FeatInfo::num_ids() const
    {
        return static_cast<size_t>(get_max_id() + 1);
    }

    int
    FeatInfo::get_id(int instance, FeatId feat_id) const
    {
        int key = feat_id;
        if (instance != 0) key = ~key;

        auto lookup = key2id_.find(key);
        if (lookup != key2id_.end())
            return lookup->second;
        return UNUSED_ID;
    }

    bool 
    FeatInfo::is_instance0_id(int id) const
    {
        return id < id_boundary_;
    }

    bool
    FeatInfo::is_real(int id) const
    {
        return is_real_.at(id);
    }

    const std::vector<FeatId>&
    FeatInfo::feat_ids0() const
    {
        return feat_ids0_;
    }

    const std::vector<FeatId>&
    FeatInfo::feat_ids1() const
    {
        return feat_ids1_;
    }



    const size_t DOMAIN_STORE_MAX_MEM = 4294967296; // 4GB

    DomainStore::DomainStore()
        : store_{}, workspace_{}, max_mem_size_(DOMAIN_STORE_MAX_MEM)
    {
        const size_t DEFAULT_SIZE = 1024*1024 / sizeof(Block::value_type); // 1mb of domains
        Block block;
        block.reserve(DEFAULT_SIZE);
        store_.push_back(std::move(block));
    }

    DomainStore::Block&
    DomainStore::get_block_with_capacity(size_t cap)
    {
        // allocate a new block if necessary
        Block& block = store_.back();
        if (block.capacity() - block.size() < cap)
        {
            size_t mem = get_mem_size();
            size_t rem_capacity = (max_mem_size_ - mem) / sizeof(Block::value_type);
            if (rem_capacity > block.capacity() * 2) // double size of blocks each time,..
                rem_capacity = block.capacity() * 2; // .. unless memory limit almost reached
            else if (rem_capacity > 0)
                std::cerr << "WARNING: almost running out of memory, "
                    << static_cast<double>(rem_capacity * sizeof(Block::value_type)) / (1024.0*1024.0)
                    << " mb out of "
                    << static_cast<double>(max_mem_size_) / (1024.0*1024.0)
                    << " mb left" << std::endl;
            else
                throw std::runtime_error("DomainStore: out of memory");

            Block new_block;
            new_block.reserve(rem_capacity);
            store_.push_back(std::move(new_block));

            //std::cout << "DomainStore memory: " << mem << " bytes, "
            //    << (static_cast<double>(get_mem_size()) / (1024.0 * 1024.0)) << " mb ("
            //    << store_.size() << " blocks)" << std::endl;
        }

        return store_.back();
    }

    size_t
    DomainStore::get_mem_size() const
    {
        size_t mem = 0;
        for (const Block& b : store_)
            mem += b.capacity() * sizeof(Block::value_type);
        return mem;
    }

    void
    DomainStore::set_max_mem_size(size_t mem)
    {
        max_mem_size_ = mem;
    }

    size_t
    DomainStore::get_max_mem_size() const
    {
        return max_mem_size_;
    }

    void
    DomainStore::refine_workspace(Split split, bool is_left_child, FeatIdMapper fmap)
    {
        int id = visit_split(
                [&fmap](const LtSplit& s) { return fmap(s.feat_id); },
                [&fmap](const BoolSplit& bs) { return fmap(bs.feat_id); },
                split);
        DomainT dom;

        auto it = std::find_if(workspace_.begin(), workspace_.end(),
                [id](const DomainPair& p) { return p.first == id; });

        if (it != workspace_.end())
            dom = it->second;

        dom = visit_split(
                [=](const LtSplit& s) { return refine_domain(dom, s, is_left_child); },
                [=](const BoolSplit& bs) {
                    LtSplit s{bs.feat_id, 1.0};
                    return refine_domain(dom, s, is_left_child);
                },
                split);
        if (it == workspace_.end())
        {
            workspace_.push_back({ id, dom });
            for (int i = workspace_.size() - 1; i > 0; --i) // ids sorted
                if (workspace_[i-1].first > workspace_[i].first)
                    std::swap(workspace_[i-1], workspace_[i]);
            //std::sort(workspace_.begin(), workspace_.end(),
            //        [](const DomainPair& a, const DomainPair& b) {
            //            return a.first < b.first;
            //        })
        }
        else
        {
            it->second = dom;
        }
    }

    DomainBox
    DomainStore::get_workspace_box() const
    {
        size_t len = workspace_.size();
        if (len > 0)
        {
            const DomainPair *front = &workspace_[0];
            return { front, front + len };
        }
        else return DomainBox::null_box();
    }

    DomainBox
    DomainStore::push_workspace()
    {
        // this store_ block has enough space to accomodate the workspace DomainBox
        DomainBox workspace = get_workspace_box();
        Block& block = get_block_with_capacity(workspace.size());

        // push a copy of the workspace DomainBox
        size_t start_index = block.size();
        for (auto&& [id, domain] : workspace)
            block.push_back({ id, domain });

        DomainPair *ptr = &block[start_index];
        DomainBox box = { ptr, ptr + workspace_.size() };

        workspace_.clear();

        return box;
    }

    void
    DomainStore::combine_in_workspace(const DomainBox& a, const DomainBox& b)
    {
        if (!workspace_.empty())
            throw std::runtime_error("workspace not empty");

        const DomainPair *it0 = a.begin();
        const DomainPair *it1 = b.begin();

        // assume sorted
        while (it0 != a.end() && it1 != b.end())
        {
            if (it0->first == it1->first)
            {
                DomainT dom = it0->second.intersect(it1->second);
                workspace_.push_back({ it0->first, dom });
                ++it0; ++it1;
            }
            else if (it0->first < it1->first)
            {
                workspace_.push_back(*it0); // copy
                ++it0;
            }
            else
            {
                workspace_.push_back(*it1); // copy
                ++it1;
            }
        }

        // push all remaining items (one of them is already at the end, no need to compare anymore)
        for (; it0 != a.end(); ++it0)
            workspace_.push_back(*it0); // copy
        for (; it1 != b.end(); ++it1)
            workspace_.push_back(*it1); // copy
    }

    DomainBox
    DomainStore::combine_and_push(const DomainBox& a, const DomainBox& b)
    {
        combine_in_workspace(a, b);
        return push_workspace();
    }

    void
    DomainStore::clear_workspace()
    {
        workspace_.clear();
    }

    //void
    //DomainStore::push_prototype_box(const FeatInfo& finfo)
    //{
    //    if (box_size_ > 0)
    //        throw std::runtime_error("prototype already pushed");
    //    box_size_ = finfo.num_ids();

    //    Block& block = get_last_block();

    //    // push domains for instance0
    //    for (FeatId feat_id : finfo.feat_ids0())
    //    {
    //        int id = finfo.get_id(0, feat_id);
    //        if (id != static_cast<int>(block.size()))
    //            throw std::runtime_error("invalid state");

    //        if (finfo.is_real(id)) block.push_back({});
    //        else                   block.push_back(BOOL_DOMAIN);
    //    }

    //    // push domains for instance1
    //    for (FeatId feat_id : finfo.feat_ids1())
    //    {
    //        int id = finfo.get_id(1, feat_id);
    //        if (finfo.is_instance0_id(id)) continue;
    //        if (id != static_cast<int>(block.size()))
    //            throw std::runtime_error("invalid state");

    //        if (finfo.is_real(id)) block.push_back({});
    //        else                   block.push_back(BOOL_DOMAIN);
    //    }
    //}

    //DomainBox
    //DomainStore::push_box()
    //{
    //    // copy the prototype domain
    //    DomainT *ptr = &store_[0][0];
    //    return push_copy({ptr, ptr + box_size_});
    //}

    //DomainBox
    //DomainStore::push_copy(const DomainBox& other)
    //{
    //    if (box_size_ == 0) throw std::runtime_error("zero-sized prototype box");
    //    if (other.size() != box_size_) throw std::runtime_error("incompatible boxes");

    //    Block& block = get_last_block();
    //    size_t start_index = block.size();

    //    for (const DomainT& d : other)
    //        block.push_back(d);

    //    if (block.size() != start_index+box_size_) throw std::runtime_error("invalid state");

    //    DomainT *ptr = &block[start_index];
    //    return { ptr, ptr + box_size_ };
    //}

    //void
    //DomainStore::pop_last_box(const DomainBox& last_box)
    //{
    //    Block& block = store_.back(); // don't need get_last_block, because no resize needed!
    //    size_t start_index = block.size() - box_size_;
    //    if (last_box.begin() != &block[start_index])
    //        throw std::runtime_error("invalid call to pop_last_box");
    //    block.resize(start_index);
    //}




    DomainBox::DomainBox(const DomainPair *b, const DomainPair *e) : begin_(b), end_(e) { }

    DomainBox
    DomainBox::null_box() {
        return DomainBox(nullptr, nullptr);
    }

    size_t
    DomainBox::size() const
    {
        return end_ - begin_;
    }

    //bool
    //DomainBox::is_right_neighbor(const DomainBox& other) const
    //{
    //    auto it0 = begin_;
    //    auto it1 = other.begin_;
    //    
    //    for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
    //    {
    //        const RealDomain& a = *it0;
    //        const RealDomain& b = *it1;
    //        if (a == b)
    //            continue;
    //        if (a.hi != b.lo) // discontinuity
    //            return false;
    //    }

    //    return true;
    //}

    //void
    //DomainBox::join_right_neighbor(const DomainBox& other)
    //{
    //    auto it0 = begin_;
    //    auto it1 = other.begin_;
    //    
    //    for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
    //    {
    //        RealDomain& a = *it0;
    //        const RealDomain& b = *it1;
    //        if (a == b)
    //            continue;
    //        if (a.hi != b.lo) // discontinuity
    //            throw std::runtime_error("not a right neighbor");
    //        a.hi = b.hi;
    //    }
    //}

    //void
    //DomainBox::refine(Split split, bool is_left_child, FeatIdMapper fmap)
    //{
    //    visit_split(
    //            [this, &fmap, is_left_child](const LtSplit& s) {
    //                int id = fmap(s.feat_id);
    //                DomainT *ptr = begin_ + id;
    //                //RealDomain dom = util::get_or<RealDomain>(*ptr);
    //                *ptr = refine_domain(*ptr, s, is_left_child);
    //            },
    //            [this, &fmap, is_left_child](const BoolSplit& bs) {
    //                int id = fmap(bs.feat_id);
    //                DomainT *ptr = begin_ + id;
    //                LtSplit s{bs.feat_id, 1.0};
    //                //BoolDomain dom = util::get_or<BoolDomain>(*ptr);
    //                *ptr = refine_domain(*ptr, s, is_left_child);
    //            },
    //            split);
    //}

    bool
    DomainBox::overlaps(const DomainBox& other) const
    {
        auto it0 = begin_;
        auto it1 = other.begin_;

        while (it0 != end_ && it1 != other.end_)
        {
            if (it0->first == it1->first)
            {
                if (!it0->second.overlaps(it1->second))
                    return false;
                ++it0; ++it1;
            }
            else if (it0->first < it1->first) ++it0;
            else ++it1;
        }

        return true;
    }

    //bool
    //DomainBox::covers(const DomainBox& other) const
    //{
    //    auto it0 = begin_;
    //    auto it1 = other.begin_;
    //    
    //    for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
    //    {
    //        if (!it0->covers(*it1))
    //            return false;
    //    }

    //    return true;
    //}

    //void
    //DomainBox::combine(const DomainBox& other)
    //{
    //    DomainT *it0 = begin_;
    //    DomainT *it1 = other.begin_;

    //    for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
    //    {
    //        *it0 = it0->intersect(*it1);
    //        //visit_domain(
    //        //    [it0](const RealDomain& dom1) {
    //        //        auto dom0 = util::get_or<RealDomain>(*it0);
    //        //        *it0 = dom0.intersect(dom1);
    //        //    },
    //        //    [it0](const BoolDomain& dom1) {
    //        //        auto dom0 = util::get_or<BoolDomain>(*it0);
    //        //        *it0 = dom0.intersect(dom1);
    //        //    },
    //        //    *it1);
    //    }
    //}

    //void
    //DomainBox::copy(const DomainBox& other)
    //{
    //    DomainT *it0 = begin_;
    //    DomainT *it1 = other.begin_;

    //    for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
    //        *it0 = *it1;
    //}

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

    KPartiteGraph::KPartiteGraph(const AddTree& addtree, FeatIdMapper fmap)
    {
        if (addtree.base_score != 0.0)
        {
            //std::cout << "adding base_score set" << std::endl;
            IndependentSet set;
            set.vertices.push_back({ DomainBox::null_box(), addtree.base_score });
            sets_.push_back(set);
        }

        for (const AddTree::TreeT& tree : addtree.trees())
        {
            IndependentSet set;
            fill_independence_set(set, tree.root(), fmap);
            sets_.push_back(set);
        }
    }

    KPartiteGraph::KPartiteGraph(const AddTree& addtree,
            const FeatInfo& finfo, int instance)
        : KPartiteGraph(addtree, [=](FeatId feat_id) { return finfo.get_id(instance, feat_id); })
    { }


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
    KPartiteGraph::fill_independence_set(IndependentSet& set,
            AddTree::TreeT::CRef node,
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
            while (!node.is_root())
            {
                auto child_node = node;
                node = node.parent();
                store_.refine_workspace(node.get_split(), child_node.is_left_child(), fmap);
            }
            DomainBox box = store_.push_workspace();
            set.vertices.push_back({ box, leaf_value });
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

    std::tuple<FloatT, FloatT>
    KPartiteGraph::basic_bound() const
    {
        if (sets_.empty())
            return {0.0, 0.0};
        FloatT min_bound = 0.0, max_bound = 0.0;
        for (const auto& set : sets_)
        {
            FloatT min = +std::numeric_limits<FloatT>::infinity();
            FloatT max = -std::numeric_limits<FloatT>::infinity();
            for (const auto& v : set.vertices)
            {
                min = std::min(min, v.output);
                max = std::max(max, v.output);
            }
            min_bound += min;
            max_bound += max;
        }
        return {min_bound, max_bound};
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
                //std::cout << "merge " << set0.vertices.size() << " x " << it->vertices.size() << std::endl;
                for (const auto& v0 : set0.vertices)
                {
                    for (const auto& v1 : it->vertices)
                    {
                        if (v0.box.overlaps(v1.box))
                        {
                            DomainBox box = store_.combine_and_push(v0.box, v1.box);
                            FloatT output = v0.output + v1.output;
                            set1.vertices.push_back({box, output});
                        }
                    }
                }

                set0.vertices.clear();
                std::swap(set0, set1);
            }

            //std::cout << "merge new_set of size " << set0.vertices.size() << std::endl;
            new_sets.push_back(std::move(set0));
        }

        std::swap(new_sets, sets_);
    }

    //void
    //KPartiteGraph::simplify(FloatT max_err, bool overestimate)
    //{
    //    struct E { size_t set; DomainBox box; FloatT abs_err; };
    //    std::vector<E> errors;

    //    while (true) {
    //        FloatT min_err = max_err; // anything larger is skipped
    //        size_t min_set = sets_.size(); // invalid
    //        size_t min_vertex = 0;

    //        for (size_t j = 0; j < sets_.size(); ++j)
    //        {
    //            const IndependentSet& set = sets_[j];
    //            for (size_t i = 1; i < set.vertices.size(); ++i)
    //            {
    //                const Vertex& v0 = set.vertices[i-1];
    //                const Vertex& v1 = set.vertices[i];
    //                if (v0.box.is_right_neighbor(v1.box))
    //                {
    //                    FloatT err = std::abs(v0.output - v1.output);
    //                    if (err <= min_err)
    //                    {
    //                        for (E e : errors)
    //                        {
    //                           if (v0.box.overlaps(e.box) || v1.box.overlaps(e.box))
    //                                err += e.abs_err;
    //                        }
    //                        if (err <= min_err)
    //                        {
    //                            min_err = err;
    //                            min_set = j;
    //                            min_vertex = i;
    //                        }
    //                    }
    //                }
    //            }
    //        }

    //        // nothing found, everything larger than max_err, we're done
    //        if (min_set == sets_.size())
    //            break;

    //        // merge the two neighboring vertices with the smallest error
    //        //std::cout << "simplify " << min_set << ", " << min_vertex << " with error " << min_err << std::endl;

    //        IndependentSet& set = sets_[min_set];
    //        Vertex& v0 = set.vertices[min_vertex-1];
    //        const Vertex& v1 = set.vertices[min_vertex];
    //        FloatT err = std::abs(v0.output - v1.output); // measure before changing v0

    //        //std::cout << "is_right_neighbor " << v0.box.is_right_neighbor(v1.box) << std::endl;

    //        // update v0
    //        //std::cout << "before " << v0.box << ", " << v0.output << std::endl;
    //        //std::cout << "    +  " << v1.box << ", " << v0.output << std::endl;
    //        v0.box.join_right_neighbor(v1.box);
    //        //std::cout << "after " << v0.box << std::endl;

    //        v0.output = overestimate ? std::max(v0.output, v1.output) : std::min(v0.output, v1.output);
    //        v0.max_bound = v0.output;
    //        v0.min_bound = v0.output;

    //        //std::cout << "new vertex output " << v0.output << std::endl;

    //        // remove v1
    //        for (size_t i = min_vertex + 1; i < set.vertices.size(); ++i)
    //            set.vertices[i-1] = set.vertices[i];
    //        set.vertices.pop_back();

    //        // store error of region
    //        errors.push_back({min_set, v0.box, err});


    //        //for (auto e : errors)
    //        //    std::cout << "- error: " << e.abs_err << " for " << e.box << std::endl;
    //    }
    //}

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
    
    template <typename T> inline T& get0(two_of<T>& t) { return std::get<0>(t); }
    template <typename T> inline const T& get0(const two_of<T>& t) { return std::get<0>(t); }
    template <typename T> inline T& get1(two_of<T>& t) { return std::get<1>(t); }
    template <typename T> inline const T& get1(const two_of<T>& t) { return std::get<1>(t); }

    std::ostream&
    operator<<(std::ostream& s, const CliqueInstance& ci)
    {
        return s
            << "    output=" << ci.output << ", heuristic=" << ci.heuristic
            << ", bound=" << ci.output_bound() << ", " << std::endl
            << "    indep_set=" << ci.indep_set << ", vertex=" << ci.vertex;
    }

    std::ostream&
    operator<<(std::ostream& s, const Clique& c)
    {
        FloatT diff = get1(c.instance).output_bound() - get0(c.instance).output_bound();
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

    KPartiteGraphOptimize::KPartiteGraphOptimize(KPartiteGraph& g0, KPartiteGraph& g1)
        : graph_{g0, g1} // minimize g0, maximize g1
        , cliques_()
        , cmp_{1.0}
        , heuristic_type(KPartiteGraphOptimize::RECOMPUTE)
        , num_steps{0, 0}
        , num_update_fails{0}
        , num_rejected{0}
        , num_box_checks{0}
        , solutions{}
        , start_time{0.0}
    {
        auto&& [output_bound0, max0] = g0.propagate_outputs(); // min output bound of first clique
        auto&& [min1, output_bound1] = g1.propagate_outputs(); // max output bound of first clique

        g0.sort_bound_asc(); // choose vertex with smaller `output` first
        g1.sort_bound_desc(); // choose vertex with larger `output` first

        bool unsat = std::isinf(output_bound0) || std::isinf(output_bound1); // some empty indep set
        if (!unsat && (g0.num_independent_sets() > 0 || g1.num_independent_sets() > 0))
        {
            cliques_.push_back({
                DomainBox::null_box(), // empty domain, ie no restrictions
                {
                    {0.0, output_bound0, 0, 0}, // output, bound, indep_set, vertex
                    {0.0, output_bound1, 0, 0}
                }
            });
        }

        start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count() * 1e-3;
    }

    KPartiteGraphOptimize::KPartiteGraphOptimize(
            const KPartiteGraphOptimize& other, size_t i, size_t K)
        : graph_{other.graph_}
        , cliques_{}
        , cmp_{other.cmp_.eps}
        , heuristic_type(other.heuristic_type)
        , num_steps{0, 0}
        , num_update_fails{0}
        , num_rejected{0}
        , num_box_checks{0}
        , solutions{}
        , start_time{other.start_time}
    {
        for (size_t j = i; j < other.cliques_.size(); j += K)
            cliques_.push_back(other.cliques_[j]); // copy
        std::make_heap(cliques_.begin(), cliques_.end(), cmp_);
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

    FloatT
    KPartiteGraphOptimize::get_eps() const
    {
        return cmp_.eps;
    }

    static std::mutex stdout_mutex;

    void
    KPartiteGraphOptimize::set_eps(FloatT eps, bool rebuild_heap)
    {
        // we maximize diff, so h(x) must be underestimated to make the search prefer deeper solutions
        if (eps <= 0.0) throw std::runtime_error("nonsense eps");
        if (eps > 1.0) throw std::runtime_error("nonsense eps");

        if (eps != cmp_.eps)
        {
            // re-add the best solution for the current eps so we can find it again with a better eps
            const Solution *best_sol = nullptr;
            for (int j = solutions.size() - 1; j >= 0; --j)
            {
                const Solution& sol = solutions[j];
                //cliques_.push_back({ // add all solutions again, probably a waste
                //    sol.box,
                //    {
                //        {
                //            sol.output0,
                //            0.0, // no heuristic value, we're in a goal state
                //            static_cast<short>(get0(graph_).sets_.size()), // indep set
                //            0, // vertex
                //        },
                //        {
                //            sol.output1,
                //            0.0, // no heuristic value, we're in a goal state
                //            static_cast<short>(get1(graph_).sets_.size()), // indep set
                //            0, // vertex
                //        }
                //    }
                //});
                if (sol.eps != cmp_.eps)
                    break; // we're done, these are older solutions with lower epses
                if (best_sol == nullptr || sol.output_difference() > best_sol->output_difference())
                    best_sol = &sol;
            }
            if (best_sol != nullptr)
            {
                //{
                //    std::lock_guard g(stdout_mutex);
                //    std::cout << "re-adding " << best_sol->output_difference()
                //        << ", is_valid=" << best_sol->is_valid
                //        << ", eps=" << best_sol->eps
                //        << std::endl;
                //}
                cliques_.push_back({
                    best_sol->box,
                    {
                        {
                            best_sol->output0,
                            0.0, // no heuristic value, we're in a goal state
                            static_cast<short>(get0(graph_).sets_.size()), // indep set
                            0, // vertex
                        },
                        {
                            best_sol->output1,
                            0.0, // no heuristic value, we're in a goal state
                            static_cast<short>(get1(graph_).sets_.size()), // indep set
                            0, // vertex
                        }
                    }
                });
            }
            cmp_.eps = eps;
            if (rebuild_heap)
                std::make_heap(cliques_.begin(), cliques_.end(), cmp_);
        }
        //std::cout << "ARA* EPS set to " << cmp_.eps << std::endl;
    }

    void
    KPartiteGraphOptimize::use_dyn_prog_heuristic()
    {
        if (get0(num_steps) + get1(num_steps) != 0)
            throw std::runtime_error("cannot change heuristic mid optimization");
        heuristic_type = DYN_PROG;
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
            static_cast<int>(std::get<instance>(graph_).sets_.size());
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
        for (size_t i = ci.vertex; i < set.size(); ++i) // (!) including ci.vertex!
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
                // the bound for the current clique instance is ci.output + ci.heuristic
                // we store the components separately so we can relax A*'s heuristic:
                //      f(c) = g(c) + eps * h(c)
                //      with g(c) = ci.output
                //           h(c) = ci.heuristic
                if constexpr (instance==0)
                    ci.heuristic = v.min_bound;  // minimize instance 0
                else ci.heuristic = v.max_bound; // maximize instance 1

                //// TEST: can we get a better heuristic value by scanning all layers, no dynamic programming?
                //FloatT alt_heuristic = v.output;
                //for (size_t j = ci.indep_set + 1; j < graph.sets_.size(); ++j)
                //{
                //    FloatT tmp = -std::numeric_limits<FloatT>::infinity();
                //    if constexpr (instance==0) tmp = std::numeric_limits<FloatT>::infinity();

                //    const auto& set1 = graph.sets_[j].vertices;

                //    for (const Vertex& v1 : set1)
                //    {
                //        if (!(c.box.overlaps(v1.box) && v.box.overlaps(v1.box)))
                //            continue;
                //        if constexpr (instance==0)
                //            tmp = tmp > v.output ? v.output : tmp;  // minimize instance 0
                //        else tmp = tmp < v.output ? v.output : tmp; // maximize instance 1
                //    }
                //    alt_heuristic += tmp;
                //}

                //std::cout << "alternative " << ci.indep_set << " ";
                //if constexpr (instance==0)
                //    std::cout << "min " << (alt_heuristic > ci.heuristic);
                //else std::cout << "max " << (alt_heuristic < ci.heuristic);
                //std::cout << " " << ci.heuristic  << ", " << alt_heuristic << std::endl;

                return true; // update successful!
            }
            else ++num_update_fails;
        }
        return false; // out of compatible vertices in `indep_set`
    }

    template <size_t instance, typename BA, typename OF>
    void
    KPartiteGraphOptimize::step_instance(Clique&& c, BA box_adjuster, OF output_filter)
    {
        // invariant: Clique `c` can be extended
        //
        // Things to do (contd.):
        // 3. construct the new clique by merging another vertex from the selected graph
        // 4. [update old] update active instance of given clique `c`, add again if still extendible
        // 5. [update new] check whether we can extend further later (all cliques in
        //    `cliques_` must be extendible)
        //    possible states are (solution=all trees contributed a value
        //                         extendible=valid `indep_set` & `vertex` values exist
        //      - extendible    extendible
        //      - solution      extendible
        //      - extendible    solution
        //    if the clique is not in any of the above states, do not add to `cliques_`

        const KPartiteGraph& graph = std::get<instance>(graph_);

        // v is vertex to merge with: new_c = c + v
        CliqueInstance& ci = std::get<instance>(c.instance);
        const Vertex& v = graph.sets_[ci.indep_set].vertices.at(ci.vertex);
        ci.vertex += 1; // (!) mark the merged vertex as 'used' in old clique, so we dont merge it again later

        // prepare `new_c`, we do this before update_clique(c)
        store_.combine_in_workspace(v.box, c.box); // box of new clique in store_ workspace
        Clique new_c = {
            store_.get_workspace_box(), // temporary workspace box (for update_box), update later!
            c.instance // copy
        };
        CliqueInstance& new_ci = std::get<instance>(new_c.instance);
        new_ci.output += v.output; // output of clique is previous output + output of newly merged vertex
        new_ci.indep_set += 1;
        new_ci.vertex = 0; // start from beginning in new `indep_set` (= tree)
        new_ci.heuristic = 0.0; // if not a solution, will be set by update_clique

        // == UPDATE OLD
        // push old clique if it still has a valid extension
        // we only need to update the current instance: box did not change so
        // other instance cannot be affected
        if (update_clique<instance>(c))
        {
            // if the output filter rejects this, then no more valid expansions
            // of `c` exist (outputs will only get worse)
            bool is_valid_output = output_filter(
                    get0(c.instance).output_bound(),
                    get1(c.instance).output_bound());
            if (is_valid_output)
            {
                //std::cout << "push old again: " << c << std::endl;
                pq_push(std::move(c));
            }
            else
            {
                //std::cout << "rejected old because of output " << c << std::endl;
                ++num_rejected;
            }
        }

        // == UPDATE NEW
        ++num_box_checks;
        bool is_valid_box = box_adjuster(store_);
        bool is_solution0 = is_instance_solution<0>(new_c);
        bool is_solution1 = is_instance_solution<1>(new_c);

        // check if this newly created clique is a solution
        if (is_solution0 && is_solution1)
        {
            bool is_valid_output = output_filter(
                    get0(new_c.instance).output,
                    get1(new_c.instance).output);

            if (is_valid_box && is_valid_output)
            {
                //std::cout << "SOLUTION (" << solutions.size() << "): " << new_c << std::endl;
                // push back to queue so it is 'extracted' when it actually is the optimal solution
                new_c.box = store_.push_workspace();
                pq_push(std::move(new_c));
            }
            else
            {
                //std::cout << "discarding invalid solution" << std::endl;
                ++num_rejected;
            }
        }
        else
        {
            // update both instances of `new_c`, the new box could result in
            // changes in `output_bound`s and `vertex`s values for both
            // instances!  (update_clique updates `output_bound` and `vertex`)
            bool is_valid0 = is_solution0 || update_clique<0>(new_c);
            bool is_valid1 = is_solution1 || update_clique<1>(new_c);
            bool is_valid_output = output_filter(
                    get0(new_c.instance).output_bound(),
                    get1(new_c.instance).output_bound());

            // there is a valid extension of this clique and it is not yet a solution -> push to `cliques_`
            if (is_valid0 && is_valid1 && is_valid_box && is_valid_output)
            {
                //std::cout << "push new: " << new_c << std::endl;
                new_c.box = store_.push_workspace();
                pq_push(std::move(new_c));
            }
            else
            {
                //std::cout << "reject: " << is_valid0 << is_valid1
                //    << is_valid_box << is_valid_output << std::endl;
                ++num_rejected;
            }
        }

        std::get<instance>(num_steps)++;
    }

    template <size_t instance, typename BA, typename OF>
    void
    KPartiteGraphOptimize::expand_clique_instance(Clique&& c, BA box_adjuster, OF output_filter)
    {
        // invariant: Clique `c` can be extended
        //
        // Things to do (cont. step_aux)
        // 3. find first next vertex with box that overlaps c.box
        // 4. construct combined box
        // 5. run box_adjuster, if reject, continue to next vertex
        // 6. compute heuristic
        // 7. check output_filter, if reject, continue to next vertex
        // 8. push new clique to pq

        const CliqueInstance& ci = std::get<instance>(c.instance);
        const KPartiteGraph& graph = std::get<instance>(graph_);
        const auto& next_set = graph.sets_[ci.indep_set];

        for (const Vertex& v : next_set.vertices)
        {
            if (!c.box.overlaps(v.box))
                continue;

            store_.clear_workspace();
            store_.combine_in_workspace(c.box, v.box);
            DomainBox box = store_.get_workspace_box();

            // do we accept this new box?
            ++num_box_checks;
            if (!box_adjuster(store_))
            {
                ++num_rejected;
                continue;
            }

            // recompute heuristic for `instance`
            FloatT heuristic0 = 0.0;
            constexpr size_t plusone0 = instance == 0 ? 1 : 0; // include this indep_set? only when not expanding
            for (auto it = get0(graph_).sets_.cbegin() + get0(c.instance).indep_set + plusone0;
                    it < get0(graph_).sets_.cend() && !std::isinf(heuristic0); ++it)
            {
                FloatT min = std::numeric_limits<FloatT>::infinity(); // minimize this value for graph0
                for (const Vertex& w : it->vertices)
                    if (box.overlaps(w.box))
                        min = std::min(w.output, min);
                heuristic0 += min;
            }

            // recompute heuristic for `other_instance`
            FloatT heuristic1 = 0.0;
            constexpr size_t plusone1 = instance == 1 ? 1 : 0;
            for (auto it = get1(graph_).sets_.cbegin() + get1(c.instance).indep_set + plusone1;
                    it < get1(graph_).sets_.cend() && !std::isinf(heuristic1); ++it)
            {
                FloatT max = -std::numeric_limits<FloatT>::infinity(); // maximize this value for graph1
                for (const Vertex& w : it->vertices)
                    if (box.overlaps(w.box))
                        max = std::max(w.output, max);
                heuristic1 += max;
            }

            // some set where none of the vertices overlap box -> don't add to pq
            if (std::isinf(heuristic0) || std::isinf(heuristic1))
            {
                std::cout << "inf heuristic" << std::endl;
                continue;
            }

            //std::cout << "heuristic " << heuristic0 << ", " << heuristic1 << std::endl;

            // construct new clique
            box = store_.push_workspace(); // (!) push workspace box to store!
            Clique new_c = {
                box,
                c.instance // copy
            };
            CliqueInstance& new_ci = std::get<instance>(new_c.instance);
            new_ci.output += v.output;
            new_ci.indep_set += 1;
            // new_ci.vertex is unused

            // update heuristics
            get0(new_c.instance).heuristic = heuristic0;
            get1(new_c.instance).heuristic = heuristic1;

            //std::cout << "new_c " << new_c << std::endl;

            // do we accept these output bounds?
            if (!output_filter(
                    get0(new_c.instance).output_bound(),
                    get1(new_c.instance).output_bound()))
            {
                ++num_rejected;
                continue;
            }

            pq_push(std::move(new_c));
        }

        std::get<instance>(num_steps)++;
    }

    template <typename BA, typename OF>
    bool
    KPartiteGraphOptimize::step_aux(BA box_adjuster, OF output_filter)
    {
        if (cliques_.empty())
            return false;

        // Things to do:
        // 1. check whether the top of the pq is a solution
        // 2. determine which graph to use to extend the best clique
        // --> goto step_instance / expand_clique_instance

        //std::cout << "ncliques=" << cliques_.size()
        //    << ", num_steps=" << get0(num_steps) << ":" << get1(num_steps)
        //    << ", num_update_fails=" << num_update_fails
        //    << ", num_box_checks=" << num_box_checks
        //    << std::endl;

        Clique c = pq_pop();

        bool is_solution0 = is_instance_solution<0>(c);
        bool is_solution1 = is_instance_solution<1>(c);

        //std::cout << '[' << cliques_.size() << "] " << "best clique " << c << std::endl;

        // move result to solutions if this is a solution
        if (is_solution0 && is_solution1)
        {
            //std::cout << "SOLUTION added "
            //    << get0(c.instance).output << ", "
            //    << get1(c.instance).output << std::endl;
            double t = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count() * 1e-3;
            solutions.push_back({
                std::move(c.box),
                get0(c.instance).output,
                get1(c.instance).output,
                cmp_.eps,
                t - start_time,
                true,
            });

            // NEW: manually increase eps
            // ARA*: decrease eps
            //if (eps_incr_ > 0.0 && cmp_.eps < 1.0)
            //{
            //    FloatT new_eps = cmp_.eps + eps_incr_;
            //    new_eps = new_eps > 1.0 ? 1.0 : new_eps;
            //    //std::cout << "ARA*: eps update: " << cmp_.eps << " -> " << new_eps << std::endl;
            //    set_eps(new_eps, eps_incr_);

            //    // push this solution clique again, if we find it again next
            //    // time, we know that this is a solution with a better eps bound
            //    pq_push(std::move(c));
            //}
        }
        // if not a solution, extend from graph0 first, then graph1, then graph0 again...
        else if (!is_solution0 && (get0(c.instance).indep_set <= get1(c.instance).indep_set
                    || is_solution1))
        {
            //std::cout << "step(): extend graph0" << std::endl;
            if (heuristic_type == DYN_PROG)
                step_instance<0>(std::move(c), box_adjuster, output_filter);
            else
                expand_clique_instance<0>(std::move(c), box_adjuster, output_filter);
        }
        else if (!is_solution1)
        {
            //std::cout << "step(): extend graph1" << std::endl;
            if (heuristic_type == DYN_PROG)
                step_instance<1>(std::move(c), box_adjuster, output_filter);
            else
                expand_clique_instance<1>(std::move(c), box_adjuster, output_filter);
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
                [](DomainStore&) { return true; },
                [](FloatT, FloatT) { return true; });
    }

    bool
    KPartiteGraphOptimize::step(BoxAdjuster ba)
    {
        return step_aux(
                ba,
                [](FloatT, FloatT) { return true; }); // accept all outputs
    }

    bool
    KPartiteGraphOptimize::step(BoxAdjuster ba, FloatT max_output0, FloatT min_output1)
    {
        return step_aux(
                ba,
                [max_output0, min_output1](FloatT output0, FloatT output1) {
                    return output0 <= max_output0 && output1 >= min_output1;
                });
    }

    bool
    KPartiteGraphOptimize::step(BoxAdjuster ba, FloatT min_output_difference)
    {
        return step_aux(
                ba,
                [min_output_difference](FloatT output0, FloatT output1) {
                    return (output1 - output0) >= min_output_difference;
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
    KPartiteGraphOptimize::steps(int K, BoxAdjuster ba)
    {
        for (int i=0; i<K; ++i)
            if (!step(ba)) return false;
        return true;
    }

    bool
    KPartiteGraphOptimize::steps(int K, BoxAdjuster ba, FloatT max_output0, FloatT min_output1)
    {
        for (int i=0; i<K; ++i)
            if (!step(ba, max_output0, min_output1)) return false;
        return true;
    }

    bool
    KPartiteGraphOptimize::steps(int K, BoxAdjuster ba, FloatT min_output_difference)
    {
        for (int i=0; i<K; ++i)
            if (!step(ba, min_output_difference)) return false;
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
            get0(c.instance).output_bound(cmp_.eps),
            get1(c.instance).output_bound(cmp_.eps)
        };
    }

    size_t
    KPartiteGraphOptimize::num_candidate_cliques() const
    {
        return cliques_.size();
    }

    const KPartiteGraph&
    KPartiteGraphOptimize::graph0() const { return get0(graph_); }

    const KPartiteGraph&
    KPartiteGraphOptimize::graph1() const { return get1(graph_); }






    // PARALLEL
    
    static bool DEFAULT_BOX_ADJUSTER(DomainStore&) { return true; }

    Worker::Worker()
        : index_(0)
        , work_flag_(false), stop_flag_(false)
        , redistribute_(RDIST_DISABLED)
        , num_millisecs_(0)
        , new_valid_solutions_(0)
        , thread_{}
        , mutex_{}
        , cv_{}
        , opt_{}
        , box_adjuster_{DEFAULT_BOX_ADJUSTER}
    {}

    SharedWorkerInfo::SharedWorkerInfo(FloatT eps)
        : max_output0(std::numeric_limits<FloatT>::quiet_NaN())
        , min_output1(std::numeric_limits<FloatT>::quiet_NaN())
        , min_output_difference(std::numeric_limits<FloatT>::quiet_NaN())
        , best_bound(std::numeric_limits<FloatT>::quiet_NaN())
        , new_eps(eps)
     {}
    
    void 
    KPartiteGraphParOpt::worker_fun(std::deque<Worker> *workers, const SharedWorkerInfo *info,
            size_t self_index)
    {
        Worker *self = &workers->at(self_index);

        while (true)
        {
            std::unique_lock lock(self->mutex_);
            self->cv_.wait(lock, [self]() { return self->work_flag_; });

            if (self->stop_flag_)
            {
                break;
            }
            if (self->num_millisecs_ != 0)
            {
                using clock = std::chrono::steady_clock;
                auto start = clock::now();
                auto stop = start + std::chrono::milliseconds(self->num_millisecs_);

                // if necessary, update eps first
                if (info->new_eps != self->opt_->get_eps())
                    self->opt_->set_eps(info->new_eps, false);

                // heap was reconstructed during redistribute_work, make it a
                // heap again (and maybe eps changed too)
                std::make_heap(self->opt_->cliques_.begin(),
                        self->opt_->cliques_.end(), self->opt_->cmp_);

                if (!std::isnan(info->max_output0)) // assume min_output1 is also valid
                {
                    while (clock::now() < stop)
                        if (!self->opt_->steps(100, self->box_adjuster_,
                                    info->max_output0, info->min_output1))
                            break;
                }
                else if (!std::isnan(info->min_output_difference))
                {
                    while (clock::now() < stop)
                        if (!self->opt_->steps(100, self->box_adjuster_, info->min_output_difference))
                            break;
                }
                else
                {
                    while (clock::now() < stop)
                        if (!self->opt_->steps(100, self->box_adjuster_))
                            break;
                }
                self->num_millisecs_ = 0;
            }
            if (self->redistribute_ == Worker::RDIST_SETUP)
            {
                // signal to main thread that we're in this section, we have to
                // make sure all workers are in this section before we start
                // copying cliques between workers
                self->redistribute_ = Worker::RDIST_READY;
                lock.unlock();
                self->cv_.notify_one();

                lock.lock();
                self->cv_.wait(lock, [self]() { return self->redistribute_ == Worker::RDIST_GO; });

                // == DO REDISTRIBUTE: copy cliques from each worker
                size_t num_threads = workers->size();
                std::vector<Clique> new_cliques;
                for (size_t i = 0; i < num_threads; ++i)
                {
                    const KPartiteGraphOptimize *opt = &*workers->at(i).opt_;
                    for (size_t j = self_index; j < opt->cliques_.size(); j += num_threads)
                        new_cliques.push_back(opt->cliques_[j]);
                }
                //std::make_heap(new_cliques.begin(), new_cliques.end(), self->opt_->cmp_);

                self->redistribute_ = Worker::RDIST_DONE;
                lock.unlock();
                self->cv_.notify_one();

                lock.lock();
                self->cv_.wait(lock, [self]() { return self->redistribute_ == Worker::RDIST_STORE; });

                // == STORE NEW SEARCH CLIQUES
                std::swap(self->opt_->cliques_, new_cliques);
                self->redistribute_ = Worker::RDIST_DISABLED;

                // mark the newly added solutions as invalid if not better than bounds of all workers
                self->new_valid_solutions_ = 0;
                FloatT eps = self->opt_->cmp_.eps;
                for (int j = self->opt_->solutions.size() - 1; j >= 0; --j)
                {
                    Solution& sol = self->opt_->solutions[j];
                    if (sol.eps != eps) break; // only look at the last solutions for the current eps
                    if (sol.output_difference() < info->best_bound)
                    {
                        //std::lock_guard l(m);
                        //std::cout << "w" << self->index_ << ": invalid "
                        //    << sol.output_difference() << " < " << info->best_bound 
                        //    << " (" << self->opt_->cmp_.eps << ")" << std::endl;
                        sol.is_valid = false;
                    }
                    else
                    {
                        //std::lock_guard l(m);
                        //std::cout << "w" << self->index_ << ":   valid "
                        //    << sol.output_difference() << " >= " << info->best_bound
                        //    << " (" << self->opt_->cmp_.eps << ")" << std::endl;
                        sol.is_valid = true;
                        self->new_valid_solutions_ += 1;
                    }
                }
            }
            self->work_flag_ = false;
            lock.unlock();
            self->cv_.notify_one();
        }
    }
    
    KPartiteGraphParOpt::KPartiteGraphParOpt(size_t num_threads,
            const KPartiteGraphOptimize& opt)
        : workers_{new std::deque<Worker>(num_threads)}
        , info_{new SharedWorkerInfo(opt.cmp_.eps)}
    {
        for (size_t i = 0; i < num_threads; ++i)
        {
            //workers_->emplace(workers_->end());
            Worker& w = workers_->at(i);
            std::lock_guard guard(w.mutex_);
            w.index_ = i;
            w.opt_.emplace(opt, i, num_threads);
            w.opt_->store_.set_max_mem_size(opt.store_.get_max_mem_size());
            w.thread_ = std::thread(worker_fun, &*workers_, &*info_, i);
        }
    }

    void
    KPartiteGraphParOpt::join_all()
    {
        for (size_t i = 0; i < num_threads(); ++i)
        {
            Worker& w = workers_->at(i);
            {
                std::lock_guard(w.mutex_);
                w.work_flag_ = true; 
                w.stop_flag_ = true;
            }
            w.cv_.notify_one();
        }
        for (size_t i = 0; i < num_threads(); ++i)
            workers_->at(i).thread_.join();
    }

    void
    KPartiteGraphParOpt::wait()
    {
        // try to lock the mutex, if successful, workers is done and waiting for work
        for (size_t i = 0; i < num_threads(); ++i)
        {
            Worker *w = &workers_->at(i);
            std::unique_lock lock(w->mutex_);
            w->cv_.wait(lock, [w](){ return !w->work_flag_; });
        }
    }

    void
    KPartiteGraphParOpt::redistribute_work()
    {
        //auto start = std::chrono::system_clock::now();

        // ask all workers to go to their redistribute_work code section
        for (size_t i = 0; i < num_threads(); ++i)
        {
            Worker& w = workers_->at(i);
            {
                std::lock_guard(w.mutex_);
                w.work_flag_ = true; 
                w.redistribute_ = Worker::RDIST_SETUP;
            }
            w.cv_.notify_one();
        }

        // wait untill all workers have notified that they have entered the redistribute_work
        // section
        for (size_t i = 0; i < num_threads(); ++i)
        {
            Worker *w = &workers_->at(i);
            std::unique_lock lock(w->mutex_);
            w->cv_.wait(lock, [w](){ return w->redistribute_ == Worker::RDIST_READY; });
        }

        // All workers are in state RDIST_READY -> no worker is modifying itself in
        // unexpected ways. We can tell them to start redistributing search cliques

        for (size_t i = 0; i < num_threads(); ++i)
        {
            Worker *w = &workers_->at(i);
            {
                std::unique_lock lock(w->mutex_);
                w->redistribute_ = Worker::RDIST_GO;
            }
            w->cv_.notify_one();
        }

        // wait for all workers to finish redistributing, then we can tell them to swap out their
        // search cliques
        for (size_t i = 0; i < num_threads(); ++i)
        {
            Worker *w = &workers_->at(i);
            std::unique_lock lock(w->mutex_);
            w->cv_.wait(lock, [w](){ return w->redistribute_ == Worker::RDIST_DONE; });
        }

        // all workers are done with redistribution, tell them to swap their search states
        // and
        // each worker checks its solutions: A solution is "valid" when it is
        // currently better than the bounds of all workers
        FloatT best_bound = -std::numeric_limits<FloatT>::infinity();
        for (size_t i = 0; i < num_threads(); ++i)
        {
            const Worker& w = workers_->at(i);
            const auto& c = w.opt_->cliques_;
            FloatT eps = w.opt_->cmp_.eps;
            if (c.empty()) continue;
            FloatT bound = w.opt_->cliques_.front().output_difference(eps);
            //std::cout << "w" << i << " eps=" << eps << ", bound=" << bound << std::endl;
            best_bound = std::max(best_bound, bound);
        }
        info_->best_bound = best_bound;
        //std::cout << "best bound: " << best_bound << std::endl;

        for (size_t i = 0; i < num_threads(); ++i)
        {
            Worker *w = &workers_->at(i);
            {
                std::unique_lock lock(w->mutex_);
                w->redistribute_ = Worker::RDIST_STORE;
            }
            w->cv_.notify_one();
        }

        wait();

        //auto stop = std::chrono::system_clock::now();
        //double dur = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
        //std::cout << "redistribute_work in " << (dur * 1e-6) << std::endl;
    }

    void
    KPartiteGraphParOpt::steps_for(size_t num_millisecs)
    {
        //auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < num_threads(); ++i)
        {
            Worker& w = workers_->at(i);
            {
                std::lock_guard(w.mutex_);
                w.work_flag_ = true; 
                w.num_millisecs_ = num_millisecs;
            }
            w.cv_.notify_one();
        }
        wait();
        redistribute_work();
        

        //auto stop = std::chrono::steady_clock::now();
        //double dur = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
        //std::cout << "duration steps_for " << (dur * 1e-6) << std::endl;
    }

    FloatT
    KPartiteGraphParOpt::get_eps() const
    {
        return info_->new_eps;
    }

    void
    KPartiteGraphParOpt::set_eps(FloatT eps)
    {
        if (eps <= 0.0) throw std::runtime_error("nonsense eps");
        if (eps > 1.0) throw std::runtime_error("nonsense eps");

        info_->new_eps = eps; // will be picked up by workers in 'steps_for'
    }

    const KPartiteGraphOptimize&
    KPartiteGraphParOpt::worker_opt(size_t worker_index) const
    {
        return *workers_->at(worker_index).opt_;
    }

    void
    KPartiteGraphParOpt::set_output_limits(FloatT max_output0, FloatT min_output1)
    {
        info_->max_output0 = max_output0;
        info_->min_output1 = min_output1;
        info_->min_output_difference = std::numeric_limits<FloatT>::quiet_NaN();
    }

    void 
    KPartiteGraphParOpt::set_output_limits(FloatT min_output_difference)
    {
        info_->max_output0 = std::numeric_limits<FloatT>::quiet_NaN();
        info_->min_output1 = std::numeric_limits<FloatT>::quiet_NaN();
        info_->min_output_difference = min_output_difference;
    }

    size_t
    KPartiteGraphParOpt::num_solutions() const {
        size_t sum = 0;
        for (size_t i = 0; i < num_threads(); ++i)
            sum += workers_->at(i).opt_->solutions.size();
        return sum;
    }

    size_t
    KPartiteGraphParOpt::num_new_valid_solutions() const {
        size_t sum = 0;
        for (size_t i = 0; i < num_threads(); ++i)
            sum += workers_->at(i).new_valid_solutions_;
        return sum;
    }

    size_t
    KPartiteGraphParOpt::num_candidate_cliques() const {
        size_t sum = 0;
        for (size_t i = 0; i < num_threads(); ++i)
            sum += workers_->at(i).opt_->num_candidate_cliques();
        return sum;
    }

    two_of<FloatT>
    KPartiteGraphParOpt::current_bounds() const {
        FloatT lower = 0, upper = 0;
        for (size_t i = 0; i < num_threads(); ++i)
        {
            auto p = workers_->at(i).opt_->current_bounds();
            lower = std::min(lower, get0(p));
            upper = std::max(upper, get1(p));
        }
        return { lower, upper };
    }
    
    std::vector<size_t>
    KPartiteGraphParOpt::current_memory() const {
        std::vector<size_t> mem;
        mem.reserve(num_threads());
        for (size_t i = 0; i < num_threads(); ++i)
            mem.push_back(workers_->at(i).opt_->store().get_mem_size());
        return mem;
    }



    // ------------------------------------------------------------------------

    bool
    EasyBoxAdjuster::handle_one_out_of_k(const OneOutOfK& c,
                std::vector<DomainPair>& workspace) const
    {
        // check if there's a TRUE feature, if so, we need to make all others FALSE
        auto it1 = c.ids.begin();
        int true_id = -1;
        int unset_id = -1;
        bool unset_in_box = false;
        size_t num_false = 0;
        //std::cout << "EasyBoxAdjuster: " << DomainBox(&*workspace.begin(), &*workspace.end()) << std::endl;
        for (size_t i = 0; i < workspace.size() && it1 != c.ids.end();)
        {
            if (workspace[i].first == *it1)
            {
                if (!workspace[i].second.overlaps(FALSE_DOMAIN)) // not FALSE domain, assume TRUE domain
                {                                                // -> set all other to false
                    if (true_id != -1)
                    {
                        std::cout << "EasyBoxAdjuster: more than one true, REJECT!" << std::endl;
                        return false; // two TRUE features, and there can only be one
                    }
                    true_id = *it1;
                }
                else if (!workspace[i].second.overlaps(TRUE_DOMAIN)) // not TRUE, so must be FALSE
                    ++num_false;
                else { unset_id = *it1; unset_in_box = true; } // not not FALSE and not not TRUE, so must be anything
                ++i; ++it1;
            }
            else if (workspace[i].first > *it1) { unset_id = *it1; ++it1; }
            else ++i;
        }

        //std::cout << "EasyBoxAdjuster: num_false: " << num_false << std::endl;

        int num_added = 0;

        // CASE 0: everything is FALSE, that's invalid
        if (num_false == c.ids.size())
        {
            std::cout << "EasyBoxAdjuster: all false, REJECT!" << std::endl;
            return false;
        }

        // CASE 1: everything except one is set to FALSE -> write TRUE in the one
        else if (true_id == -1 && num_false + 1 == c.ids.size())
        {
            if (it1 == c.ids.end() - 1) unset_id = *(c.ids.end() - 1);
            if (unset_id == -1) throw std::runtime_error("assertion error");

            if (unset_in_box)
            {
                for (auto&& [id, dom] : workspace)
                {
                    if (id == unset_id)
                    {
                        dom = TRUE_DOMAIN;
                        std::cout << "EasyBoxAdjuster: " << id << " set to TRUE" << std::endl;
                        break;
                    }
                }
            }
            else
            {
                std::cout << "EasyBoxAdjuster: " << unset_id << " set to TRUE (added)" << std::endl;
                workspace.push_back({ unset_id, TRUE_DOMAIN });
                for (int i = workspace.size() - 1; i > 0; --i) // ids sorted
                    if (workspace[i-1].first > workspace[i].first)
                        std::swap(workspace[i-1], workspace[i]);
            }
        }

        // CASE 2: we found one TRUE -> write FALSE in all others
        else if (true_id != -1)
        {
            it1 = c.ids.begin();
            size_t end = workspace.size();
            for (size_t i = 0; i < end && it1 != c.ids.end();)
            {
                if (workspace[i].first == *it1)
                {
                    if (workspace[i].first != true_id)
                        workspace[i].second = FALSE_DOMAIN;
                    ++i; ++it1;
                }
                else if (workspace[i].first > *it1) // we skipped over *it1, so it's not in the box -> add it
                {
                    workspace.push_back({*it1, FALSE_DOMAIN});
                    std::cout << "EasyBoxAdjuster: setting " << *it1 << " to FALSE" << std::endl;
                    ++it1;
                    ++num_added;
                }
                else ++i;
            }
        }

        // CASE 3: anything in between: we can't do much, no unit propagation possible
        else {}

        // make sure all newly added ids are in sorted order
        if (num_added > 0)
        {
            std::sort(workspace.begin(), workspace.end(),
                    [](const DomainPair& a, const DomainPair& b) {
                        return a.first < b.first;
                    });
        }

        return true;
    }

    bool
    EasyBoxAdjuster::operator()(DomainStore& store) const
    {
        auto& workspace = store.workspace();
        for (const OneOutOfK& c : one_out_of_ks_)
        {
            if (!handle_one_out_of_k(c, workspace))
                return false;
        }

        return true; // accept the box in the store's workspace
    }

    void
    EasyBoxAdjuster::add_one_out_of_k(std::vector<int> ids)
    {
        if (!ids.empty())
        {
            std::sort(ids.begin(), ids.end());
            one_out_of_ks_.push_back({ids});
        }
    }

} /* namespace treeck */
