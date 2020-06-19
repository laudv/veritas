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

    DomainStore::DomainStore(const FeatInfo& finfo)
        : store_{}, box_size_(0), max_mem_size_(DOMAIN_STORE_MAX_MEM)
    {
        const size_t DEFAULT_SIZE = 1024*1024 / sizeof(Domain); // 1mb of domains
        Block block;
        block.reserve(DEFAULT_SIZE);
        store_.push_back(std::move(block));

        push_prototype_box(finfo);
    }

    DomainStore::Block&
    DomainStore::get_last_block()
    {
        Block& block = store_.back();
        if (block.capacity() - block.size() < box_size_)
        {
            size_t mem = get_mem_size();
            size_t rem_capacity = (max_mem_size_ - mem) / sizeof(Domain);
            if (rem_capacity > block.capacity() * 2) // double size of blocks each time, unless memory limit almost reached
                rem_capacity = block.capacity() * 2;
            else if (rem_capacity > 0)
                std::cerr << "WARNING: almost running out of memory, "
                    << static_cast<double>(rem_capacity * sizeof(Domain)) / (1024.0*1024.0)
                    << " mb left" << std::endl;
            else
                throw std::runtime_error("DomainStore: out of memory");

            Block new_block;
            new_block.reserve(rem_capacity);
            store_.push_back(std::move(new_block));

            std::cout << "DomainStore memory: " << mem << " bytes, "
                << (static_cast<double>(get_mem_size()) / (1024.0 * 1024.0)) << " mb ("
                << store_.size() << " blocks)" << std::endl;
        }
        return store_.back();
    }

    void
    DomainStore::push_prototype_box(const FeatInfo& finfo)
    {
        if (box_size_ > 0)
            throw std::runtime_error("prototype already pushed");
        box_size_ = finfo.num_ids();

        Block& block = get_last_block();

        // push domains for instance0
        for (FeatId feat_id : finfo.feat_ids0())
        {
            int id = finfo.get_id(0, feat_id);
            if (id != block.size()) throw std::runtime_error("invalid state");

            if (finfo.is_real(id)) block.push_back({});
            else                   block.push_back(BOOL_DOMAIN);
        }

        // push domains for instance1
        for (FeatId feat_id : finfo.feat_ids1())
        {
            int id = finfo.get_id(1, feat_id);
            if (finfo.is_instance0_id(id)) continue;
            if (id != block.size()) throw std::runtime_error("invalid state");

            if (finfo.is_real(id)) block.push_back({});
            else                   block.push_back(BOOL_DOMAIN);
        }
    }

    size_t
    DomainStore::get_mem_size() const
    {
        size_t mem = 0;
        for (const Block& b : store_)
            mem += b.capacity() * sizeof(Domain);
        return mem;
    }

    void
    DomainStore::set_max_mem_size(size_t mem)
    {
        max_mem_size_ = mem;
    }

    DomainBox
    DomainStore::push_box()
    {
        // copy the prototype domain
        DomainT *ptr = &store_[0][0];
        return push_copy({ptr, ptr + box_size_});
    }

    DomainBox
    DomainStore::push_copy(const DomainBox& other)
    {
        if (box_size_ == 0) throw std::runtime_error("zero-sized prototype box");
        if (other.size() != box_size_) throw std::runtime_error("incompatible boxes");

        Block& block = get_last_block();
        size_t start_index = block.size();

        for (const DomainT& d : other)
            block.push_back(d);

        if (block.size() != start_index+box_size_) throw std::runtime_error("invalid state");

        DomainT *ptr = &block[start_index];
        return { ptr, ptr + box_size_ };
    }




    DomainBox::DomainBox(DomainT *b, DomainT *e) : begin_(b), end_(e) { }

    DomainBox
    DomainBox::null_box() {
        return DomainBox(nullptr, nullptr);
    }

    DomainBox::const_iterator
    DomainBox::begin() const
    {
        return begin_;
    }

    DomainBox::const_iterator
    DomainBox::end() const
    {
        return end_;
    }

    size_t
    DomainBox::size() const
    {
        return end_ - begin_;
    }

    bool
    DomainBox::is_right_neighbor(const DomainBox& other) const
    {
        auto it0 = begin_;
        auto it1 = other.begin_;
        
        for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
        {
            const RealDomain& a = *it0;
            const RealDomain& b = *it1;
            if (a == b)
                continue;
            if (a.hi != b.lo) // discontinuity
                return false;
        }

        return true;
    }

    void
    DomainBox::join_right_neighbor(const DomainBox& other)
    {
        auto it0 = begin_;
        auto it1 = other.begin_;
        
        for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
        {
            RealDomain& a = *it0;
            const RealDomain& b = *it1;
            if (a == b)
                continue;
            if (a.hi != b.lo) // discontinuity
                throw std::runtime_error("not a right neighbor");
            a.hi = b.hi;
        }
    }

    void
    DomainBox::refine(Split split, bool is_left_child, FeatIdMapper fmap)
    {
        visit_split(
                [this, &fmap, is_left_child](const LtSplit& s) {
                    int id = fmap(s.feat_id);
                    DomainT *ptr = begin_ + id;
                    //RealDomain dom = util::get_or<RealDomain>(*ptr);
                    *ptr = refine_domain(*ptr, s, is_left_child);
                },
                [this, &fmap, is_left_child](const BoolSplit& bs) {
                    int id = fmap(bs.feat_id);
                    DomainT *ptr = begin_ + id;
                    LtSplit s{bs.feat_id, 1.0};
                    //BoolDomain dom = util::get_or<BoolDomain>(*ptr);
                    *ptr = refine_domain(*ptr, s, is_left_child);
                },
                split);
    }

    bool
    DomainBox::overlaps(const DomainBox& other) const
    {
        //std::cout << "OVERLAPS" << std::endl;
        //std::cout << "  " << *this << std::endl;
        //std::cout << "  " << other << std::endl;

        auto it0 = begin_;
        auto it1 = other.begin_;
        
        for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
        {
            //bool overlaps = visit_domain(
            //    [it1](const RealDomain& dom0) {
            //        auto dom1 = util::get_or<RealDomain>(*it1);
            //        return dom0.overlaps(dom1);
            //    },
            //    [it1](const BoolDomain& dom0) {
            //        auto dom1 = util::get_or<BoolDomain>(*it1);
            //        return (dom0.value_ & dom1.value_) != 0;
            //    },
            //    *it0);

            bool overlaps = it0->overlaps(*it1);

            if (!overlaps)
                return false;
        }

        return true;
    }

    bool
    DomainBox::covers(const DomainBox& other) const
    {
        auto it0 = begin_;
        auto it1 = other.begin_;
        
        for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
        {
            if (!it0->covers(*it1))
                return false;
        }

        return true;
    }

    void
    DomainBox::combine(const DomainBox& other) const
    {
        DomainT *it0 = begin_;
        DomainT *it1 = other.begin_;

        // assume sorted
        for (; it0 != end_ && it1 != other.end_; ++it0, ++it1)
        {
            *it0 = it0->intersect(*it1);
            //visit_domain(
            //    [it0](const RealDomain& dom1) {
            //        auto dom0 = util::get_or<RealDomain>(*it0);
            //        *it0 = dom0.intersect(dom1);
            //    },
            //    [it0](const BoolDomain& dom1) {
            //        auto dom0 = util::get_or<BoolDomain>(*it0);
            //        *it0 = dom0.intersect(dom1);
            //    },
            //    *it1);
        }
    }

    std::ostream&
    operator<<(std::ostream& s, const DomainBox& box)
    {
        s << "DBox { ";
        for (auto it = box.begin(); it != box.end(); ++it)
        {
            int id = it - box.begin();
            bool is_everything = visit_domain(
                    [](const RealDomain& dom) { return dom.is_everything(); },
                    [](const BoolDomain& dom) { return dom.is_everything(); },
                    *it);
            if (!is_everything)
                s << id << ":" << *it << " ";
        }
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

    KPartiteGraph::KPartiteGraph(DomainStore* store)
        : store_(store)
    {
        //sets_.push_back({
        //    std::vector<Vertex>{{{}, 0.0}} // one dummy vertex
        //});
    }

    KPartiteGraph::KPartiteGraph(DomainStore *store, const AddTree& addtree, FeatIdMapper fmap)
        : store_(store)
    {
        if (addtree.base_score != 0.0)
        {
            //std::cout << "adding base_score set" << std::endl;
            IndependentSet set;
            set.vertices.push_back({store_->push_box(), addtree.base_score});
            sets_.push_back(set);
        }

        for (const AddTree::TreeT& tree : addtree.trees())
        {
            IndependentSet set;
            fill_independence_set(set, tree.root(), fmap);

            sets_.push_back(set);
        }
    }

    KPartiteGraph::KPartiteGraph(DomainStore *store, const AddTree& addtree,
            const FeatInfo& finfo, int instance)
        : KPartiteGraph(store, addtree, [=](FeatId feat_id) { return finfo.get_id(instance, feat_id); })
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
            DomainBox box = store_->push_box();
            while (!node.is_root())
            {
                auto child_node = node;
                node = node.parent();
                box.refine(node.get_split(), child_node.is_left_child(), fmap);
            }
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
                //std::cout << "merge " << set0.vertices.size() << " x " << it->vertices.size() << std::endl;
                for (const auto& v0 : set0.vertices)
                {
                    for (const auto& v1 : it->vertices)
                    {
                        if (v0.box.overlaps(v1.box))
                        {
                            DomainBox box = store_->push_copy(v0.box);
                            box.combine(v1.box);
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

    void
    KPartiteGraph::simplify(FloatT max_err, bool overestimate)
    {
        struct E { size_t set; DomainBox box; FloatT abs_err; };
        std::vector<E> errors;

        while (true) {
            FloatT min_err = max_err; // anything larger is skipped
            size_t min_set = sets_.size(); // invalid
            size_t min_vertex = 0;

            for (size_t j = 0; j < sets_.size(); ++j)
            {
                const IndependentSet& set = sets_[j];
                for (size_t i = 1; i < set.vertices.size(); ++i)
                {
                    const Vertex& v0 = set.vertices[i-1];
                    const Vertex& v1 = set.vertices[i];
                    if (v0.box.is_right_neighbor(v1.box))
                    {
                        FloatT err = std::abs(v0.output - v1.output);
                        if (err <= min_err)
                        {
                            for (E e : errors)
                            {
                               if (v0.box.overlaps(e.box) || v1.box.overlaps(e.box))
                                    err += e.abs_err;
                            }
                            if (err <= min_err)
                            {
                                min_err = err;
                                min_set = j;
                                min_vertex = i;
                            }
                        }
                    }
                }
            }

            // nothing found, everything larger than max_err, we're done
            if (min_set == sets_.size())
                break;

            // merge the two neighboring vertices with the smallest error
            //std::cout << "simplify " << min_set << ", " << min_vertex << " with error " << min_err << std::endl;

            IndependentSet& set = sets_[min_set];
            Vertex& v0 = set.vertices[min_vertex-1];
            const Vertex& v1 = set.vertices[min_vertex];
            FloatT err = std::abs(v0.output - v1.output); // measure before changing v0

            //std::cout << "is_right_neighbor " << v0.box.is_right_neighbor(v1.box) << std::endl;

            // update v0
            //std::cout << "before " << v0.box << ", " << v0.output << std::endl;
            //std::cout << "    +  " << v1.box << ", " << v0.output << std::endl;
            v0.box.join_right_neighbor(v1.box);
            //std::cout << "after " << v0.box << std::endl;

            v0.output = overestimate ? std::max(v0.output, v1.output) : std::min(v0.output, v1.output);
            v0.max_bound = v0.output;
            v0.min_bound = v0.output;

            //std::cout << "new vertex output " << v0.output << std::endl;

            // remove v1
            for (size_t i = min_vertex + 1; i < set.vertices.size(); ++i)
                set.vertices[i-1] = set.vertices[i];
            set.vertices.pop_back();

            // store error of region
            errors.push_back({min_set, v0.box, err});


            //for (auto e : errors)
            //    std::cout << "- error: " << e.abs_err << " for " << e.box << std::endl;
        }
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

    //FloatT
    //CliqueInstance::output_bound(FloatT eps) const
    //{
    //    return output + eps * prev_bound;
    //}

    bool
    CliqueMaxDiffPqCmp::operator()(const Clique& a, const Clique& b) const
    {
        FloatT diff_a = get1(a.instance).output_bound(eps) - get0(a.instance).output_bound(eps);
        FloatT diff_b = get1(b.instance).output_bound(eps) - get0(b.instance).output_bound(eps);

        return diff_a < diff_b;
    }

    std::ostream&
    operator<<(std::ostream& s, const CliqueInstance& ci)
    {
        return s
            << "    output=" << ci.output << ", bound=" << ci.output_bound() << ", " << std::endl
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

    //KPartiteGraphOptimize::KPartiteGraphOptimize(KPartiteGraph& g0)
    //    : KPartiteGraphOptimize(g0.store_, g0, {g0.store_}) { }

    //KPartiteGraphOptimize::KPartiteGraphOptimize(bool, KPartiteGraph& g1)
    //    : KPartiteGraphOptimize(g1.store_, {g1.store_}, g1) { }

    KPartiteGraphOptimize::KPartiteGraphOptimize(KPartiteGraph& g0, KPartiteGraph& g1)
        : KPartiteGraphOptimize(g0.store_, g0, g1) { }

    KPartiteGraphOptimize::KPartiteGraphOptimize(DomainStore *store, KPartiteGraph& g0, KPartiteGraph& g1)
        : store_(store)
        , graph_{g0, g1} // minimize g0, maximize g1
        , cliques_()
        , cmp_{1.0}
        , eps_incr_{0.0}
        , nsteps{0, 0}
        , nupdate_fails{0}
        , nrejected{0}
        , nbox_filter_calls{0}
        , solutions()
    {
        if (g0.store_ != store_ || g1.store_ != store_)
            throw std::runtime_error("invalid store");

        auto&& [output_bound0, max0] = g0.propagate_outputs(); // min output bound of first clique
        auto&& [min1, output_bound1] = g1.propagate_outputs(); // max output bound of first clique

        g0.sort_bound_asc(); // choose vertex with smaller `output` first
        g1.sort_bound_desc(); // choose vertex with larger `output` first

        bool unsat = std::isinf(output_bound0) || std::isinf(output_bound1); // some empty indep set
        if (!unsat && (g0.num_independent_sets() > 0 || g1.num_independent_sets() > 0))
        {
            DomainBox box = store_->push_box();
            cliques_.push_back({
                box, // empty domain, ie no restrictions
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

    FloatT
    KPartiteGraphOptimize::get_eps() const
    {
        return cmp_.eps;
    }

    void
    KPartiteGraphOptimize::set_eps(FloatT eps, FloatT eps_incr)
    {
        // we maximize diff, so h(x) must be underestimated to make the search prefer deeper solutions
        if (eps > 1.0) throw std::runtime_error("nonsense eps");
        if (eps_incr < 0.0) throw std::runtime_error("nonsense eps_incr");

        eps_incr_ = eps_incr;
        cmp_.eps = eps;
        std::make_heap(cliques_.begin(), cliques_.end(), cmp_);

        //std::cout << "ARA* EPS set to " << cmp_.eps << std::endl;
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
                // the bound for the current clique instance is ci.output + ci.prev_bound
                // we store the components separately so we can relax A*'s heuristic:
                //      f(c) = g(c) + eps * h(c)
                //      with g(c) = ci.output
                //           h(c) = ci.prev_bound
                if constexpr (instance==0)
                    ci.prev_bound = v.min_bound;  // minimize instance 0
                else ci.prev_bound = v.max_bound; // maximize instance 1

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

        // prepare `new_c`
        Clique new_c = {
            DomainBox::null_box(), // null box for now, we'll first check if we can reuse `c`'s box
            c.instance // copy
        };

        // v is vertex to merge with: new_c = c + v
        CliqueInstance& ci = std::get<instance>(c.instance);
        const Vertex& v = graph.sets_[ci.indep_set].vertices.at(ci.vertex);
        ci.vertex += 1; // (!) mark the merged vertex as 'used' in old clique, so we dont merge it again later

        CliqueInstance& new_ci = std::get<instance>(new_c.instance);
        new_ci.output += v.output; // output of clique is previous output + output of newly merged vertex
        new_ci.indep_set += 1;
        new_ci.vertex = 0; // start from beginning in new `indep_set` (= tree)
        new_ci.prev_bound = 0.0; // if not a solution, will be set by update_clique

        // UPDATE OLD
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

                // `new_c` needs a new box
                new_c.box = store_->push_copy(c.box);
                new_c.box.combine(v.box);
            }
            else
            {
                //std::cout << "rejected old because of output " << c << std::endl;
                ++nrejected;

                // `new_c` can reuse `c`'s box
                new_c.box = c.box;
                new_c.box.combine(v.box);
            }
        }
        else
        {
            //std::cout << "done with old: " << c << std::endl;

            // `new_c` can reuse `c`'s box
            new_c.box = c.box;
            new_c.box.combine(v.box);
        }

        // UPDATE NEW
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
                    get0(new_c.instance).output_bound(),
                    get1(new_c.instance).output_bound());

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

        std::get<instance>(nsteps)++;
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

        //std::cout << '[' << cliques_.size() << "] " << "best clique " << c << std::endl;

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

            // ARA*: decrease eps
            if (eps_incr_ > 0.0 && cmp_.eps < 1.0)
            {
                FloatT new_eps = cmp_.eps + eps_incr_;
                new_eps = new_eps > 1.0 ? 1.0 : new_eps;
                std::cout << "ARA*: eps update: " << cmp_.eps << " -> " << new_eps << std::endl;
                set_eps(new_eps, eps_incr_);
            }
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
        return true;
    }

    bool
    KPartiteGraphOptimize::steps(int K, BoxFilter bf, FloatT min_output_difference)
    {
        for (int i=0; i<K; ++i)
            if (!step(bf, min_output_difference)) return false;
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
            get0(c.instance).output_bound(),
            get1(c.instance).output_bound()
        };
    }

    size_t
    KPartiteGraphOptimize::num_candidate_cliques() const
    {
        return cliques_.size();
    }
} /* namespace treeck */
