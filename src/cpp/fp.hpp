/**
 * \file fp.hpp
 *
 * Fixed precision.
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_FP_HPP
#define VERITAS_FP_HPP

#include "basics.hpp"
#include "box.hpp"
#include "interval.hpp"
#include "tree.hpp"
#include "addtree.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>

namespace veritas {

class FpMap {
public:
    using Id = int;
    using Map = std::vector<std::vector<FloatT>>;

    Map splits_; // FeatId -> {sorted split values}
    bool finalized_;

public:
    FpMap() : splits_{}, finalized_{true} {}

    inline void add(const AddTree& at) {
        for (size_t i = 0; i < at.size(); ++i)
            add(at[i]);
    }

    inline void add(const Tree& t) { add(t, t.root()); }

    inline void add(const LtSplit& s) { add(s.feat_id, s.split_value); }

    inline void add(const Tree& t, NodeId id) {
        if (t.is_internal(id)) {
            add(t.get_split(id));
            add(t, t.left(id));
            add(t, t.right(id));
        }
    }

    inline void add(BoxRef box) {
        for (const IntervalPair& pair : box)
            add(pair);
    }

    inline void add(FlatBox box) {
        for (int feat_id = 0; feat_id < static_cast<int>(box.size()); ++feat_id)
            add(feat_id, box[feat_id]);
    }

    inline void add(const IntervalPair& pair) {
        add(pair.feat_id, pair.interval);
    }

    inline void add(FeatId feat_id, Interval ival) {
        if (!ival.lo_is_unbound()) add(feat_id, ival.lo);
        if (!ival.hi_is_unbound()) add(feat_id, ival.hi);
    }

    inline void add(FeatId fid, FloatT value) {
        if (fid < 0)
            throw std::runtime_error("invalid feat_id < 0");
        while (splits_.size() < static_cast<size_t>(fid)+1)
            splits_.emplace_back(); // new empty vector
        splits_[fid].push_back(value);
        finalized_ = false;
    }

    inline void finalize() {
        // Sort and remove duplicates
        for (auto& v : splits_) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
        finalized_ = true;
    }

    inline void ensure_finalized() const {
        if (!finalized_)
            throw std::runtime_error("FpMap not finalized");
    }

    // invariant: tree.eval(val) == tree_fp.eval(transform(val))
    inline FpT transform(FeatId feat_id, FloatT val) const {
        ensure_finalized();

        const auto& vs = splits_.at(feat_id);
        auto it = std::lower_bound(vs.begin(), vs.end(), val,
                                   std::less_equal<FloatT>());
        // it might be == end()!
        FpT x = static_cast<FpT>(it - vs.begin());
        //std::cout << "transform(" << id << ", " << val << ") = " << x << " \n";
        return x;
    }

    inline FpT transform(const LtSplit& s) const {
        return transform(s.feat_id, s.split_value);
    }

    //inline IntervalFp interval(FeatId feat_id, FloatT val) const {
    //    FpT valfp = transform(feat_id, val);
    //    return {valfp, valfp+1};
    //}

    inline IntervalFp transform(FeatId id, const Interval& ival) const {
        ensure_finalized();

        const auto& vs = splits_.at(id);
        auto it_lo = std::lower_bound(vs.begin(), vs.end(), ival.lo,
                                      std::less_equal<FloatT>());
        auto it_hi = std::upper_bound(it_lo, vs.end(), ival.hi,
                                      std::less_equal<FloatT>());

        //std::cout << "lo " << ival.lo << "->"
        //          << ((it_lo == vs.end()) ? Limits<FloatT>::max : *it_lo)
        //          << ",  hi " << ival.hi << "->"
        //          << ((it_hi == vs.end()) ? Limits<FloatT>::max : *it_hi)
        //          << std::endl;

        FpT lo = static_cast<FpT>(it_lo - vs.begin());
        FpT hi = static_cast<FpT>(it_hi - vs.begin()) + 1;

        // If ival.lo is smaller than any split, then the interval is
        // unconstrained to the left
        if (lo == 0)
            lo = Limits<FpT>::min;

        // If ival.hi is larger than any split, then the interval is
        // unconstrained to the right
        if (hi > static_cast<FpT>(vs.size()))
            hi = Limits<FpT>::max;

        IntervalFp ivalfp{lo, hi};
        //std::cout << "transform(" << id << ", " << ival << ") -> " << ivalfp << std::endl;
        return ivalfp;
    }

    inline TreeFp transform(const Tree& t) const {
        TreeFp u(t.num_leaf_values());
        transform(t, t.root(), u, u.root());
        return u;
    }

    inline void transform(const Tree& t, NodeId id, TreeFp& tfp, NodeId idfp) const {
        if (t.is_leaf(id)) {
            for (int i = 0; i < t.num_leaf_values(); ++i)
                tfp.leaf_value(idfp, i) = t.leaf_value(id, i);
        } else {
            LtSplit s = t.get_split(id);
            LtSplitFp sfp{s.feat_id, transform(s)};
            tfp.split(idfp, sfp);
            transform(t, t.left(id), tfp, tfp.left(idfp));
            transform(t, t.right(id), tfp, tfp.right(idfp));
        }
    }

    inline AddTreeFp transform(const AddTree& at) const {
        AddTreeFp atfp(at.num_leaf_values(), at.get_type());
        for (int i = 0; i < at.num_leaf_values(); ++i)
            atfp.base_score(i) = at.base_score(i);
        for (size_t i = 0; i < at.size(); ++i)
            atfp.add_tree(transform(at[i]));
        return atfp;
    }

    inline FloatT itransform(FeatId id, FpT valfp) const {
        if (valfp <= 0)
            return Limits<FloatT>::min;
        const auto& vs = splits_.at(id);
        if (valfp > static_cast<FpT>(vs.size()))
            return Limits<FloatT>::max;
        return vs[valfp-1];
    }

    inline Interval itransform(FeatId id, const IntervalFp& ivalfp) const {
        return {itransform(id, ivalfp.lo), itransform(id, ivalfp.hi)};
    }

    inline void print() const {
        FeatId id = 0;
        for (const auto& vs : splits_) {
            std::cout << "FeatId" << id++ << ": ";
            for (FloatT v : vs)
                std::cout << v << ' ';
            std::cout << "\n";
        }
    }

}; // class FpMap

} // namespace veritas

#endif // VERITAS_FP_HPP
