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
#include "interval.hpp"
#include "tree.hpp"
#include "addtree.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>

namespace veritas {

class FpMap {
public:
    using Id = int;
    using Map = std::vector<std::vector<FloatT>>;

    Map edges_; // FeatId -> {sorted split values}
    bool finalized_;

public:
    FpMap() : edges_{}, finalized_{true} {}

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

    inline void add(FeatId fid, FloatT value) {
        if (fid < 0)
            throw std::runtime_error("invalid feat_id < 0");
        while (edges_.size() < static_cast<size_t>(fid)+1)
            edges_.push_back({Limits<FloatT>::max});
        edges_[fid].push_back(value);
        finalized_ = false;
    }

    inline void finalize() {
        for (auto& v : edges_)
            std::sort(v.begin(), v.end());
        finalized_ = true;
    }

    inline void ensure_finalized() const {
        if (!finalized_)
            throw std::runtime_error("FpMap not finalized");
    }

    // invariant: tree.eval(val) == tree_fp.eval(transform(val))
    inline FpT transform(FeatId id, FloatT val) const {
        ensure_finalized();

        const auto& vs = edges_.at(id);
        auto it = std::upper_bound(vs.begin(), vs.end(), val);
        if (it == vs.end())
            throw std::runtime_error("illegal state");
        FpT x = static_cast<FpT>(it - vs.begin());
        std::cout << "map(" << id << ", " << val << ") = " << x << std::endl;
        return x;
    }

    inline FpT operator()(FeatId id, FloatT val) const {
        return transform(id, val);
    }

    inline FpT transform_exact(FeatId id, FloatT val) const {
        ensure_finalized();

        const auto& vs = edges_.at(id);
        auto it = std::lower_bound(vs.begin(), vs.end(), val);
        if (it == vs.end())
            throw std::runtime_error("illegal state");
        if (*it != val) {
            std::cout << "not exact: find " << *it << ", " << val << std::endl;
            throw std::runtime_error("not exact");
        }
        FpT x = static_cast<FpT>(it - vs.begin());
        std::cout << "map(" << id << ", " << val << ") = " << x << std::endl;
        return x;
    }

    inline FpT transform_exact(const LtSplit& s) const {
        return transform_exact(s.feat_id, s.split_value);
    }

    inline TreeFp transform(const Tree& t) const {
        TreeFp u;
        transform(t, u, t.root());
        return u;
    }

    inline void transform(const Tree& t, TreeFp& tfp, NodeId id) const {
        if (t.is_leaf(id)) {
            tfp.leaf_value(id) = t.leaf_value(id);
        } else {
            LtSplit s = t.get_split(id);
            LtSplitFp sfp{s.feat_id, transform_exact(s)};
            tfp.split(id, sfp);
            transform(t, tfp, t.left(id));
            transform(t, tfp, t.right(id));
        }
    }

    inline AddTreeFp transform(const AddTree& at) const {
        AddTreeFp atfp;
        for (size_t i = 0; i < at.size(); ++i)
            atfp.add_tree(transform(at[i]));
        return atfp;
    }

    inline void print() const {
        FeatId id = 0;
        for (const auto& vs : edges_) {
            std::cout << "FeatId" << id++ << ": ";
            for (FloatT v : vs)
                std::cout << v << ' ';
            std::cout << "\n";
        }
    }

}; // class FpMap

} // namespace veritas

#endif // VERITAS_FP_HPP
