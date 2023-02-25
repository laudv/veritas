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
#include "domain.hpp"
#include "tree.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>

namespace veritas {

    using FpT = int;

    static const FpT FpT_MIN = std::numeric_limits<FpT>::min();
    static const FpT FpT_MAX = std::numeric_limits<FpT>::max();

    struct FpDomain {
        FpT lo;
        FpT hi;

        inline FpDomain(FpT l=FpT_MIN, FpT h=FpT_MAX) : lo(l), hi(h) {}

        inline bool is_everything() const { return lo == FpT_MIN && hi == FpT_MAX; }
        inline bool overlaps(const FpDomain& other)
        {
            return lo < other.hi && hi > other.lo;
        }
    };

    class FpMap {
    public:
        using Id = int;
        using Map = std::vector<std::vector<FloatT>>;

        Map edges_;
        bool finalized_;

    public:
        FpMap() : edges_{}, finalized_{true} {}

        inline void add(const AddTree& at) { for (auto t : at) add(t); }
        inline void add(const Tree& t) { add(t.root_const()); }
        inline void add(Tree::ConstRef n) { if (n.is_internal()) add(n.get_split()); }
        inline void add(const LtSplit& s) { add(s.feat_id, s.split_value); }
        inline void add(FeatId fid, FloatT value)
        {
            if (fid < 0)
                throw std::runtime_error("invalid feat_id < 0");
            while (edges_.size() < static_cast<size_t>(fid))
                edges_.push_back({-FLOATT_INF, FLOATT_INF});
            edges_[fid].push_back(value);
            finalized_ = false;
        }

        inline void finalize()
        {
            for (const auto& v : edges_)
                std::sort(v.begin(), v.end());
            finalized_ = true;
        }

        inline FpDomain transform(const Domain& dom) const
        {
            if (!finalized_)
                throw std::runtime_error("FpMap not finalized");

            // TODO
            throw std::runtime_error("not implemented");
            return {};
        }

        inline void transform(const AddTree& at) const
        {

        }
    };

} // namespace veritas

#endif // VERITAS_FP_HPP
