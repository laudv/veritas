/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_DOMAIN_HPP
#define VERITAS_DOMAIN_HPP

#include "basics.hpp"

#include <limits>
#include <sstream>
#include <cmath>
#include <tuple>
#include <vector>
#include <algorithm>

namespace veritas {

    struct Domain;
    std::ostream& operator<<(std::ostream& s, const Domain& d);

    struct Domain {
        FloatT lo, hi;

        inline Domain()
            : lo(-FLOATT_INF)
            , hi(FLOATT_INF) {}

        inline Domain(FloatT lo, FloatT hi) : lo(lo), hi(hi) // inclusive!
        {
#ifndef VERITAS_SAFETY_CHECKS_DISABLED
            if (lo > hi)
            {
                std::stringstream s;
                s << "Domain<real> error: lo > hi: [" << lo << ", " << hi << "]";
                throw std::invalid_argument(s.str());
            }
#endif
        }

        static inline Domain from_lo(FloatT lo) { return {lo, FLOATT_INF}; }
        static inline Domain from_hi_inclusive(FloatT hi) { return {-FLOATT_INF, hi}; }
        static inline Domain from_hi_exclusive(FloatT hi) { return Domain::exclusive(-FLOATT_INF, hi); }
        static inline Domain inclusive(FloatT lo, FloatT hi) { return {lo, hi}; }
        static inline Domain exclusive(FloatT lo, FloatT hi)
        { return {lo, std::nextafter(hi, -FLOATT_INF)}; }

        inline bool is_everything() const { return *this == Domain(); };
        inline bool contains(FloatT v) const { return lo >= v && v <= hi; }
        inline bool overlaps(const Domain& other) const
        {
            // [   ]
            //     [     ] edges are inclusive
            return lo <= other.hi && hi >= other.lo;
        }

        inline Domain intersect(const Domain& other) const
        {
            #ifndef VERITAS_SAFETY_CHECKS_DISABLED
            if (!overlaps(other))
            {
                std::stringstream ss;
                ss << "Domain::intersect: no overlap " << *this << " and " << other;
                throw std::runtime_error(ss.str());
            }
            #endif
            return { std::max(lo, other.lo), std::min(hi, other.hi) };
        }

        inline bool lo_is_inf() const { return std::isinf(lo); }
        inline bool hi_is_inf() const { return std::isinf(hi); }

        // consistent with LtSplit: strictly less than, left domain exclusive
        inline std::tuple<Domain, Domain> split(FloatT value) const
        {
            return {
                Domain::exclusive(lo, value),
                Domain::inclusive(value, hi)
            };
        }

        inline bool operator==(const Domain& other) const {
            return lo == other.lo && hi == other.hi;
        }
    };

    inline std::ostream& operator<<(std::ostream& s, const Domain& d)
    {
        if (d.is_everything())
            return s << "Dom(R)";
        if (d.hi_is_inf())
            return s << "Dom(>=" << d.lo << ')';
        if (d.lo_is_inf())
            return s << "Dom(< " << d.hi << ')';
        return s << "Dom(" << d.lo << ',' << d.hi << ')';
    }




    struct DomainPair {
        FeatId feat_id;
        Domain domain;
    };

    /** A sorted list of pairs */
    using Box = std::vector<DomainPair>;

    inline void refine_box(Box& doms, FeatId feat_id, const Domain& dom)
    {
        Domain new_dom;
        auto it = std::find_if(doms.begin(), doms.end(), [feat_id](const DomainPair& p)
                { return p.feat_id == feat_id; });

        if (it != doms.end())
            new_dom = it->domain;
        new_dom = new_dom.intersect(dom);

        if (it == doms.end())
        {
            doms.push_back({ feat_id, new_dom });
            for (int i = doms.size() - 1; i > 0; --i) // ids sorted
                if (doms[i-1].feat_id > doms[i].feat_id)
                    std::swap(doms[i-1], doms[i]);
            //std::sort(workspace_.begin(), workspace_.end(),
            //        [](const DomainPair& a, const DomainPair& b) {
            //            return a.first < b.first;
            //        })
        }
        else
        {
            it->domain = new_dom;
        }
    }

    /** An iterator of `Domain`s sorted asc by their feature ids
     *  This represents a hypercube in the input space. */
    class BoxRef {
    public:
        using const_iterator = const DomainPair *;

    private:
        const_iterator begin_, end_;

    public:
        inline BoxRef(const_iterator b, const_iterator e) : begin_(b), end_(e) {}
        inline BoxRef(const Box& d) : BoxRef(nullptr, nullptr)
        {
            if (d.size() != 0)
            {
                begin_ = &d[0];
                end_ = begin_ + d.size();
            }
        }
        inline static BoxRef null_box() { return {nullptr, nullptr}; }
        inline const_iterator begin() const { return begin_; }
        inline const_iterator end() const { return end_; }

        inline bool overlaps(const BoxRef& other) const
        {
            auto it0 = begin_;
            auto it1 = other.begin_;

            while (it0 != end_ && it1 != other.end_)
            {
                if (it0->feat_id == it1->feat_id)
                {
                    if (!it0->domain.overlaps(it1->domain))
                        return false;
                    ++it0; ++it1;
                }
                else if (it0->feat_id < it1->feat_id) ++it0;
                else ++it1;
            }

            return true;
        }

        inline size_t size() const { return end_ - begin_; }
    };

    inline
    std::ostream&
    operator<<(std::ostream& s, const BoxRef& box)
    {
        s << "Box { ";
        for (auto&& [id, dom] : box)
            s << id << ":" << dom << " ";
        s << '}';
        return s;
    }


} // namespace veritas


#endif // VERITAS_DOMAIN_HPP
