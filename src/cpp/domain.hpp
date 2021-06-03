/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_DOMAIN_HPP
#define VERITAS_DOMAIN_HPP

#include <limits>
#include <sstream>
#include <cmath>
#include <tuple>

namespace veritas {

    using FloatT = float;
    static const FloatT FLOATT_INF = std::numeric_limits<FloatT>::infinity();

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


} // namespace veritas


#endif // VERITAS_DOMAIN_HPP
