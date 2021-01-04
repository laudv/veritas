/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include <cmath>
#include <algorithm>
#include <exception>
#include <limits>
#include <tuple>
#include <sstream>
#include <iostream>

#include "domain.h"

#ifdef __GNUC__
#include <unistd.h>
#include <execinfo.h>
#endif

namespace veritas {

    RealDomain::RealDomain()
        : lo(-std::numeric_limits<FloatT>::infinity())
        , hi(std::numeric_limits<FloatT>::infinity()) {}

    RealDomain::RealDomain(FloatT value, bool is_lo) : RealDomain()
    {
        if (is_lo) lo = value;
        else       hi = value;
    }

    RealDomain::RealDomain(FloatT lo, FloatT hi) : lo(lo), hi(hi)
    {
        if (lo >= hi)
        {
            #ifdef __GNUC__
            // print backtrace if gcc
            void *array[10];
            size_t size;
            size = backtrace(array, 10);
            backtrace_symbols_fd(array, size, STDERR_FILENO);
            #endif

            std::stringstream s;
            s << "Domain<real> error: lo >= hi: [" << lo << ", " << hi << ")";
            throw std::invalid_argument(s.str());
        }
    }

    bool
    RealDomain::is_everything() const
    {
        return lo == -std::numeric_limits<FloatT>::infinity()
            && hi == std::numeric_limits<FloatT>::infinity();
    }

    WhereFlag
    RealDomain::where_is(FloatT value) const
    {
        if (hi <= value) // hi is excluded from the domain
            return WhereFlag::RIGHT;
        else if (lo > value) // lo is included in the domain
            return WhereFlag::LEFT;
        return WhereFlag::IN_DOMAIN;
    }

    WhereFlag
    RealDomain::where_is_strict(FloatT value) const
    {
        if (hi <= value)
            return WhereFlag::RIGHT;
        else if (lo >= value) // note <= instead of <
            return WhereFlag::LEFT;
        return WhereFlag::IN_DOMAIN; // does not include lo, hi
    }

    bool
    RealDomain::contains(FloatT value) const
    {
        return where_is(value) == WhereFlag::IN_DOMAIN;
    }

    bool
    RealDomain::contains_strict(FloatT value) const
    {
        return where_is_strict(value) == WhereFlag::IN_DOMAIN;
    }

    bool
    RealDomain::overlaps(const RealDomain& other) const
    {
        return this->lo < other.hi && this->hi > other.lo;
    }

    RealDomain
    RealDomain::intersect(const RealDomain& o) const
    {
        if (!overlaps(o))
        {
            std::stringstream ss;
            ss << "RealDomain::intersect: no overlap " << *this << " and " << o;
            throw std::runtime_error(ss.str());
        }

        FloatT nlo = std::max(lo, o.lo);
        FloatT nhi = std::min(hi, o.hi);

        return { nlo, nhi };
    }

    bool
    RealDomain::covers(const RealDomain& other) const
    {
        return where_is(other.lo) == WhereFlag::IN_DOMAIN
            && where_is(other.hi) == WhereFlag::IN_DOMAIN;
    }

    bool
    RealDomain::covers_strict(const RealDomain& other) const
    {
        return where_is_strict(other.lo) == WhereFlag::IN_DOMAIN
            && where_is_strict(other.hi) == WhereFlag::IN_DOMAIN;
    }

    bool
    RealDomain::lo_is_inf() const
    {
        return std::isinf(lo);
    }

    bool
    RealDomain::hi_is_inf() const
    {
        return std::isinf(hi);
    }
    
    std::tuple<RealDomain, RealDomain>
    RealDomain::split(FloatT value) const
    {
        return {{lo, value}, {value, hi}};
    }

    std::ostream& operator<<(std::ostream& s, const RealDomain& d)
    {
        if (d.lo_is_inf() && d.hi_is_inf())
            return s << "Dom(R)";
        if (d.hi_is_inf())
            return s << "Dom(>=" << d.lo << ')';
        if (d.lo_is_inf())
            return s << "Dom(< " << d.hi << ')';
        return s << "Dom(" << d.lo << ',' << d.hi << ')';
    }




    BoolDomain::BoolDomain() : value_(-1) {}
    BoolDomain::BoolDomain(bool value) : value_(value ? 1 : 0) {}

    bool
    BoolDomain::is_everything() const
    {
        return value_ == -1;
    }

    bool
    BoolDomain::is_true() const
    {
        return value_ == 1;
    }

    bool
    BoolDomain::is_false() const
    {
        return value_ == 0;
    }

    bool
    BoolDomain::contains(bool value) const
    {
        return is_everything() || (value && is_true()) || (!value && is_false());
    }

    BoolDomain
    BoolDomain::intersect(const BoolDomain& o) const
    {
        if (is_everything())
            return o;
        if (o.is_everything())
            return *this;
        if (value_ != o.value_)
            throw std::runtime_error("BoolDomain::intersect: non-overlapping domain");
        return o;
    }

    std::tuple<BoolDomain, BoolDomain>
    BoolDomain::split() const
    {
        if (!is_everything()) throw std::runtime_error("cannot split BoolDomain");
        return {true, false};
    }


    std::ostream& operator<<(std::ostream& s, const BoolDomain& d)
    {
        if (d.is_true())
            return s << "Dom(true)";
        if (d.is_false())
            return s << "Dom(false)";
        return s << "Dom(B)"; // is_everything == true
    }



    std::ostream&
    operator<<(std::ostream& s, const Domain& d)
    {
        visit_domain(
            [&s](const RealDomain& d) { s << d; },
            [&s](const BoolDomain& d) { s << d; }, d);
        return s;
    }

    bool
    operator==(const Domain& a, const Domain& b)
    {
        if (a.index() != b.index())
            return false;

        return visit_domain(
            [b](const RealDomain& ra) {
                const RealDomain& rb = std::get<RealDomain>(b);
                return ra == rb;
            }, 
            [b](const BoolDomain& ba) {
                const BoolDomain& bb = std::get<BoolDomain>(b);
                return ba.is_true() == bb.is_true()
                        && ba.is_false() == bb.is_false();
            }, a);
    }

} /* namespace veritas */
