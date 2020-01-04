#include <exception>
#include <limits>
#include <tuple>
#include <sstream>
#include <iostream>

#include "domain.h"

namespace treeck {

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
    
    std::tuple<RealDomain, RealDomain>
    RealDomain::split(FloatT value) const
    {
        return {{lo, value}, {value, hi}};
    }

    std::ostream& operator<<(std::ostream& s, const RealDomain& d)
    {
        return s << "RealDomain(" << d.lo << ", " << d.hi << ')';
    }




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

    WhereFlag
    BoolDomain::where_is(bool value) const
    {
        // [ false -------- true ]
        //   LEFT      IN_DOMAIN     if this->is_true()
        //   IN_DOMAIN     RIGHT     if this->is_false()
        if (is_everything())
            return IN_DOMAIN;
        if (is_true())
            return value ? IN_DOMAIN : LEFT;
        else
            return value ? RIGHT : IN_DOMAIN; 
    }

    bool
    BoolDomain::contains(bool value) const
    {
        return is_everything() || (value && is_true()) || (!value && is_false());
    }

    std::tuple<BoolDomain, BoolDomain>
    BoolDomain::split() const
    {
        if (!is_everything()) throw std::runtime_error("cannot split BoolDomain");
        return {false, true};
    }


    std::ostream& operator<<(std::ostream& s, const BoolDomain& d)
    {
        if (d.is_true())
            return s << "BoolDomain(true)";
        if (d.is_false())
            return s << "BoolDomain(false)";
        return s << "BoolDomain(-)"; // is_everything == true
    }




    bool
    operator==(const Domain& a, const Domain& b)
    {
        if (a.index() != b.index())
            return false;

        return visit_domain(
            [b](const RealDomain& ra) {
                const RealDomain& rb = std::get<RealDomain>(b);
                return ra.lo == rb.lo && ra.hi == rb.hi;
            }, 
            [b](const BoolDomain& ba) {
                const BoolDomain& bb = std::get<BoolDomain>(b);
                return ba.is_true() == bb.is_true()
                        && ba.is_false() == bb.is_false();
            }, a);
    }

} /* namespace treeck */
