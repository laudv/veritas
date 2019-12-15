#include <exception>
#include <limits>
#include <tuple>
#include <sstream>
#include <iostream>

#include "domain.h"

namespace treeck {

    RealDomain::RealDomain()
        : lo(-std::numeric_limits<double>::infinity())
        , hi(std::numeric_limits<double>::infinity()) {}

    RealDomain::RealDomain(double lo, double hi)
        : lo(lo)
        , hi(hi)
    {
        if (lo >= hi)
        {
            std::stringstream s;
            s << "RealDomain Error: lo >= hi: [" << lo << ", " << hi << ")";
            throw std::invalid_argument(s.str());
        }
    }

    bool
    RealDomain::is_everything() const
    {
        return lo == -std::numeric_limits<double>::infinity()
            && hi == std::numeric_limits<double>::infinity();
    }

    WhereFlag
    RealDomain::where_is(double value) const
    {
        if (hi <= value) // hi is excluded from the domain
            return WhereFlag::RIGHT;
        else if (lo > value) // lo is included in the domain
            return WhereFlag::LEFT;
        return WhereFlag::IN_DOMAIN;
    }

    WhereFlag
    RealDomain::where_is_strict(double value) const
    {
        if (hi <= value)
            return WhereFlag::RIGHT;
        else if (lo >= value) // note <= instead of <
            return WhereFlag::LEFT;
        return WhereFlag::IN_DOMAIN; // does not include lo, hi
    }

    bool
    RealDomain::contains(double value) const
    {
        return where_is(value) == WhereFlag::IN_DOMAIN;
    }

    bool
    RealDomain::contains_strict(double value) const
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
    RealDomain::split(double value) const
    {
        return std::make_tuple(
                RealDomain(this->lo, value),
                RealDomain(value, this->hi));
    }

    std::ostream&
    operator<<(std::ostream& s, const RealDomain& d)
    {
        return s << "RealDomain(" << d.lo << ", " << d.hi << ')';
    }


    Domains::Domains() : domains_{} {}
    Domains::Domains(vec_t v) : domains_(v) {}
    size_t Domains::size() const { return domains_.size(); }
    void Domains::resize(size_t size) { domains_.resize(size); }
    Domains::iterator Domains::begin() { return domains_.begin(); }
    Domains::iterator Domains::end() { return domains_.end(); }
    Domains::const_iterator Domains::begin() const { return domains_.cbegin(); }
    Domains::const_iterator Domains::end() const { return domains_.cend(); }
    Domains::const_iterator Domains::cbegin() const { return domains_.cbegin(); }
    Domains::const_iterator Domains::cend() const { return domains_.cend(); }

    const RealDomain& Domains::operator[](size_t i) const { return domains_.at(i); }
    RealDomain& Domains::operator[](size_t i) { return domains_.at(i); }

    const Domains::vec_t& Domains::vec() const { return domains_; }

} /* namespace treeck */
