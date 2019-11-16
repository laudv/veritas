#include <exception>
#include <limits>
#include <tuple>

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
            throw std::invalid_argument("lo >= hi");
    }

    bool
    RealDomain::contains(double value) const
    {
        return this->lo <= value && value < this->hi;
    }

    bool
    RealDomain::overlaps(const RealDomain& other) const
    {
        return this->lo < other.hi && this->hi > other.lo;
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

} /* namespace treeck */
