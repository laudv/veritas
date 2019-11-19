#include <exception>
#include <limits>
#include <tuple>
#include <sstream>

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

    ContainsFlag
    RealDomain::contains(double value) const
    {
        if (hi < value)
            return ContainsFlag::LARGER;
        else if (lo >= value)
            return ContainsFlag::SMALLER;
        return ContainsFlag::IN;
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

    const RealDomain& Domains::operator[](size_t i) const { return domains_[i]; }
    RealDomain& Domains::operator[](size_t i) { return domains_[i]; }

    const Domains::vec_t& Domains::vec() const { return domains_; }

} /* namespace treeck */
