#ifndef TREECK_DOMAIN_H
#define TREECK_DOMAIN_H

#include <tuple>
#include <ostream>

namespace treeck {

    /** An interval [lo, hi) */
    struct RealDomain {
        double lo, hi;

        RealDomain();
        RealDomain(double lo, double hi);
        
        bool is_everything() const;
        bool contains(double value) const;
        bool overlaps(const RealDomain& other) const;
        std::tuple<RealDomain, RealDomain> split(double value) const;
    };

    std::ostream& operator<<(std::ostream& s, const RealDomain& d);

} /* namespace treeck */

#endif /* TREECK_DOMAIN_H */
