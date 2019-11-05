#ifndef TREECK_DOMAIN_H
#define TREECK_DOMAIN_H

namespace treeck {

    /** An interval [lo, hi) */
    struct RealDomain {
        double lo, hi;

        RealDomain();
        RealDomain(double lo, double hi);
        
        bool contains(double value) const;
        bool overlaps(const RealDomain& other) const;
        std::tuple<RealDomain, RealDomain> split(double value) const;
    };

} /* namespace treeck */

#endif /* TREECK_DOMAIN_H */
