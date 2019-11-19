#ifndef TREECK_DOMAIN_H
#define TREECK_DOMAIN_H

#include <tuple>
#include <ostream>
#include <vector>

namespace treeck {

    /**
     * ---------------------x--------------------------> (x is value)
     *      case 1       case 3          case 2
     *      [----)   |-------------)     |----)
     */
    /**
     *           [--- real domain ---)
     * ---X1--------------x2-----------------x3-----------> (real axis)
     *
     * x1 -> SMALLER: value not in domain and smaller than any value in the domain
     * x2 -> IN:      value in domain
     * x3 -> LARGER:  value not in domain and larger than any value in the domain (also value==hi)
     */
    enum ContainsFlag {
        SMALLER = -1,  // x1: value lies to the left of the domain
        IN      = 0,   // case 3: value lies in the domain
        LARGER  = 1    // case 1: value lies to the right of the domain
    };

    /** An interval [lo, hi) */
    struct RealDomain {
        double lo, hi;

        RealDomain();
        RealDomain(double lo, double hi);
        
        bool is_everything() const;
        ContainsFlag contains(double value) const;
        bool overlaps(const RealDomain& other) const;
        std::tuple<RealDomain, RealDomain> split(double value) const;
    };

    std::ostream& operator<<(std::ostream& s, const RealDomain& d);

    class Domains {
    public:
        using vec_t = std::vector<RealDomain>;
        using iterator = typename vec_t::iterator;
        using const_iterator = typename vec_t::const_iterator;

    private:
        vec_t domains_;

    public:
        Domains();
        Domains(vec_t domains);
        size_t size() const;
        void resize(size_t size);

        iterator begin();
        iterator end();
        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        const RealDomain& operator[](size_t i) const;
        RealDomain& operator[](size_t i);

        const vec_t& vec() const;
    };

} /* namespace treeck */

#endif /* TREECK_DOMAIN_H */
