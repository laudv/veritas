#ifndef TREECK_DOMAIN_H
#define TREECK_DOMAIN_H

#include <tuple>
#include <ostream>
#include <vector>

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
