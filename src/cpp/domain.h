#ifndef TREECK_DOMAIN_H
#define TREECK_DOMAIN_H

#include <tuple>
#include <ostream>
#include <type_traits>
#include <vector>
#include <variant>

#include "util.h"

namespace treeck {

    using FloatT = float;

    /**          lo                  hi
     *           [--- real domain ---)
     * ---X1--------------x2-----------------x3-----------> (real axis)
     *
     * x1 -> LEFT:      value not in domain and smaller than any value in the domain
     * x2 -> IN_DOMAIN: value in domain
     * x3 -> RIGHT:     value not in domain and larger than any value in the domain (also value==hi)
     */
    enum WhereFlag {
        LEFT      = -1,  // x1: value lies to the left of the domain
        IN_DOMAIN = 0,   // x2: value lies in the domain
        RIGHT     = 1    // x3: value lies to the right of the domain
    };

    // real domain
    struct RealDomain {
        FloatT lo, hi;

        RealDomain();
        RealDomain(FloatT lo, FloatT hi);
        RealDomain(FloatT value, bool is_lo);

        bool is_everything() const;
        WhereFlag where_is(FloatT value) const;
        WhereFlag where_is_strict(FloatT value) const;
        bool contains(FloatT value) const;
        bool contains_strict(FloatT value) const;
        bool overlaps(const RealDomain& other) const;
        bool covers(const RealDomain& other) const;
        bool covers_strict(const RealDomain& value) const;

        std::tuple<RealDomain, RealDomain> split(FloatT value) const;
    };

    std::ostream& operator<<(std::ostream& s, const RealDomain& d);

    // bool domain
    struct BoolDomain {
        short value_; // -1 unset (everything), 0 false, 1 true

        BoolDomain();
        BoolDomain(bool value);

        bool is_everything() const;
        bool is_true() const;
        bool is_false() const;

        bool contains(bool value) const;

        std::tuple<BoolDomain, BoolDomain> split() const;
    };

    std::ostream& operator<<(std::ostream& s, const BoolDomain& d);

    using Domain = std::variant<RealDomain, BoolDomain>;

    template <typename RealF, typename BoolF>
    static
    std::enable_if_t<std::is_same_v<
            std::invoke_result_t<RealF, const RealDomain&>,
            std::invoke_result_t<BoolF, const BoolDomain&>>,
        std::invoke_result_t<RealF, const RealDomain&>>
    visit_domain(RealF&& f1, BoolF&& f2, const Domain& dom)
    {
        return std::visit([f1, f2](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, RealDomain>)
                return f1(arg);
            else if constexpr (std::is_same_v<T, BoolDomain>)
                return f2(arg);
            else 
                static_assert(util::always_false<T>::value, "non-exhaustive visit_domain");
        }, dom);
    }

    bool operator==(const Domain& a, const Domain& b);

} /* namespace treeck */

#endif /* TREECK_DOMAIN_H */
