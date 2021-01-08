/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_DOMAIN_H
#define VERITAS_DOMAIN_H

#include <tuple>
#include <ostream>
#include <type_traits>
#include <vector>
#include <variant>
#include <sstream>
#include <cmath>

#include "util.h"

//#ifdef __GNUC__
//#include <unistd.h>
//#include <execinfo.h>
//#include <iostream>
//#endif

namespace veritas {

    using FloatT = float;

    /**          lo                  hi
     *           [--- real domain ---]
     * ---X1--------------x2-----------------x3-----------> (real axis)
     *
     * x1 -> LEFT:      value not in domain and smaller than any value in the domain
     * x2 -> IN_DOMAIN: value in domain
     * x3 -> RIGHT:     value not in domain and larger than any value in the domain
     */
    enum WhereFlag {
        LEFT      = -1,  // x1: value lies to the left of the domain
        IN_DOMAIN = 0,   // x2: value lies in the domain
        RIGHT     = 1    // x3: value lies to the right of the domain
    };

    // real domain
    struct RealDomain {
        FloatT lo, hi;

        inline RealDomain()
            : lo(-std::numeric_limits<FloatT>::infinity())
            , hi(std::numeric_limits<FloatT>::infinity()) {}

        inline RealDomain(FloatT lo, FloatT hi) : lo(lo), hi(hi) // inclusive!
        {
            if (lo > hi)
            {
                //#ifdef __GNUC__
                //// print backtrace if gcc
                //void *array[10];
                //size_t size;
                //size = backtrace(array, 10);
                //backtrace_symbols_fd(array, size, STDERR_FILENO);
                //std::cerr << "\n";
                //#endif

                std::stringstream s;
                s << "Domain<real> error: lo > hi: [" << lo << ", " << hi << "]";
                throw std::invalid_argument(s.str());
            }
        }

        static RealDomain from_lo(FloatT lo);
        static RealDomain from_hi_inclusive(FloatT hi);
        static RealDomain from_hi_exclusive(FloatT hi);
        static inline RealDomain inclusive(FloatT lo, FloatT hi) { return {lo, hi}; }
        static inline RealDomain exclusive(FloatT lo, FloatT hi)
        { return {lo, std::nextafter(hi, -std::numeric_limits<FloatT>::infinity())}; }

        bool is_everything() const;
        WhereFlag where_is(FloatT value) const;
        WhereFlag where_is_strict(FloatT value) const;
        bool contains(FloatT value) const;
        bool contains_strict(FloatT value) const;
        RealDomain intersect(const RealDomain& o) const;
        bool overlaps(const RealDomain& other) const;
        //bool covers(const RealDomain& other) const;
        //bool covers_strict(const RealDomain& value) const;

        bool lo_is_inf() const;
        bool hi_is_inf() const;

        // consistent with Split::test, (left_dom, right_dom)
        std::tuple<RealDomain, RealDomain> split(FloatT value) const;
    };

    std::ostream& operator<<(std::ostream& s, const RealDomain& d);

    inline bool operator==(const RealDomain& a, const RealDomain& b)
    { return a.lo == b.lo && a.hi == b.hi; }

    // bool domain
    struct BoolDomain {
        short value_; // -1 unset (everything), 0 false, 1 true

        BoolDomain();
        BoolDomain(bool value);

        bool is_everything() const;
        bool is_true() const;
        bool is_false() const;

        bool contains(bool value) const;
        BoolDomain intersect(const BoolDomain& o) const;

        // consistent with Split::test, (left_dom, right_dom)
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

    std::ostream& operator<<(std::ostream& s, const Domain& d);

    VERITAS_ENABLE_TYPENAME(RealDomain);
    VERITAS_ENABLE_TYPENAME(BoolDomain);


} /* namespace veritas */

#endif /* VERITAS_DOMAIN_H */
