/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_CONSTRAINTS_HPP
#define VERITAS_CONSTRAINTS_HPP

#include "domain.hpp"
#include <vector>
#include <iostream>

namespace veritas {

    enum UpdateResult { UNCHANGED = 0, UPDATED = 1, INVALID = 2 };

    struct Eq {
        UpdateResult update(Domain& left_dom, Domain& right_dom) const;
    };

    struct LtEq {
        UpdateResult update(Domain& left_dom, Domain& right_dom) const;
    };

    struct AnyComp {
        FeatId left, right;
        enum { EQ, LTEQ } tag;
        union {
            Eq eq;
            LtEq lteq;
        } comp;
    };

    class ConstraintPropagator  {
        //std::vector<AnyExpr> exprs_;
        std::vector<AnyComp> comps_;
        //std::vector<BinaryConstraint> bin_constraints_;

    public:
        void add_eq(int left, int right);
        void add_lteq(int left, int right);

        template <typename F>
        bool check(BoxRef box, F prop_f)
        {

            return true;
        }

    }; // class ConstraintPropagator

} // namespace veritas

#endif // VERITAS_CONSTRAINTS_HPP

