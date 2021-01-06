/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_BOX_CHECKER_H
#define VERITAS_BOX_CHECKER_H

#include "domain.h"
#include "graph.h" // DomainT, DomainPair

#include <vector>
#include <cfloat>

namespace veritas {
namespace box_checker {

    enum UpdateResult { UNCHANGED = 0, UPDATED = 1, INVALID = 2 };

    struct Var {};

    struct Sum {
        int left, right;
        static UpdateResult update(DomainT& self, DomainT& left_dom, DomainT& right_dom);
    };

    struct Sub {
        int left, right;
        // uses Sum::update(left, self, right)
    };

    struct Prod {
        int left, right;
        static UpdateResult update(DomainT& self, DomainT& left_dom, DomainT& right_dom);
    };

    struct Div {
        int left, right; // left / right = self <=> left = self * right
        // uses Prod::update(left, self, right)
    };

    struct Pow2 {
        int arg;
        static UpdateResult update(DomainT& self, DomainT& arg_dom);
    };

    struct Sqrt {
        int arg;

        // does not reuse Pow2::update because of negative
        // (!) a = b² <=> b = +/-sqrt(a)
        // a = sqrt(b) <=> b = a² -> both a and b positive
        static UpdateResult update(DomainT& self, DomainT& arg_dom);
    };

    struct AnyExpr {
        DomainT dom;
        enum { VAR, SUM, SUB, PROD, DIV, POW2, SQRT } tag;
        union {
            Var var;
            Sum sum;
            Sub sub;
            Prod prod;
            Div div;
            Pow2 pow2;
            Sqrt sqrt;
        };
    };


    struct Eq {
        UpdateResult update(DomainT& left_dom, DomainT& right_dom) const;
    };

    struct LtEq {
        UpdateResult update(DomainT& left_dom, DomainT& right_dom) const;
    };

    struct AnyComp {
        int left, right;
        enum { EQ, LTEQ } tag;
        union {
            Eq eq;
            LtEq lteq;
        } comp;
    };

} /* namespace box_checker */

    class BoxChecker {
        int max_id_;
        std::vector<box_checker::AnyExpr> exprs_;
        std::vector<box_checker::AnyComp> comps_;
    public:
        BoxChecker(int max_id);

        int add_const(FloatT value);
        int add_sum(int left, int right);
        int add_sub(int left, int right);
        int add_prod(int left, int right);
        int add_div(int left, int right);
        int add_pow2(int arg);
        int add_sqrt(int arg);

        void add_eq(int left, int right);
        void add_lteq(int left, int right);

        DomainT get_expr_dom(int expr_id) const;

        /** Copy the domains for the Vars from the workspace */
        void copy_from_workspace(const std::vector<DomainPair>& workspace);

        /** Copy the potentially updated domains back into the workspace */
        void copy_to_workspace(std::vector<DomainPair>& workspace) const;

        /** Do one step in the update domain propagation process */
        box_checker::UpdateResult update();

    private:
        box_checker::UpdateResult update_comp(const box_checker::AnyComp &c);
        box_checker::UpdateResult update_expr(box_checker::AnyExpr &e);
    };

} /* namespace veritas */

#endif /* VERITAS_BOX_CHECKER_H */
