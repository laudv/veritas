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

    struct UnitVec2 {
        int a, b;
        static UpdateResult update(DomainT& self, DomainT& a_dom, DomainT& b_dom);
    };

    struct AnyExpr {
        DomainT dom;
        enum { VAR, SUM, SUB, PROD, DIV, POW2, SQRT, UNIT_VEC2 } tag;
        union {
            Var var;
            Sum sum;
            Sub sub;
            Prod prod;
            Div div;
            Pow2 pow2;
            Sqrt sqrt;
            UnitVec2 unit_vec2;
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

    struct Binary {
        std::vector<int> ids;
        int k;
    };

    struct BinaryConstraint {
        enum { AT_LEAST_K = 1, AT_MOST_K = 2, K_OUT_OF_N = 1 | 2 } tag;
        std::vector<int> ids;
        int k;
    };


} /* namespace box_checker */

    class RepeatedBoxChecker;

    class BoxChecker {
        int num_vars_;
        int max_num_updates_;
        std::vector<box_checker::AnyExpr> exprs_;
        std::vector<box_checker::AnyComp> comps_;
        std::vector<box_checker::BinaryConstraint> bin_constraints_;
    public:
        BoxChecker(int num_vars, int max_num_updates = 5);

        int add_const(FloatT value);
        int add_sum(int left, int right);
        int add_sub(int left, int right);
        int add_prod(int left, int right);
        int add_div(int left, int right);
        int add_pow2(int arg);
        int add_sqrt(int arg);
        int add_unit_vec2(int a, int b); // a / sqrt(a² + b²)

        void add_eq(int left, int right);
        void add_lteq(int left, int right);
        void add_at_most_k(std::vector<int> ids, int k);
        void add_at_least_k(std::vector<int> ids, int k);
        void add_k_out_of_n(std::vector<int> ids, int k, bool strict);

        DomainT get_expr_dom(int expr_id) const;

        /** Copy the domains for the Vars from the workspace */
        void copy_from_workspace(const std::vector<DomainPair>& workspace);

        /** Copy the potentially updated domains back into the workspace */
        void copy_to_workspace(std::vector<DomainPair>& workspace) const;

        /** Do one step in the update domain propagation process */
        box_checker::UpdateResult update();

        /** Do max_num_updates updates, ignore further possible updates (ie UpdateResult::CHANGED) */
        bool update(std::vector<DomainPair>& workspace);
        inline bool update(DomainStore& store) { return update(store.workspace()); }

    private:
        box_checker::UpdateResult update_comp(const box_checker::AnyComp &c);
        box_checker::UpdateResult update_expr(box_checker::AnyExpr &e);
        box_checker::UpdateResult update_bin_constraint(const box_checker::BinaryConstraint &c);
    };

} /* namespace veritas */

#endif /* VERITAS_BOX_CHECKER_H */
