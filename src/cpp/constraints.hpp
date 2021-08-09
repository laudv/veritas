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

#define RETURN_IF_INVALID(res) if ((res) == UpdateResult::INVALID) { return UpdateResult::INVALID; }

namespace veritas {

    enum UpdateResult : int { UNCHANGED = 0, UPDATED = 1, INVALID = 2 };

    struct Var {
        FeatId feat_id;
    };

    struct Const {
        FloatT value;
    };

    struct Sum {
        int left, right;
        static UpdateResult update(Domain& self, Domain& left_dom, Domain& right_dom);
    };

    struct AnyExpr {
        Domain dom;
        enum { VAR, CONST, SUM, SUB, PROD, DIV, POW2, SQRT, UNIT_VEC2 } tag;
        union {
            Var var;
            Const constant;
            Sum sum;
            /*
            Sub sub;
            Prod prod;
            Div div;
            Pow2 pow2;
            Sqrt sqrt;
            UnitVec2 unit_vec2;
            */
        };
    };


    struct Eq {
        static UpdateResult update(Domain& left_dom, Domain& right_dom);
    };

    struct LtEq {
        static UpdateResult update(Domain& left_dom, Domain& right_dom);
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
        std::vector<AnyExpr> exprs_;
        std::vector<AnyComp> comps_;
        //std::vector<BinaryConstraint> bin_constraints_;
        
        FeatId max_feat_id_;
        int num_additional_expr_; /* non-vars */

        void process_feat_id(std::initializer_list<FeatId> ids);
        void copy_from_box(const Box&);
        void copy_to_box(Box& box) const;

        UpdateResult aggregate_update_result(std::initializer_list<UpdateResult> l);

        template <typename F>
        UpdateResult update_comp(const AnyComp& comp, const F& push_box_fun)
        {
            AnyExpr& left = exprs_[comp.left];
            AnyExpr& right = exprs_[comp.right];

            UpdateResult comp_res = UNCHANGED;

            if (comp.tag == AnyComp::EQ)
                comp_res = Eq::update(left.dom, right.dom);
            else if (comp.tag == AnyComp::LTEQ)
                comp_res = LtEq::update(left.dom, right.dom);
            else throw std::runtime_error("invalid");

            RETURN_IF_INVALID(comp_res);
            auto left_res = update_expr(left, push_box_fun);
            RETURN_IF_INVALID(left_res);
            auto right_res = update_expr(right, push_box_fun);

            return aggregate_update_result({comp_res, left_res, right_res});
        }

        template <typename F>
        UpdateResult update_expr(AnyExpr& expr, const F& push_box_fun)
        {
            UpdateResult res = UNCHANGED;

            switch (expr.tag)
            {
                case AnyExpr::VAR:
                case AnyExpr::CONST:
                    push_box_fun();
                    break;
                case AnyExpr::SUM: {
                    AnyExpr& left = exprs_[expr.sum.left];
                    AnyExpr& right = exprs_[expr.sum.right];
                    UpdateResult op_res = Sum::update(expr.dom, left.dom, right.dom);
                    RETURN_IF_INVALID(op_res);
                    UpdateResult left_res = update_expr(left, push_box_fun);
                    UpdateResult right_res = update_expr(right, push_box_fun);
                    res = aggregate_update_result({op_res, left_res, right_res});
                    break;
                }
                case AnyExpr::SUB:
                    break;
                case AnyExpr::PROD:
                    break;
                case AnyExpr::DIV:
                    break;
                case AnyExpr::POW2:
                    break;
                case AnyExpr::SQRT:
                    break;
                case AnyExpr::UNIT_VEC2:
                    break;
            }

            return res;
        }

    public:

        int max_num_updates = 10;

        void add_eq(int left, int right);
        void add_lteq(int left, int right);

        int add_var(FeatId id);
        int add_const(FloatT value);
        int add_sum(int left, int right);

        template <typename F>
        void check(Box& box, const F& push_box_fun)
        {
            copy_from_box(box);

            for (int i = 0; i < max_num_updates; i++)
            {
                for (const auto& c : comps_)
                {
                    update_comp(c, [this, &box, &push_box_fun]() {
                        copy_to_box(box);
                        push_box_fun(box);
                    });
                }
            }
        }

    }; // class ConstraintPropagator

} // namespace veritas

#undef RETURN_IF_INVALID

#endif // VERITAS_CONSTRAINTS_HPP

