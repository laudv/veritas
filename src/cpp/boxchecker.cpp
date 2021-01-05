/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "boxchecker.h"
#include <cmath>
#include <iomanip>

namespace veritas {

#define RETURN_IF_INVALID(res) if ((res) == box_checker::INVALID) { return box_checker::INVALID; }
#define CHECK_DOM_BOUNDARY(v) if (std::isnan(v)) { return box_checker::INVALID; }

namespace box_checker {

    static FloatT nextf(FloatT f)
    { return std::nextafter(f, std::numeric_limits<FloatT>::infinity()); }

    static FloatT prevf(FloatT f)
    { return std::nextafter(f, -std::numeric_limits<FloatT>::infinity()); }


    UpdateResult
    Sum::update(DomainT& self, DomainT& ldom, DomainT& rdom) const
    {
        std::cout << "SUM1 " << self << " = " << ldom << " + " << rdom << ";" << std::endl;
        DomainT new_self = DomainT(
                std::max(self.lo, prevf(ldom.lo+rdom.lo)),
                std::min(self.hi, prevf(ldom.hi+rdom.hi)));
        DomainT new_ldom = DomainT(
                std::max(ldom.lo, self.lo-rdom.lo),
                std::min(ldom.hi, self.hi-rdom.hi));
        DomainT new_rdom = DomainT(
                std::max(rdom.lo, self.lo-ldom.lo),
                std::min(rdom.hi, self.hi-ldom.hi));
        UpdateResult res = static_cast<UpdateResult>(
                !(self == new_self
                && ldom == new_ldom
                && rdom == new_rdom));

        std::cout << "SUM2 " << new_self << " = " << new_ldom << " + " << new_rdom << ";" << std::endl;
        std::cout << std::setprecision(20) << new_rdom.lo << ", " << new_rdom.hi << std::endl;

        self = new_self;
        ldom = new_ldom;
        rdom = new_rdom;

        return res;
    }

    UpdateResult
    Prod::update(DomainT& self, DomainT& ldom, DomainT& rdom) const
    {
        std::runtime_error("unimplemented");
        return INVALID;
    }

    UpdateResult
    Eq::update(DomainT& ldom, DomainT& rdom) const
    {
        // L == R -> share the same domain
        FloatT new_lo = std::max(ldom.lo, rdom.lo);
        FloatT new_hi = std::min(ldom.hi, rdom.hi);
        CHECK_DOM_BOUNDARY(new_lo);
        CHECK_DOM_BOUNDARY(new_hi);
        UpdateResult res = static_cast<UpdateResult>(
               (ldom.lo != new_lo || ldom.hi != new_hi)
            || (rdom.lo != new_lo || rdom.hi != new_hi));
        ldom.lo = new_lo;
        rdom.lo = new_lo;
        ldom.hi = new_hi;
        rdom.hi = new_hi;
        return res;
    }

    UpdateResult
    LtEq::update(DomainT& ldom, DomainT& rdom) const
    {
        // LEFT <= RIGHT
        FloatT new_lo = std::max(ldom.lo, rdom.lo);
        FloatT new_hi = std::min(ldom.hi, rdom.hi);
        CHECK_DOM_BOUNDARY(new_lo);
        CHECK_DOM_BOUNDARY(new_hi);
        UpdateResult res = static_cast<UpdateResult>(
               (ldom.lo != new_lo || ldom.hi != new_hi)
            || (rdom.lo != new_lo || rdom.hi != new_hi));
        rdom.lo = new_lo;
        ldom.hi = new_hi;
        return res;
    }

} /* namespace box_checker */

    BoxChecker::BoxChecker(int max_id) : max_id_(max_id), exprs_(), comps_()
    {
        // provide space for the attributes
        box_checker::AnyExpr expr { };
        exprs_.resize(max_id_, expr);
    }

    int
    BoxChecker::add_const(FloatT value)
    {
        box_checker::AnyExpr e;
        //FloatT s = 0.00001*((0.0<value) - (value<0.0)) + 0.00001*(value==0.0);
        e.dom = DomainT(value, std::nextafter(value,
                    std::numeric_limits<FloatT>::infinity()));
        e.tag = box_checker::AnyExpr::VAR;
        e.var = {};
        exprs_.push_back(e);
        return exprs_.size() - 1;
    }

    int
    BoxChecker::add_sum(int left, int right)
    {
        box_checker::AnyExpr e;
        e.tag = box_checker::AnyExpr::SUM;
        e.sum = {left, right};
        exprs_.push_back(e);
        return exprs_.size() - 1;
    }

    int
    BoxChecker::add_prod(int left, int right)
    {
        box_checker::AnyExpr e;
        e.tag = box_checker::AnyExpr::PROD;
        e.prod = {left, right};
        exprs_.push_back(e);
        return exprs_.size() - 1;
    }

    void
    BoxChecker::add_eq(int left, int right)
    {
        box_checker::AnyComp c;
        c.left = left;
        c.right = right;
        c.comp.eq = {};
        comps_.push_back(c);
    }

    void
    BoxChecker::add_lteq(int left, int right)
    {
        box_checker::AnyComp c;
        c.left = left;
        c.right = right;
        c.comp.lteq = {};
        comps_.push_back(c);
        return;
    }



    void
    BoxChecker::copy_from_workspace(const std::vector<DomainPair>& workspace)
    {
        // workspace is ordered by .first (attribute id)

        size_t j = 0;
        for (int i = 0; i < max_id_; ++i)
        {
            if (j < workspace.size() && workspace[j].first == i)
            {
                exprs_[i].dom = workspace[j].second;
                ++j;
            }
            else // no domain present in workspace for attribute i, set to anything
            {
                exprs_[i].dom = RealDomain();
            }
        }
    }

    void
    BoxChecker::copy_to_workspace(std::vector<DomainPair>& workspace) const
    {
        size_t j = 0;
        size_t sz = workspace.size();
        for (int i = 0; i < max_id_; ++i)
        {
            if (j < sz && workspace[j].first == i)
            {
                workspace[j].second = exprs_[i].dom;
                ++j;
            }
            else if (!exprs_[i].dom.is_everything())
            {
                workspace.push_back({i, exprs_[i].dom}); // add new domain to workspace
            }
        }

        // sort newly added domain ids, if any
        if (sz < workspace.size())
        {
            std::sort(workspace.begin(), workspace.end(),
                    [](const DomainPair& a, const DomainPair& b) {
                        return a.first < b.first;
                    });
        }
    }

    box_checker::UpdateResult 
    aggregate_update_result(std::initializer_list<box_checker::UpdateResult> l)
    {
        box_checker::UpdateResult res = box_checker::UNCHANGED;
        for (auto r : l)
        {
            if (r == box_checker::INVALID)
                return box_checker::INVALID;
            if (r == box_checker::UPDATED)
                res = box_checker::UPDATED;
        };
        return res;
    }

    box_checker::UpdateResult
    BoxChecker::update()
    {
        box_checker::UpdateResult res = box_checker::UNCHANGED;
        for (const auto &c : comps_)
        {
            res = aggregate_update_result({res, update_comp(c)});
            RETURN_IF_INVALID(res);
        }
        return res;
    }

    box_checker::UpdateResult
    BoxChecker::update_comp(const box_checker::AnyComp &c)
    {
        box_checker::AnyExpr& left = exprs_[c.left];
        box_checker::AnyExpr& right = exprs_[c.right];

        box_checker::UpdateResult comp_res = box_checker::UNCHANGED;

        switch (c.tag)
        {
        case box_checker::AnyComp::EQ:
            std::cout << "EQ" << std::endl;
            comp_res = c.comp.eq.update(left.dom, right.dom);
            break;
        case box_checker::AnyComp::LTEQ:
            comp_res = c.comp.lteq.update(left.dom, right.dom);
            break;
        };

        RETURN_IF_INVALID(comp_res);

        auto left_res = update_expr(left);
        auto right_res = update_expr(right);

        return aggregate_update_result({comp_res, left_res, right_res});
    }

    box_checker::UpdateResult
    BoxChecker::update_expr(box_checker::AnyExpr &e)
    {
        box_checker::UpdateResult res = box_checker::UNCHANGED;

        switch (e.tag)
        {
        case box_checker::AnyExpr::VAR:
            // leaf -> don't need to do anything
            break;
        case box_checker::AnyExpr::SUM: {
            box_checker::AnyExpr& left = exprs_[e.sum.left];
            box_checker::AnyExpr& right = exprs_[e.sum.right];
            auto sum_res = e.sum.update(e.dom, left.dom, right.dom);
            RETURN_IF_INVALID(sum_res);
            auto left_res = update_expr(left);
            auto right_res = update_expr(right);
            res = aggregate_update_result({sum_res, left_res, right_res});
            break;
        }
        case box_checker::AnyExpr::PROD: {
            box_checker::AnyExpr& left = exprs_[e.prod.left];
            box_checker::AnyExpr& right = exprs_[e.prod.right];
            auto prod_res = e.prod.update(e.dom, left.dom, right.dom);
            RETURN_IF_INVALID(prod_res);
            auto left_res = update_expr(left);
            auto right_res = update_expr(right);
            res = aggregate_update_result({prod_res, left_res, right_res});
            break;
        }};

        return res;
    }

} /* namespace veritas */
