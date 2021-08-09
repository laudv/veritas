/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "constraints.hpp"

namespace veritas {

    UpdateResult
    Sum::update(Domain& self, Domain& ldom, Domain& rdom)
    {
        std::cout << "SUM0 "
            << "self: " << self
            << ", l: " << ldom
            << ", r: " << rdom
            << std::endl;

        // self = ldom + rdom
        FloatT new_self_lo, new_self_hi;
        FloatT new_ldom_lo, new_ldom_hi;
        FloatT new_rdom_lo, new_rdom_hi;
        new_self_lo = std::max(self.lo, ldom.lo+rdom.lo);
        new_self_hi = std::min(self.hi, ldom.hi+rdom.hi);
        new_ldom_lo = std::max(ldom.lo, self.lo-rdom.hi);
        new_ldom_hi = std::min(ldom.hi, self.hi-rdom.lo);
        new_rdom_lo = std::max(rdom.lo, self.lo-ldom.hi),
        new_rdom_hi = std::min(rdom.hi, self.hi-ldom.lo);

        if (new_self_lo > new_self_hi
                || new_ldom_lo > new_ldom_hi
                || new_rdom_lo > new_rdom_hi)
            return INVALID;

        UpdateResult res = static_cast<UpdateResult>(
                !(self.lo == new_self_lo && self.hi == new_self_hi
                && ldom.lo == new_ldom_lo && ldom.hi == new_ldom_hi
                && rdom.lo == new_rdom_lo && rdom.hi == new_rdom_hi));

        self = {new_self_lo, new_self_hi};
        ldom = {new_ldom_lo, new_ldom_hi};
        rdom = {new_rdom_lo, new_rdom_hi};

        std::cout << "SUM1 "
            << "self: " << self
            << ", l: " << ldom
            << ", r: " << rdom
            << " res=" << res
            << std::endl;

        return res;
    }

    UpdateResult
    Eq::update(Domain& ldom, Domain& rdom)
    {
        // L == R -> share the same domain
        FloatT new_lo = std::max(ldom.lo, rdom.lo);
        FloatT new_hi = std::min(ldom.hi, rdom.hi);

        //std::cout << "EQ ldom " << ldom << ", rdom " << rdom << std::endl;

        if (new_lo > new_hi)
            return INVALID;

        UpdateResult res = static_cast<UpdateResult>(
               (ldom.lo != new_lo || ldom.hi != new_hi)
            || (rdom.lo != new_lo || rdom.hi != new_hi));
        ldom.lo = new_lo;
        rdom.lo = new_lo;
        ldom.hi = new_hi;
        rdom.hi = new_hi;

        //std::cout << "-- ldom " << ldom << ", rdom " << rdom << std::endl;

        return res;
    }

    UpdateResult
    LtEq::update(Domain& ldom, Domain& rdom)
    {
        // LEFT <= RIGHT
        FloatT new_lo = std::max(ldom.lo, rdom.lo);
        FloatT new_hi = std::min(ldom.hi, rdom.hi);

        //std::cout << "LTEQ ldom " << ldom << ", rdom " << rdom << std::endl;

        if (ldom.lo > new_hi || new_lo > rdom.hi)
            return INVALID;

        UpdateResult res = static_cast<UpdateResult>(
               (ldom.lo != new_lo || ldom.hi != new_hi)
            || (rdom.lo != new_lo || rdom.hi != new_hi));

        ldom = {ldom.lo, new_hi};
        rdom = {new_lo, rdom.hi};

        //std::cout << "---- ldom " << ldom << ", rdom " << rdom << std::endl;

        return res;
    }

    void
    ConstraintPropagator::copy_from_box(const Box& box)
    {
        /*
        size_t i = 0, j = 0;
        for (int i = 0; i < num_features_; ++i) // box is sorted by item.feat_id
        {
            AnyExpr e;
            e.tag = AnyExpr::VAR;
            if (j < box.size() && box[j].feat_id == i)
            {
                e.dom = box[j].domain;
                ++j;
            }
            exprs_.push_back(e);
        }

        // add dependent_exprs
        exprs_.insert(exprs_.end(), dependent_exprs_.begin(), dependent_exprs_.end());
        */
    }

    void
    ConstraintPropagator::copy_to_box(Box& workspace) const
    {
        /*
        size_t j = 0;
        size_t sz = workspace.size();
        for (int i = 0; i <= max_feat_id_; ++i)
        {
            if (j < sz && workspace[j].feat_id == i)
            {
                workspace[j].domain = exprs_[i].dom;
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
                        return a.feat_id < b.feat_id;
                    });
        }
        */
    }

    UpdateResult 
    ConstraintPropagator::aggregate_update_result(std::initializer_list<UpdateResult> l)
    {
        UpdateResult res = UNCHANGED;
        for (auto r : l)
        {
            if (r == INVALID)
                return INVALID;
            if (r == UPDATED)
                res = UPDATED;
        };
        return res;
    }

    void
    ConstraintPropagator::add_eq(int left, int right)
    {
        AnyComp c;
        c.left = left;
        c.right = right;
        c.comp.eq = {};
        c.tag = AnyComp::EQ;
        comps_.push_back(c);
    }

    void
    ConstraintPropagator::add_lteq(int left, int right)
    {
        AnyComp c;
        c.left = left;
        c.right = right;
        c.comp.lteq = {};
        c.tag = AnyComp::LTEQ;
        comps_.push_back(c);
    }

    int
    ConstraintPropagator::add_var(FeatId feat_id)
    {
        if (exprs_.size() > 0 && exprs_.back().tag != AnyExpr::VAR)
            throw std::runtime_error("add all vars first!");
        AnyExpr e;
        e.tag = AnyExpr::VAR;
        e.var.feat_id = feat_id;
        exprs_.push_back(e);
        return exprs_.size() - 1;
    }

    int
    ConstraintPropagator::add_const(FloatT value)
    {
        AnyExpr e;
        e.tag = AnyExpr::CONST;
        e.constant = {value};
        exprs_.push_back(e);
        return exprs_.size() - 1;
    }

    int
    ConstraintPropagator::add_sum(int left, int right)
    {
        AnyExpr e;
        e.tag = AnyExpr::SUM;
        e.sum = {left, right};
        exprs_.push_back(e);
        return exprs_.size() - 1;
    }


} // namespace veritas
