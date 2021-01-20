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
#define INVALID_IF_NAN(v) if (std::isnan(v)) { return box_checker::INVALID; }

namespace box_checker {

    UpdateResult
    Sum::update(DomainT& self, DomainT& ldom, DomainT& rdom)
    {
        //std::cout << "SUM0 "
        //    << "self: " << self
        //    << ", l: " << ldom
        //    << ", r: " << rdom
        //    << std::endl;

        // self = ldom + rdom
        DomainT new_self, new_ldom, new_rdom;
        try
        {
            new_self = {std::max(self.lo, ldom.lo+rdom.lo),
                        std::min(self.hi, ldom.hi+rdom.hi)};
            new_ldom = {std::max(ldom.lo, self.lo-rdom.hi),
                        std::min(ldom.hi, self.hi-rdom.lo)};
            new_rdom = {std::max(rdom.lo, self.lo-ldom.hi),
                        std::min(rdom.hi, self.hi-ldom.lo)};
        }
        catch (const std::exception& e) { return INVALID; }

        UpdateResult res = static_cast<UpdateResult>(
                !(self == new_self
                && ldom == new_ldom
                && rdom == new_rdom));

        self = new_self;
        ldom = new_ldom;
        rdom = new_rdom;

        //std::cout << "SUM1 "
        //    << "self: " << self
        //    << ", l: " << ldom
        //    << ", r: " << rdom
        //    << " res=" << res
        //    << std::endl;

        return res;
    }

    UpdateResult
    Prod::update(DomainT& self, DomainT& ldom, DomainT& rdom)
    {
        // self = ldom * rdom
        // update self -> result variable
        FloatT m, M, x, contains_zero;
        m = ldom.lo * rdom.lo;
        M = m;
        x = ldom.lo * rdom.hi;
        m = std::fmin(m, x); M = std::fmax(M, x);
        x = ldom.hi * rdom.lo;
        m = std::fmin(m, x); M = std::fmax(M, x);
        x = ldom.hi * rdom.hi;
        m = std::fmin(m, x); M = std::fmax(M, x);

        DomainT new_self;
        try {
            new_self = {std::fmax(self.lo, m), std::fmin(self.hi, M)};
        }
        catch (const std::exception&) { return INVALID; }

        // update ldom -> +/- inf if denominator 0 = OK
        m = self.lo / rdom.lo;
        M = m;
        x = self.lo / rdom.hi;
        m = std::fmin(m, x); M = std::fmax(M, x);
        x = self.hi / rdom.lo;
        m = std::fmin(m, x); M = std::fmax(M, x);
        x = self.hi / rdom.hi;
        m = std::fmin(m, x); M = std::fmax(M, x);

        contains_zero = (rdom.lo < 0.0 && rdom.hi > 0.0); // (!) if 0.0 in rdom, div can grow to +/- inf!
        m = std::fmin(m, -contains_zero / 0.0); // NaN if not contains 0.0 -> chooses m, else -inf
        M = std::fmax(M, contains_zero / 0.0); // NaN if not contains 0.0 -> chooses M, else +inf

        DomainT new_ldom;
        try {
            new_ldom = {std::fmax(ldom.lo, m), std::fmin(ldom.hi, M)};
        }
        catch (const std::exception& e) { return INVALID; }

        // update rdom -> +/- inf if denominator 0 = OK
        m = self.lo / ldom.lo;
        M = m;
        x = self.lo / ldom.hi;
        m = std::fmin(m, x); M = std::fmax(M, x);
        x = self.hi / ldom.lo;
        m = std::fmin(m, x); M = std::fmax(M, x);
        x = self.hi / ldom.hi;
        m = std::fmin(m, x); M = std::fmax(M, x);

        contains_zero = (ldom.lo < 0.0 && ldom.hi > 0.0);
        m = std::fmin(m, -contains_zero / 0.0); // NaN if not contains 0.0 -> chooses m, else -inf
        M = std::fmax(M, contains_zero / 0.0); // NaN if not contains 0.0 -> chooses M, else +inf

        DomainT new_rdom;
        try {
            new_rdom = {std::fmax(rdom.lo, m), std::fmin(rdom.hi, M)};
        }
        catch (const std::exception& e) { return INVALID; }

        UpdateResult res = static_cast<UpdateResult>(
                !(self == new_self
                && ldom == new_ldom
                && rdom == new_rdom));

        //std::cout << "PROD "
        //    << "self: " << self << " -> " << new_self
        //    << ", l: " << ldom << " -> " << new_ldom
        //    << ", r: " << rdom << " -> " << new_rdom
        //    << std::endl;

        self = new_self;
        ldom = new_ldom;
        rdom = new_rdom;

        return res;
    }

    UpdateResult
    Pow2::update(DomainT& self, DomainT& adom)
    {
        //std::cout << "POW2_0 "
        //    << "self: " << self
        //    << ", adom: " << adom
        //    << std::endl;

        // self = adom * adom
        FloatT m, M, x, contains_zero;
        m = adom.lo * adom.lo;
        M = m;
        x = adom.hi * adom.hi;
        m = std::fmin(m, x); M = std::fmax(M, x);

        contains_zero = (adom.lo<0.0) && (adom.hi>0.0);
        m = std::fmin(0.0 / contains_zero, m); // 0.0/0.0 is nan -> m stays unchanged, else 0/1==0

        DomainT new_self;
        try {
            new_self = {std::fmax(self.lo, m), std::fmin(self.hi, M)};
        }
        catch (const std::exception& e) { return INVALID; }

        // update adom: adom = sqrt(self)
        M = std::sqrt(self.hi);
        DomainT new_adom;
        try {
            new_adom = {std::fmax(adom.lo, -M), std::fmin(adom.hi, M)};
        }
        catch (const std::exception& e) { return INVALID; }

        UpdateResult res = static_cast<UpdateResult>(
                !(self == new_self
                && adom == new_adom));

        self = new_self;
        adom = new_adom;

        //std::cout << "POW2_1 "
        //    << "self: " << self
        //    << ", adom: " << adom
        //    << " res=" << res
        //    << std::endl;

        return res;
    }

    UpdateResult
    Sqrt::update(DomainT& self, DomainT& adom)
    {
        //std::cout << "SQRT0 "
        //    << "self: " << self
        //    << ", adom: " << adom
        //    << std::endl;

        // self = sqrt(adom) => both self and adom > 0

        // limit both doms to positive
        DomainT new_self, new_adom;
        try {
            new_self = { std::fmax(self.lo, 0.0f), self.hi };
            new_adom = { std::fmax(adom.lo, 0.0f), adom.hi };
        }
        catch (const std::exception& e) { return INVALID; }

        try {
            // self in [sqrt(adom.lo), sqrt(adom.hi)] (sqrt monotonic)
            new_self = {
                std::fmax(new_self.lo, std::sqrt(new_adom.lo)),
                std::fmin(new_self.hi, std::sqrt(new_adom.hi))
            };

            new_adom = {
                std::fmax(new_adom.lo, new_self.lo*new_self.lo),
                std::fmin(new_adom.hi, new_self.hi*new_self.hi)
            };

        }
        catch (const std::exception& e) { return INVALID; }

        UpdateResult res = static_cast<UpdateResult>(
                !(self == new_self
                && adom == new_adom));

        self = new_self;
        adom = new_adom;

        //std::cout << "SQRT1 "
        //    << "self: " << self
        //    << ", adom: " << adom
        //    << " res=" << res
        //    << std::endl;

        return res;
    }

    UpdateResult
    UnitVec2::update(DomainT& self, DomainT& adom, DomainT& bdom)
    {
        std::cout << "UNIT_VEC2_0 "
            << std::setprecision(12)
            << "self: " << self
            << ", adom: " << adom
            << ", bdom: " << bdom
            << std::endl;

        // use double precision to minize rounding errors

        // self = a / sqrt(a² + b²)
        double m, M, tmp;
        double alo = adom.lo, ahi = adom.hi;
        double blo = bdom.lo, bhi = bdom.hi;

        if (blo <= 0.0 && bhi >= 0.0)
        {
            m = -1.0;
            M = 1.0;
        }
        else
        {
            tmp = std::fmin(std::abs(blo), std::abs(bhi));
            tmp *= tmp;
            m = alo / std::sqrt(alo*alo + tmp);
            M = ahi / std::sqrt(ahi*ahi + tmp);
            tmp = std::fmax(std::abs(blo), std::abs(bhi));
            tmp *= tmp;
            m = std::fmin(m, alo / std::sqrt(alo*alo + tmp));
            M = std::fmax(M, ahi / std::sqrt(ahi*ahi + tmp));
            //m = std::nextafter(m, -std::numeric_limits<FloatT>::infinity());
            //M = std::nextafter(M, std::numeric_limits<FloatT>::infinity());
        }

        double slo = std::fmax(std::fmax(self.lo, -1.0), m);
        double shi = std::fmin(std::fmin(self.hi,  1.0), M);

        DomainT new_self;
        try {
            new_self = {static_cast<FloatT>(slo), static_cast<FloatT>(shi)};
        }
        catch (const std::exception& e) { std::cout << "inv1\n"; return INVALID; }

        // |a| = sqrt((self² * b²) / (1 - self²)) -> use new_self s.t. its dom is [-1, 1]
        // |a| = (|self| * |b| / sqrt(1 - self²)
        tmp = std::fmax(std::abs(slo), std::abs(shi));
        M = (tmp * std::fmax(std::abs(blo), std::abs(bhi))) / std::sqrt((1.0-tmp)*(1.0+tmp));
        //M = std::nextafter(M, std::numeric_limits<FloatT>::infinity());

        DomainT new_adom;
        try {
            new_adom = {std::fmax(adom.lo, -static_cast<FloatT>(M)),
                std::fmin(adom.hi, static_cast<FloatT>(M))};
        }
        catch (const std::exception& e) { std::cout << "inv2\n"; return INVALID; }

        // |b| = sqrt((a² * (1 - self²)) / self²)
        // |b| = (|a| * sqrt(1 - self²)) / |self|)
        if (slo <= 0.0 && shi >= 0.0) // self can be zero -> no info about b
        {
            // no info about b: if self is 0, then b can take on any value
            M = std::numeric_limits<FloatT>::infinity();
        }
        else
        {
            tmp = std::fmin(std::abs(slo), std::abs(shi));
            M = std::fmax(std::abs(alo), std::abs(ahi)) * std::sqrt((1.0-tmp)*(1.0+tmp)) / tmp;
            //M = std::nextafter(M, std::numeric_limits<FloatT>::infinity());
        }

        DomainT new_bdom;
        try {
            new_bdom = {std::fmax(bdom.lo, -static_cast<FloatT>(M)),
                std::fmin(bdom.hi, static_cast<FloatT>(M))};
        }
        catch (const std::exception& e) { std::cout << "inv3 " << M << "\n"; return INVALID; }

        UpdateResult res = static_cast<UpdateResult>(
                !(self == new_self
                && adom == new_adom
                && bdom == new_bdom
                ));

        self = new_self;
        adom = new_adom;
        bdom = new_bdom;

        std::cout << "UNIT_VEC2_1 "
            << std::setprecision(12)
            << "self: " << self
            << ", adom: " << adom
            << ", bdom: " << bdom
            << ", res: " << res
            << std::endl;

        return res;
    }

    UpdateResult
    Eq::update(DomainT& ldom, DomainT& rdom) const
    {
        // L == R -> share the same domain
        FloatT new_lo = std::max(ldom.lo, rdom.lo);
        FloatT new_hi = std::min(ldom.hi, rdom.hi);

        //std::cout << "EQ ldom " << ldom << ", rdom " << rdom << std::endl;

        if (new_lo > new_hi)
            return box_checker::INVALID;

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

        UpdateResult res = static_cast<UpdateResult>(
               (ldom.lo != new_lo || ldom.hi != new_hi)
            || (rdom.lo != new_lo || rdom.hi != new_hi));

        try {
            ldom = {ldom.lo, new_hi};
            rdom = {new_lo, rdom.hi};
        } catch (std::exception&) { return INVALID; }

        return res;
    }

} /* namespace box_checker */

    BoxChecker::BoxChecker(int num_vars, int max_num_updates)
        : num_vars_(num_vars)
        , max_num_updates_(max_num_updates)
        , exprs_()
        , comps_()
    {
        // provide space for the attributes
        box_checker::AnyExpr expr { };
        exprs_.resize(num_vars_, expr);
    }

    int
    BoxChecker::add_const(FloatT value)
    {
        box_checker::AnyExpr e;
        e.tag = box_checker::AnyExpr::CONST;
        e.constant = {value};
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
    BoxChecker::add_sub(int left, int right)
    {
        box_checker::AnyExpr e;
        e.tag = box_checker::AnyExpr::SUB;
        e.sub = {left, right};
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

    int
    BoxChecker::add_div(int left, int right)
    {
        box_checker::AnyExpr e;
        e.tag = box_checker::AnyExpr::DIV;
        e.div = {left, right};
        exprs_.push_back(e);
        return exprs_.size() - 1;
    }

    int
    BoxChecker::add_pow2(int arg)
    {
        box_checker::AnyExpr e;
        e.tag = box_checker::AnyExpr::POW2;
        e.pow2 = {arg};
        exprs_.push_back(e);
        return exprs_.size() - 1;
    }

    int
    BoxChecker::add_sqrt(int arg)
    {
        box_checker::AnyExpr e;
        e.tag = box_checker::AnyExpr::SQRT;
        e.pow2 = {arg};
        exprs_.push_back(e);
        return exprs_.size() - 1;
    }

    int
    BoxChecker::add_unit_vec2(int a, int b)
    {
        box_checker::AnyExpr e;
        e.tag = box_checker::AnyExpr::UNIT_VEC2;
        e.unit_vec2 = {a, b};
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
        c.tag = box_checker::AnyComp::EQ;
        comps_.push_back(c);
    }

    void
    BoxChecker::add_lteq(int left, int right)
    {
        box_checker::AnyComp c;
        c.left = left;
        c.right = right;
        c.comp.lteq = {};
        c.tag = box_checker::AnyComp::LTEQ;
        comps_.push_back(c);
    }

    void
    BoxChecker::add_at_most_k(std::vector<int> ids, int k)
    {
        bin_constraints_.push_back({
                box_checker::BinaryConstraint::AT_MOST_K, ids, k });
    }

    void
    BoxChecker::add_at_least_k(std::vector<int> ids, int k)
    {
        bin_constraints_.push_back({
                box_checker::BinaryConstraint::AT_LEAST_K, ids, k });
    }

    void
    BoxChecker::add_k_out_of_n(std::vector<int> ids, int k, bool strict)
    {
        if (strict) // both at most k and at least k == exactly k
        {
            bin_constraints_.push_back({
                    box_checker::BinaryConstraint::K_OUT_OF_N, ids, k });
        }
        else // only at most k
        {
            add_at_most_k(ids, k);
        }
    }


    DomainT
    BoxChecker::get_expr_dom(int id) const
    {
        return exprs_.at(id).dom;
    }



    void
    BoxChecker::copy_from_workspace(const std::vector<DomainPair>& workspace)
    {
        // workspace is ordered by .first (attribute id)

        size_t j = 0;
        for (int i = 0; i < num_vars_; ++i)
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

        // reset all intermediary values
        for (size_t i = num_vars_; i < exprs_.size(); ++i)
        {
            auto& expr = exprs_[i];
            if (expr.tag == box_checker::AnyExpr::CONST)
                expr.dom = {expr.constant.value, expr.constant.value};
            else
                expr.dom = {}; // reset domain of non-consts
        }
    }

    void
    BoxChecker::copy_to_workspace(std::vector<DomainPair>& workspace) const
    {
        size_t j = 0;
        size_t sz = workspace.size();
        for (int i = 0; i < num_vars_; ++i)
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
        for (const auto& c : comps_)
        {
            res = aggregate_update_result({res, update_comp(c)});
            RETURN_IF_INVALID(res);
        }
        for (const auto& c : bin_constraints_)
        {
            res = aggregate_update_result({res, update_bin_constraint(c)});
            RETURN_IF_INVALID(res);
        }
        return res;
    }

    bool
    BoxChecker::update(std::vector<DomainPair>& workspace)
    {
        copy_from_workspace(workspace);
        for (int i = 0; i < max_num_updates_; ++i)
        {
            box_checker::UpdateResult res = update();
            if (res == box_checker::UNCHANGED)
                break;
            if (res == box_checker::INVALID)
            {
                //std::cout << "invalid\n";
                return false; // reject this state
            }

        }
        copy_to_workspace(workspace);

        return true; // accept this state
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

        //switch (e.tag)
        //{
        //    case box_checker::AnyExpr::VAR: std::cout << "VAR\n"; break;
        //    case box_checker::AnyExpr::CONST: std::cout << "CONST\n"; break;
        //    case box_checker::AnyExpr::SUM: std::cout << "SUM\n"; break;
        //    case box_checker::AnyExpr::SUB: std::cout << "SUB\n"; break;
        //    case box_checker::AnyExpr::PROD: std::cout << "PROD\n"; break;
        //    case box_checker::AnyExpr::DIV: std::cout << "DIV\n"; break;
        //    case box_checker::AnyExpr::POW2: std::cout << "POW2\n"; break;
        //    case box_checker::AnyExpr::SQRT: std::cout << "SQRT\n"; break;
        //    case box_checker::AnyExpr::UNIT_VEC2: std::cout << "UNIT_VEC2\n"; break;
        //}

        switch (e.tag)
        {
        case box_checker::AnyExpr::VAR:
        case box_checker::AnyExpr::CONST:
            // leaf -> don't need to do anything
            break;
        case box_checker::AnyExpr::SUM: {
            box_checker::AnyExpr& left = exprs_[e.sum.left];
            box_checker::AnyExpr& right = exprs_[e.sum.right];
            auto sum_res = box_checker::Sum::update(e.dom, left.dom, right.dom);
            RETURN_IF_INVALID(sum_res);
            auto left_res = update_expr(left);
            auto right_res = update_expr(right);
            res = aggregate_update_result({sum_res, left_res, right_res});
            break;
        }
        case box_checker::AnyExpr::SUB: { // c = a-b <=> a = c+b
            box_checker::AnyExpr& left = exprs_[e.sub.left];
            box_checker::AnyExpr& right = exprs_[e.sub.right];
            auto sum_res = box_checker::Sum::update(left.dom, e.dom, right.dom);
            RETURN_IF_INVALID(sum_res);
            auto left_res = update_expr(left);
            auto right_res = update_expr(right);
            res = aggregate_update_result({sum_res, left_res, right_res});
            break;
        }
        case box_checker::AnyExpr::PROD: {
            box_checker::AnyExpr& left = exprs_[e.prod.left];
            box_checker::AnyExpr& right = exprs_[e.prod.right];
            auto prod_res = box_checker::Prod::update(e.dom, left.dom, right.dom);
            RETURN_IF_INVALID(prod_res);
            auto left_res = update_expr(left);
            auto right_res = update_expr(right);
            res = aggregate_update_result({prod_res, left_res, right_res});
            break;
        }
        case box_checker::AnyExpr::DIV: {
            box_checker::AnyExpr& left = exprs_[e.div.left];
            box_checker::AnyExpr& right = exprs_[e.div.right];
            // c = l/r <=> l = c*r --> reuse Prod::update
            auto div_res = box_checker::Prod::update(left.dom, e.dom, right.dom);
            RETURN_IF_INVALID(div_res);
            auto left_res = update_expr(left);
            auto right_res = update_expr(right);
            res = aggregate_update_result({div_res, left_res, right_res});
            break;
        }
        case box_checker::AnyExpr::POW2: {
            box_checker::AnyExpr& arg = exprs_[e.pow2.arg];
            auto pow2_res = box_checker::Pow2::update(e.dom, arg.dom);
            RETURN_IF_INVALID(pow2_res);
            auto arg_res = update_expr(arg);
            res = aggregate_update_result({pow2_res, arg_res});
            break;
        }
        case box_checker::AnyExpr::SQRT: {
            box_checker::AnyExpr& arg = exprs_[e.sqrt.arg];
            auto sqrt_res = box_checker::Sqrt::update(e.dom, arg.dom);
            RETURN_IF_INVALID(sqrt_res);
            auto arg_res = update_expr(arg);
            res = aggregate_update_result({sqrt_res, arg_res});
            break;
        }
        case box_checker::AnyExpr::UNIT_VEC2: {
            box_checker::AnyExpr& a = exprs_[e.unit_vec2.a];
            box_checker::AnyExpr& b = exprs_[e.unit_vec2.b];
            auto uvec_res = box_checker::UnitVec2::update(e.dom, a.dom, b.dom);
            RETURN_IF_INVALID(uvec_res);
            auto a_res = update_expr(a);
            auto b_res = update_expr(b);
            res = aggregate_update_result({uvec_res, a_res, b_res});
            break;
        }};

        return res;
    }

    box_checker::UpdateResult
    BoxChecker::update_bin_constraint(const box_checker::BinaryConstraint& c)
    {
        auto res = box_checker::UNCHANGED;

        int num_true = 0;
        int num_false = 0;

        for (auto id : c.ids)
        {
            const box_checker::AnyExpr& e = exprs_[id];
            num_true += !e.dom.overlaps(FALSE_DOMAIN); // not FALSE => assume TRUE
            num_false += !e.dom.overlaps(TRUE_DOMAIN); // not TRUE => assume FALSE
        }

        int num_unset = c.ids.size() - num_true - num_false;

        //std::cout << "BinaryConstraint " << num_true << ", " << num_false << ", " << num_unset << std::endl;

        // AT_LEAST_K
        if ((c.tag & box_checker::BinaryConstraint::AT_LEAST_K) != 0)
        {
            int most_true_possible = num_unset + num_true; // all unset become true
            if (most_true_possible < c.k) // impossible to make at least k true
                return box_checker::INVALID;

            if (most_true_possible == c.k && num_unset > 0) // set all remaining unset to true
            {
                for (auto id : c.ids)
                {
                    box_checker::AnyExpr& e = exprs_[id];
                    if (e.dom.is_everything())
                    {
                        e.dom = TRUE_DOMAIN;
                        //std::cout << "AT_LEAST_K set " << id << " to TRUE" << std::endl;
                    }
                }
                res = box_checker::UPDATED;
            }
        }

        // AT_MOST_K
        if ((c.tag & box_checker::BinaryConstraint::AT_MOST_K) != 0)
        {
            if (num_true > c.k) // more trues that allowed
                return box_checker::INVALID;

            if (num_true == c.k && num_unset > 0) // set all remaining to false
            {
                for (auto id : c.ids)
                {
                    box_checker::AnyExpr& e = exprs_[id];
                    if (e.dom.is_everything())
                    {
                        e.dom = FALSE_DOMAIN;
                        //std::cout << "AT_MOST_K set " << id << " to FALSE" << std::endl;
                    }
                }
                res = box_checker::UPDATED;
            }
        }

        return res;
    }

} /* namespace veritas */
