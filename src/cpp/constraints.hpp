/*
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_CONSTRAINTS_HPP
#define VERITAS_CONSTRAINTS_HPP

#include "search.hpp"

/**
 * Veritas constraints.
 *
 * You can make up feature IDs.
 * If your problems has 5 features with feature IDs 0, 1, 2, 3, 4, then you can
 * use feature ID 5, 6, ... for 'psuedo' features.
 */
namespace veritas::constraints {

    using veritas::Search;

    /**
     * `feat_x <= feat_y`
     *
     *       |---- dom x ---!!!!|
     * |!!!!!----- dom y --|
     */
    template <typename H>
    void lteq(Search<H>& s, FeatId x, FeatId y)
    {
        int grp = s.callback_group();

        // if x in [a, b], then y cannot be less than a
        s.add_callback(x, [x, y](CallbackContext<H>& ctx, Domain domx) {
            ctx.intersect(y, Domain::from_lo(domx.lo));
        }, grp);

        // if y in [a, b], then x cannot be greater than b
        s.add_callback(y, [x, y](CallbackContext<H>& ctx, Domain domy) {
            ctx.intersect(x, Domain::from_hi_inclusive(domy.hi));
        }, grp);
    }
    
    /**
     * `feat_c = feat_a + feat_b` == `feat_a = feat_c - feat_b`
     *                            == `feat_b = feat_c - feat_a`
     */
    template <typename H>
    void sum(Search<H>& s, FeatId a, FeatId b, FeatId c)
    {
        int grp = s.callback_group();
        s.add_callback(c, [a, b, c](CallbackContext<H>& ctx, Domain domc) {
            Domain doma = ctx.get(a);
            Domain domb = ctx.get(b);
            ctx.intersect(a, domc.lo - domb.hi, domc.hi - domb.lo);
            ctx.intersect(b, domc.lo - doma.hi, domc.hi - doma.lo);
        }, grp);

        s.add_callback(a, [a, b, c](CallbackContext<H>& ctx, Domain doma) {
            Domain domb = ctx.get(b);
            ctx.intersect(c, doma.lo + domb.lo, doma.hi + domb.hi);
        }, grp);

        s.add_callback(b, [a, b, c](CallbackContext<H>& ctx, Domain domb) {
            Domain doma = ctx.get(a);
            ctx.intersect(c, doma.lo + domb.lo, doma.hi + domb.hi);
        }, grp);
    }

    template <typename H>
    void onehot(Search<H>& s, std::vector<FeatId> xs)
    {
        int grp = s.callback_group();
        for (FeatId x : xs)
        {
            s.add_callback(x, [x, xs](CallbackContext<H>& ctx, Domain domx) {
                if (!domx.overlaps(FALSE_DOMAIN)) // not false, must be true
                {
                    for (FeatId y : xs)
                        ctx.intersect(y, x==y ? TRUE_DOMAIN : FALSE_DOMAIN);
                }
                else // if all others are false, then this must be true
                {
                    bool all_false = true;
                    for (FeatId y : xs)
                        all_false &= (y == x) || !ctx.get(y).overlaps(TRUE_DOMAIN);
                    if (all_false)
                        for (FeatId y : xs)
                            ctx.intersect(y, x==y ? TRUE_DOMAIN : FALSE_DOMAIN);
                }
            }, grp);
        }
    }

    template <typename H>
    void onehot(Search<H>& s, std::initializer_list<FeatId> il_xs)
    { onehot(s, std::vector<FeatId>(il_xs)); }

    /**
     * Squared distance between two 2-d points, one of which is fixed.
     * 
     * `feat_d = sqrt((x0 - feat_x1)**2 + (y0 - feat_y1)**2)`
     */
    template <typename H>
    void sqdist1(Search<H>& s, FeatId x, FeatId y, FeatId d, FloatT x0, FloatT y0)
    {
        int grp = s.callback_group();

        auto pow2 = [](double x) { return x*x; };
        auto cb = [x, y, d, x0, y0, pow2](CallbackContext<H>& ctx, Domain) {
            Domain domx = ctx.get(x);
            Domain domy = ctx.get(y);

            // single term (x-x0)**2
            //      lo       hi
            //      |--------|
            // x0                   -> min at (lo-x0)**2 -> lo-x0 > 0, x0-hi < 0
            //          x0          -> min at (0.0)**2
            //                   x0 -> min at (x0-hi)**2 -> lo-x0 < 0, x0-hi > 0
            FloatT x1m = std::max({(FloatT)0.0, domx.lo - x0, x0 - domx.hi});
            FloatT y1m = std::max({(FloatT)0.0, domy.lo - y0, y0 - domy.hi});
            FloatT x1M = std::max(std::abs(domx.lo-x0), std::abs(x0-domx.hi));
            FloatT y1M = std::max(std::abs(domy.lo-y0), std::abs(y0-domy.hi));
            FloatT dlo = std::sqrt(pow2(x1m) + pow2(y1m));
            FloatT dhi = std::sqrt(pow2(x1M) + pow2(y1M));
            std::cout
                << domx << " " << domy << " "
                << x1m << " " << y1m << " " << x1M << " " << y1M
                << " -> " << dlo << " " << dhi
                << std::endl;
            ctx.intersect(d, dlo, dhi);
        };

        s.add_callback(x, cb, grp);
        s.add_callback(y, cb, grp);

        s.add_callback(d, [x, y, d, x0, y0, pow2](CallbackContext<H>& ctx, Domain domd) {
            Domain domx = ctx.get(x);
            Domain domy = ctx.get(y);

            ctx.intersect(d, Domain::from_lo(0.0)); // distance is positive

            FloatT x1m = std::max({(FloatT)0.0, domx.lo - x0, x0 - domx.hi});
            FloatT y1m = std::max({(FloatT)0.0, domy.lo - y0, y0 - domy.hi});
            FloatT x1M = std::max(std::abs(domx.lo-x0), std::abs(x0-domx.hi));
            FloatT y1M = std::max(std::abs(domy.lo-y0), std::abs(y0-domy.hi));

            FloatT dym = std::sqrt(std::max(0.0, pow2(domd.lo) - pow2(y1M)));
            FloatT dyM = std::sqrt(std::max(0.0, pow2(domd.hi) - pow2(y1m)));
            FloatT dxm = std::sqrt(std::max(0.0, pow2(domd.lo) - pow2(x1M)));
            FloatT dxM = std::sqrt(std::max(0.0, pow2(domd.hi) - pow2(x1m)));

            std::cout << "(1) " << domx << ", " << domy << ", " << domd << " "
                << (dym+x0) << ", " << (dyM+x0) << "; "
                << (dxm+x0) << ", " << (dxM+x0) << std::endl;

            std::cout  << "(2) " << domx << ", " << domy << ", " << domd << " "
                << (x0-dyM) << ", " << (x0-dym) << "; "
                << (y0-dxM) << ", " << (y0-dxm) << std::endl;

            ctx.intersect(x, x0-dyM, x0-dym);
            ctx.intersect(y, y0-dxM, y0-dxm);

            ctx.duplicate();

            ctx.intersect(x, dym+x0, dyM+x0);
            ctx.intersect(y, dxm+x0, dxM+x0);
        }, grp);
    }

} // namespace veritas::constraints

#endif // VERITAS_CONSTRAINTS_HPP

