/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "constraints.hpp"

namespace veritas {

    UpdateResult
    Eq::update(Domain& ldom, Domain& rdom) const
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

        return res;
    }

    UpdateResult
    LtEq::update(Domain& ldom, Domain& rdom) const
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

} // namespace veritas
