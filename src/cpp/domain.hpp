/**
 * \file domain.hpp
 *
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_DOMAIN_HPP
#define VERITAS_DOMAIN_HPP

#include "basics.hpp"

#include <limits>
#include <sstream>
#include <cmath>
#include <tuple>
#include <vector>
#include <algorithm>

namespace veritas {

    struct Domain;
    std::ostream& operator<<(std::ostream& s, const Domain& d);

    /** A real interval with lower bound `lo` and upper bound `hi`. */
    struct Domain {
        FloatT lo; /**< The lower limit (possibly -inf) */
        FloatT hi; /**< The upper limit (possibly +inf) */

        /** Construct interval (-inf, inf) */
        inline Domain()
            : lo(-FLOATT_INF)
            , hi(FLOATT_INF) {}

        /** Construct interval [lo, hi] (inclusive) */
        inline Domain(FloatT lo, FloatT hi) : lo(lo), hi(hi)
        {
#ifndef VERITAS_SAFETY_CHECKS_DISABLED
            if (lo > hi)
            {
                std::stringstream s;
                s << "Domain<real> error: lo > hi: [" << lo << ", " << hi << "]";
                throw std::invalid_argument(s.str());
            }
#endif
        }

        /** Construct interval [lo, inf) */
        static inline Domain from_lo(FloatT lo) { return {lo, FLOATT_INF}; }
        /** Construct interval [-inf, hi] (inclusive) */
        static inline Domain from_hi_inclusive(FloatT hi) { return {-FLOATT_INF, hi}; }
        /** Construct interval [-inf, hi) (exclusive) */
        static inline Domain from_hi_exclusive(FloatT hi) { return Domain::exclusive(-FLOATT_INF, hi); }
        /** Construct interval [lo, hi] (inclusive) */
        static inline Domain inclusive(FloatT lo, FloatT hi) { return {lo, hi}; }
        /** Construct interval [lo, hi) (exclusive) */
        static inline Domain exclusive(FloatT lo, FloatT hi)
        { return {lo, std::isinf(hi) ? FLOATT_INF : std::nextafter(hi, -FLOATT_INF)}; }

        /** Check if equal to (-inf, inf) */
        inline bool is_everything() const { return lo == -FLOATT_INF && hi == FLOATT_INF; };
        /** Check if value `v` is in domain. */
        inline bool contains(FloatT v) const { return lo <= v && v <= hi; }
        /** Check if two domains overlap */
        inline bool overlaps(const Domain& other) const
        {
            // [   ]
            //     [     ] edges are inclusive
            return lo <= other.hi && hi >= other.lo;
        }

        /** Intersect two domains. Given domain must Domain::overlap. */
        inline Domain intersect(const Domain& other) const
        {
            #ifndef VERITAS_SAFETY_CHECKS_DISABLED
            if (!overlaps(other))
            {
                std::stringstream ss;
                ss << "Domain::intersect: no overlap " << *this << " and " << other;
                throw std::runtime_error(ss.str());
            }
            #endif
            return { std::max(lo, other.lo), std::min(hi, other.hi) };
        }

        inline bool lo_is_inf() const { return lo == -FLOATT_INF; }
        inline bool hi_is_inf() const { return hi ==  FLOATT_INF; }

        /** Split this domain in two. Domain must constain the value (Domain::contains).
         * Consistent with LtSplit: strictly less than, left domain exclusive. */
        inline std::tuple<Domain, Domain> split(FloatT value) const
        {
            return {
                Domain::exclusive(lo, value),
                Domain::inclusive(value, hi)
            };
        }

        inline bool operator==(const Domain& other) const {
            return lo == other.lo && hi == other.hi;
        }
        inline bool operator!=(const Domain& other) const { return !(*this == other); }
    }; // sturct Domain

    inline std::ostream& operator<<(std::ostream& s, const Domain& d)
    {
        if (d.is_everything())
            return s << "Dom(R)";
        if (d.hi_is_inf())
            return s << "Dom(>=" << d.lo << ')';
        if (d.lo_is_inf())
            return s << "Dom(<=" << d.hi << ')';
        return s << "Dom(" << d.lo << ',' << d.hi << ')';
    }

    /** Split value used for boolean splits (assuming feature values in {0, 1}) */
    const FloatT BOOL_SPLIT_VALUE = 0.5;
    /** [-inf, 1) domain for FALSE */
    const Domain FALSE_DOMAIN = Domain::from_hi_exclusive(BOOL_SPLIT_VALUE);
    /** [1, inf) domain for TRUE */
    const Domain TRUE_DOMAIN = Domain::from_lo(BOOL_SPLIT_VALUE);

    /** A domain annotated with its feature id. */
    struct DomainPair {
        FeatId feat_id;
        Domain domain;
    };

    /** A __sorted__ list of pairs */
    using Box = std::vector<DomainPair>;



    /** A less than split on a feature */
    struct LtSplit {
        FeatId feat_id;
        FloatT split_value;

        inline LtSplit(FeatId f, FloatT v) : feat_id(f), split_value(v) {}

        /** True goes left, false goes right */
        inline bool test(FloatT v) const { return v < split_value; }
        /** Evaluate this split on an instance. */
        inline bool test(const data& row) { return test(row[feat_id]); }

        /** Get the left and right domains of this split.
         * Strict less than, so eq goes right */
        inline std::tuple<Domain, Domain> get_domains() const
        { return Domain().split(split_value); }

        inline bool operator==(const LtSplit& o) const
        { return feat_id == o.feat_id && split_value == o.split_value; }
    };
    
    inline
    std::ostream& operator<<(std::ostream& strm, const LtSplit& s)
    { return strm << "F" << s.feat_id << " < " << s.split_value; }



    /**
     * Get a reference to the Domain entry in the given box for `feat_id`. If
     * no entry exists, then it is inserted in the right place so that the box
     * remains sorted on feat_id.
     */
    inline Domain& get_domain(Box& box, FeatId feat_id)
    {
        auto it = std::find_if(box.begin(), box.end(), [feat_id](const DomainPair& p)
                { return p.feat_id == feat_id; });

        // domain not in box, insert and ensure sorted
        if (it == box.end())
        {
            box.push_back({ feat_id, {} });
            for (it = box.end()-1; it > box.begin(); --it)
            {
                //std::cout << (it-1)->feat_id << " vs " << it->feat_id << std::endl;
                if ((it-1)->feat_id > it->feat_id)
                    std::swap(*(it-1), *it);
                else break;
            }
            //for (int i = box.size() - 1; i > 0; --i) // ids sorted
            //    if (box[i-1].feat_id > box[i].feat_id)
            //        std::swap(box[i-1], box[i]);
            //std::sort(workspace_.begin(), workspace_.end(),
            //        [](const DomainPair& a, const DomainPair& b) {
            //            return a.first < b.first;
            //        })
        }
        return it->domain;
    }

    /**
     * Include the constraint '`feat_id` in `dom`' to the box.
     *
     * If no domain is present for `feat_id`, then add it, and maintain the
     * sorted property of the box. If a domain is already present for
     * `feat_id`, then intersect it. If it does not overlap, then return false,
     * else return true.
     */
    inline bool refine_box(Box& box, FeatId feat_id, const Domain& dom)
    {
        Domain& new_dom = get_domain(box, feat_id);
        if (!new_dom.overlaps(dom))
            return false;
        new_dom = new_dom.intersect(dom);
        return true;
    }

    /**
     * See ::refine_box() and LtSplit::get_domains()
     */
    inline bool refine_box(Box& doms, const LtSplit& split, bool from_left_child)
    {
        Domain dom = from_left_child
            ? std::get<0>(split.get_domains())
            : std::get<1>(split.get_domains());

        return refine_box(doms, split.feat_id, dom);
    }

    /**
     * An iterator of `Domain`s sorted asc by their feature IDs.
     * This represents a hypercube in the input space.
     *
     * BoxRefs are used in search spaces (GraphSearch) in combination with
     * stable pointers into a BlockStore.
     */
    class BoxRef {
    public:
        using const_iterator = const DomainPair *;

    private:
        const_iterator begin_, end_;

    public:
        inline BoxRef(const_iterator b, const_iterator e) : begin_(b), end_(e) {}
        inline BoxRef(const Box& d) : BoxRef(nullptr, nullptr)
        {
            if (d.size() != 0)
            {
                begin_ = &d[0];
                end_ = begin_ + d.size();
            }
        }
        template <typename T>
        explicit inline BoxRef(const T& t) : BoxRef(t.begin, t.end) {}
        /** A box with no domain restrictions representing the full input space. */
        inline static BoxRef null_box() { return {nullptr, nullptr}; }
        inline static BoxRef invalid_box()
        { return {nullptr, reinterpret_cast<const_iterator>(size_t(1))}; }
        inline bool is_null_box() const { return begin_ == nullptr; }
        inline bool is_invalid_box() const
        { return end_ == reinterpret_cast<const_iterator>(size_t(1)); }
        inline const_iterator begin() const { return begin_; }
        inline const_iterator end() const { return end_; }

        /** Do the domains for the corresponding feature ids overlap? */
        inline bool overlaps(const BoxRef& other) const
        {
            auto it0 = begin_;
            auto it1 = other.begin_;

            while (it0 != end_ && it1 != other.end_)
            {
                if (it0->feat_id == it1->feat_id)
                {
                    if (!it0->domain.overlaps(it1->domain))
                        return false;
                    ++it0; ++it1;
                }
                else if (it0->feat_id < it1->feat_id) ++it0;
                else ++it1;
            }

            return true;
        }

        /** Does the box overlap the domain for the given feature? This is a
         * slow linear scan. */
        inline bool overlaps(FeatId feat_id, const Domain& dom) const
        {
            for (auto it = begin_; it < end_; ++it)
            {
                if (it->feat_id == feat_id)
                    return it->domain.overlaps(dom);
            }
            return true;
        }

        const static int OVERLAPS_LEFT = 1; /**< Possible output of BoxRef::overlaps(LtSplit&) */
        const static int OVERLAPS_RIGHT = 2; /**< Possible output of BoxRef::overlaps(LtSplit&) */

        /** Does the box overlap the left and right domains for the given split?
         * See LtSplit::get_domains()
         *
         * This is a slow linear scan.
         *
         * Output is:
         * - OVERLAPS_LEFT
         * - OVERLAPS_RIGHT
         * - OVERLAPS_LEFT | OVERLAPS_RIGHT
         */
        inline int overlaps(const LtSplit& split) const
        {
            for (auto it = begin_; it < end_; ++it)
            {
                if (it->feat_id == split.feat_id)
                {
                    Domain dom = it->domain;
                    Domain ldom, rdom;
                    std::tie(ldom, rdom) = split.get_domains();
                    return static_cast<int>(dom.overlaps(ldom))*OVERLAPS_LEFT
                        | static_cast<int>(dom.overlaps(rdom))*OVERLAPS_RIGHT;
                }
            }
            return OVERLAPS_LEFT | OVERLAPS_RIGHT;
        }

        inline size_t size() const { return end_ - begin_; }

        bool operator==(const BoxRef& other) const
        {
            auto it0 = begin_;
            auto it1 = other.begin_;

            for (;it0 != end_ && it1 != other.end_; ++it0, ++it1)
            {
                if (it0->feat_id != it1->feat_id)
                    return false;
                if (it0->domain != it1->domain)
                    return false;
            }

            return true;
        }
    }; // class BoxRef

    /**
     * Intersect the boxes with corresponding feature IDs.
     *
     * If `copy_b` is false, then the domains in `b` for which no corresponding
     * domain exists in `a` are excluded.
     *
     * It is assumed that both boxes overlap.
     */
    inline void
    combine_boxes(const BoxRef& a, const BoxRef& b, bool copy_b, Box& out)
    {
        if (!out.empty())
            throw std::runtime_error("output box is not empty");

        const DomainPair *it0 = a.begin();
        const DomainPair *it1 = b.begin();

        // assume sorted
        while (it0 != a.end() && it1 != b.end())
        {
            if (it0->feat_id == it1->feat_id)
            {
                Domain dom = it0->domain.intersect(it1->domain);
                out.push_back({ it0->feat_id, dom });
                ++it0; ++it1;
            }
            else if (it0->feat_id < it1->feat_id)
            {
                out.push_back(*it0); // copy
                ++it0;
            }
            else
            {
                if (copy_b)
                    out.push_back(*it1); // copy
                ++it1;
            }
        }

        // push all remaining items (one of them is already at the end, no need to compare anymore)
        for (; it0 != a.end(); ++it0)
            out.push_back(*it0); // copy
        for (; copy_b && it1 != b.end(); ++it1)
            out.push_back(*it1); // copy
    }

    inline
    std::ostream&
    operator<<(std::ostream& s, const BoxRef& box)
    {
        s << "Box { ";
        for (auto&& [id, dom] : box)
            s << id << ":" << dom << " ";
        s << '}';
        return s;
    }





} // namespace veritas


#endif // VERITAS_DOMAIN_HPP
