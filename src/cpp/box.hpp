/**
 * \file box.hpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_BOX_HPP
#define VERITAS_BOX_HPP

#include "basics.hpp"
#include "interval.hpp"
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace veritas {

/** An interval annotated with a feature ID */
template <typename T>
struct GIntervalPair { // generic interval pair
    FeatId feat_id;
    GInterval<T> interval;

    GIntervalPair(FeatId f, GInterval<T> i) : feat_id{f}, interval{i} {}

    bool operator==(const GIntervalPair& o) {
        return o.feat_id == feat_id && o.interval == interval;
    }
    bool operator!=(const GIntervalPair& o) { return !(*this == o); }
};

using IntervalPair = GIntervalPair<FloatT>;
using IntervalPairFp = GIntervalPair<FpT>;

template <typename T>
using GFlatBox = std::vector<GInterval<T>>; // generic flatbox

using FlatBox = GFlatBox<FloatT>;
using FlatBoxFp = GFlatBox<FpT>;

template <typename T> class GBoxRef;

template <typename T>
class GBox { // generic box
public:
    using BufT = std::vector<GIntervalPair<T>>;
    using FlatBoxT = GFlatBox<T>;
    using iterator = typename BufT::iterator;
    using const_iterator = typename BufT::const_iterator;

private:
    BufT& buf_;
    size_t offset_;

public:
    GBox(BufT& buf) : GBox(buf, 0) {}
    GBox(BufT& buf, size_t o) : buf_{buf}, offset_(o) {}

    inline const_iterator begin() const { return buf_.begin() + offset_; }
    inline const_iterator end() const { return buf_.end(); }
    inline iterator begin() { return buf_.begin() + offset_; }
    inline iterator end() { return buf_.end(); }
    inline size_t size() const { return end() - begin(); }
    inline bool empty() const { return size() == 0; }
    inline void clear() { buf_.erase(begin(), end()); }

    inline GInterval<T>& get_or_insert(FeatId feat_id) {
        iterator it =
            std::find_if(begin(), end(), [feat_id](const GIntervalPair<T>& p) {
                return p.feat_id >= feat_id;
            });
        if (it == end() || it->feat_id != feat_id)
            it = buf_.insert(it, GIntervalPair<T>(feat_id, GInterval<T>()));

        return it->interval;
    }

    /**
     * Include the interval for `feat_id` in the box.
     *
     * If no interval is present for `feat_id`, then add it, and maintain the
     * sorted property of the box. If an interval is already present for
     * `feat_id`, then intersect it. If it does not overlap, then return false,
     * else return true.
     */
    inline bool refine_box(FeatId feat_id, const GInterval<T>& ival) {
        auto& new_ival = get_or_insert(feat_id);
        if (!new_ival.overlaps(ival))
            return false;
        new_ival = new_ival.intersect(ival);
        return true;
    }

    /**
     * See GBox::refine_box() and GLtSplit<T>::get_intervals()
     */
    inline bool refine_box(const GLtSplit<T>& split, bool from_left_child)
    {
        auto ival = from_left_child
            ? std::get<0>(split.get_intervals())
            : std::get<1>(split.get_intervals());

        return refine_box(split.feat_id, ival);
    }

    inline void combine_boxes(const GBoxRef<T>& a, const GBoxRef<T>& b, bool copy_b) {
        if (!this->empty())
            throw std::runtime_error("output box should be empty");

        if constexpr (check_sanity())
            if (!a.overlaps(b))
                throw std::invalid_argument("cannot combine non-overlapping boxes");

        auto it0 = a.begin();
        auto it1 = b.begin();
        //
        // assume sorted
        while (it0 != a.end() && it1 != b.end())
        {
            if (it0->feat_id == it1->feat_id)
            {
                GInterval<T> ival = it0->interval.intersect(it1->interval);
                buf_.push_back({ it0->feat_id, ival });
                ++it0; ++it1;
            }
            else if (it0->feat_id < it1->feat_id)
            {
                buf_.push_back(*it0); // copy
                ++it0;
            }
            else
            {
                if (copy_b)
                    buf_.push_back(*it1); // copy
                ++it1;
            }
        }

        // push all remaining items (one of them is already at the end, no need
        // to compare anymore)
        for (; it0 != a.end(); ++it0)
            buf_.push_back(*it0); // copy
        for (; copy_b && it1 != b.end(); ++it1)
            buf_.push_back(*it1); // copy
    }
};

using Box = GBox<FloatT>;
using BoxFp = GBox<FpT>;

template <typename T>
class GBoxRef { // generic boxref
public:
    using BoxT = GBox<T>;
    using BufT = typename BoxT::BufT;
    using FlatBoxT = GFlatBox<T>;
    using const_iterator = typename BoxT::const_iterator;

private:
    static const BufT EMPTY_BOX_BUF;

    const_iterator begin_, end_;

public:
    inline GBoxRef() : begin_{EMPTY_BOX_BUF.begin()}, end_{EMPTY_BOX_BUF.end()} {}
    inline GBoxRef(const BoxT& b) : begin_{b.begin()}, end_{b.end()} {}
    inline GBoxRef(const_iterator b, const_iterator e) : begin_{b}, end_{e} {}

    inline const_iterator begin() const { return begin_; }
    inline const_iterator end() const { return end_; }
    inline size_t size() const { return end_ - begin_; }
    inline bool empty() const { return size() == 0; }

    /** Do the intervals for the corresponding feature ids overlap? */
    inline bool overlaps(const GBoxRef& other) const {
        auto it0 = begin_;
        auto it1 = other.begin_;

        while (it0 != end_ && it1 != other.end_)
        {
            if (it0->feat_id == it1->feat_id)
            {
                if (!it0->interval.overlaps(it1->interval))
                    return false;
                ++it0; ++it1;
            }
            else if (it0->feat_id < it1->feat_id) ++it0;
            else ++it1;
        }

        return true;
    }

    inline GInterval<T> get(FeatId feat_id) const {
        for (auto it = begin_; it < end_; ++it) {
            if (it->feat_id == feat_id)
                return it->interval;
        }
        return {};
    }

    inline void to_flatbox(FlatBoxT& fbox, bool clear) const {
        if (clear)
            std::fill(fbox.begin(), fbox.end(), GInterval<T>());
        if (empty()) return;

        // feat_ids are sorted, so last one is max
        FeatId max_feat_id = (end()-1)->feat_id;
        if (fbox.size() <= static_cast<size_t>(max_feat_id))
            fbox.resize(max_feat_id+1, GInterval<T>());

        for (auto &&[feat_id, ival] : *this)
            fbox[feat_id] = fbox[feat_id].intersect(ival);
    }

    FlatBoxT to_flatbox() const {
        FlatBoxT b;
        to_flatbox(b, false);
        return b;
    }
};

template <typename T>
const typename GBoxRef<T>::BufT
GBoxRef<T>::EMPTY_BOX_BUF = {};

using BoxRef = GBoxRef<FloatT>;
using BoxRefFp = GBoxRef<FpT>;

template <typename T>
inline
std::ostream&
operator<<(std::ostream& s, const GBoxRef<T>& box)
{
    s << "Box { ";
    for (auto&& [id, ival] : box)
        s << id << ":" << ival << " ";
    s << '}';
    return s;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& s, const GBox<T>& box) {
    return s << GBoxRef(box);
}

} // namespace veritas

#endif // VERITAS_BOX_HPP

