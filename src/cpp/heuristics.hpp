/**
 * \file heuristics.hpp
 *
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "domain.hpp"
#include "tree.hpp"

namespace veritas {

    class VSearch;
    template <typename Heuristic> class Search;

    /** \private */
    struct BaseState {
        BoxRef box;
        int indep_set;

        BaseState() : box(BoxRef::null_box()), indep_set(-1) {}
    };

    /** \private */
    struct MaxOutputState : public BaseState {
        FloatT g, h;
        MaxOutputState() : BaseState(), g(0.0), h(0.0) {}
    };

    struct MaxOutputHeuristic {
        using State = MaxOutputState;

        MaxOutputHeuristic() {}

        /**
         * Fields `box` and `indep_set` of `out` must be set
         * Returns true if successful, false otherwise (e.g. invalid state).
         */
        bool update_heuristic(
                State& out,
                const Search<MaxOutputHeuristic>& search,
                const State& parent,
                FloatT leaf_value) const
        {
            FloatT g = parent.g + leaf_value;
            //FloatT h = search.graph_.basic_remaining_upbound(out.indep_set+1,
            //        out.box);
            FloatT h = search.compute_basic_output_heuristic_(out);
            //std::cout << h << ", " << h2 << std::endl;

            if (!std::isinf(h))
            {
                out.g = g;
                out.h = h;
                return open_score(out) >= search.stop_when_upper_less_than;
            }
            else return false;
        }

        void print_state(std::ostream& strm, const State& s)
        {
            strm << "State g=" << s.g << ", h=" << s.h
                << ", f=" << open_score(s)
                << ", out=" << output_overestimate(s)
                << ", set=" << s.indep_set
                << std::endl;
        }

        FloatT output_overestimate(const State& state) const
        { return state.g + state.h; }
        
        FloatT open_score(const State& state) const
        { return state.g + state.h; }

        bool cmp_open_score(const State& a, const State& b) const
        { return open_score(a) < open_score(b); }

        //FloatT focal_score(const State& state) const
        //{
        //    // deeper solution first
        //    return state.indep_set;
        //}

        bool cmp_focal_score(const State& a, const State& b) const
        {
            // if depth is the same and we have an example,
            // states closest to example first
            //if (a.indep_set == b.indep_set && example.size() > 0)
            //    return compute_delta(a) < compute_delta(b);
            
            // tie breaker: biggest h first
            if (a.indep_set == b.indep_set)
                return a.h > b.h;

            // deeper solutions first
            return a.indep_set > b.indep_set;
        }
    };

    struct MinDistToExampleState : public BaseState {
        FloatT dist, g, h;
        MinDistToExampleState() : BaseState(), dist(0.0), g(0.0), h(0.0) {}
    };

    struct MinDistToExampleHeuristic {
        using State = MinDistToExampleState;
        FloatT output_threshold;
        std::vector<FloatT> example;

        MinDistToExampleHeuristic(
                const std::vector<FloatT>& example,
                FloatT output_threshold)
            : output_threshold(output_threshold)
            , example(example) {}

        /**
         * Fields `box` and `indep_set` of `out` must be set
         * Returns true if successful, false otherwise (e.g. invalid state).
         */
        bool update_heuristic(
                State& out,
                const Search<MinDistToExampleHeuristic>& search,
                const State& parent,
                FloatT leaf_value) const
        {
            FloatT g = parent.g + leaf_value;
            //FloatT h = search.graph_.basic_remaining_upbound(out.indep_set+1,
            //        out.box);
            FloatT h = search.compute_basic_output_heuristic_(out);
            //std::cout << h << ", " << h2 << std::endl;

            if (!std::isinf(h) && (g+h) > output_threshold)
            {
                out.g = g;
                out.h = h;
                out.dist = compute_delta(out);
                return true;
            }
            else
            {
                //std::cout << "heuristics update reject " << (g+h+search.base_score()) << std::endl;
                return false;
            }
        }

        FloatT output_overestimate(const State& state) const
        { return state.g + state.h; }

        FloatT open_score(const State& state) const
        { return state.dist; }

        bool cmp_open_score(const State& a, const State& b) const
        { return open_score(a) > open_score(b); /* largest first */ }

        bool cmp_focal_score(const State& a, const State& b) const
        {
            // solutions with bigger estimated output first
            // tie breaker: biggest h first
            if (a.indep_set == b.indep_set)
                return (a.g + a.h) > (b.g + b.h);

            // deeper solutions first
            return a.indep_set > b.indep_set;
        }

        FloatT compute_delta(const State& s) const
        {
            FloatT delta = 0.0;
            for (auto &&[feat_id, dom] : s.box)
            {
                FloatT x = example[feat_id];
                if (!dom.contains(x))
                {
                    FloatT d = std::min(std::abs(dom.lo - x), std::abs(dom.hi - x));
                    //std::cout << "feature " << feat_id << ": " << x << ", " << dom << " -> " << d << std::endl;
                    delta = std::max(delta, d); // linf
                    //delta += d*d; // l2
                }
            }
            return delta;
        }

        void print_state(std::ostream& strm, const State& s)
        {
            strm << "State g=" << s.g << ", h=" << s.h
                << ", f=" << open_score(s)
                << ", dist=" << s.dist
                << ", set=" << s.indep_set
                << std::endl;
        }
    };

} // namespace veritas
