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

    struct BaseHeuristic {
        /**
         * Compute an overestimate of the remaining output value to a solution
         * state.
         */
        template <typename Search, typename State>
        FloatT compute_basic_output_heuristic_(const Search& s, const State& state) const
        {
            FloatT h = 0.0;
            s.workspace_.leafiter2.setup_flatbox(state.box); // do once
            for (size_t tree_index = state.indep_set + 1;
                    tree_index < s.at_.size(); ++tree_index)
            {
                FloatT max = -FLOATT_INF;
                const Tree& t = s.at_[tree_index];
                s.workspace_.leafiter2.setup_tree(t);
                NodeId leaf_id = -1;
                while ((leaf_id = s.workspace_.leafiter2.next()) != -1)
                {
                    if (s.node_box_[tree_index][leaf_id].is_invalid_box())
                        continue;
                    max = std::max(t[leaf_id].leaf_value(), max);
                }
                h += max;
            }
            return h;
        }
    };

    struct MinHeuristic : public BaseHeuristic {
        /**
         * Increase oscore by eps, i.e., also accept states with higher scores.
         * See Search::pop_from_focal_.
         * */
        FloatT relax_open_score(FloatT oscore, FloatT eps) const
        { return oscore + (1.0-eps)*std::abs(oscore); }

        /** Returns true when a is 'better' than b, i.e. a is larger than b */
        bool cmp_open_score(FloatT a, FloatT b) const { return a < b; }
    };

    struct MaxHeuristic : public BaseHeuristic {
        /**
         * Decrease oscore by eps, i.e., also accept states with lower scores.
         * See Search::pop_from_focal_.
         */
        FloatT relax_open_score(FloatT oscore, FloatT eps) const
        { return oscore - (1.0-eps)*std::abs(oscore); }

        /** Returns true when a is 'better' than b, i.e. a is larger than b */
        bool cmp_open_score(FloatT a, FloatT b) const { return a > b; }

    };

    /** \private */
    struct MaxOutputState : public BaseState {
        FloatT g, h;
        MaxOutputState() : BaseState(), g(0.0), h(0.0) {}
    };

    struct MaxOutputHeuristic : public MaxHeuristic {
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
            FloatT h = compute_basic_output_heuristic_(search, out);

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

        /**
         * Is `a` 'better' than `b`? For this heuristic, `a` is better than
         * `b` when `a` is larger than `b` (maximizing).
         */
        bool cmp_open_score(FloatT a, FloatT b) const { return a > b; }
        bool cmp_open_score(const State& a, const State& b) const
        { return cmp_open_score(open_score(a), open_score(b)); }

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

    struct MinDistToExampleHeuristic : public MinHeuristic {
        using State = MinDistToExampleState;
        FloatT output_threshold;
        std::vector<FloatT> example;

        MinDistToExampleHeuristic(
                const std::vector<FloatT>& example,
                FloatT output_threshold)
            : output_threshold(output_threshold)
            , example(example)
        { }

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
            FloatT h = compute_basic_output_heuristic_(search, out);
            //std::cout << h << ", " << h2 << std::endl;

            if (!std::isinf(h) && (g+h) > output_threshold)
            {
                out.g = g;
                out.h = h;
                //out.dist = compute_lp_heuristic_<2>(search, out);
                out.dist = compute_linf_heuristic_(search, out);
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

        /**
         * Is `a` 'better' than `b`? For this heuristic, `a` is better than `b`
         * when `a` is less than `b` (minimizing).
         */
        bool cmp_open_score(FloatT a, FloatT b) const { return a < b; }
        bool cmp_open_score(const State& a, const State& b) const
        { return cmp_open_score(open_score(a), open_score(b)); }

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

        /**
         * Compute an underestimate of the distance to an example
         */
        template <int p>
        FloatT compute_lp_heuristic_(
                const Search<MinDistToExampleHeuristic>& s,
                const MinDistToExampleState& state) const
        {
            FloatT lp_h = 0.0;
            FloatT lp_state = 0.0;
            for (auto &&[feat_id, dom] : state.box)
            {
                FloatT x = example[feat_id];
                FloatT d = std::max({dom.lo - x, x - dom.hi, FloatT(0.0)});
                lp_state += std::pow(d, p);
            }

            s.workspace_.leafiter2.setup_flatbox(state.box); // do once
            for (size_t tree_index = state.indep_set + 1;
                    tree_index < s.at_.size(); ++tree_index)
            {
                FloatT min_lp = FLOATT_INF;
                const Tree& t = s.at_[tree_index];
                s.workspace_.leafiter2.setup_tree(t);
                NodeId leaf_id = -1;
                while ((leaf_id = s.workspace_.leafiter2.next()) != -1)
                {
                    BoxRef box = s.node_box_[tree_index][leaf_id];
                    if (box.is_invalid_box())
                        continue;

                    FloatT lp = lp_state;
                    for (auto &&[feat_id, dom] : box)
                    {
                        Domain dom0;
                        if (static_cast<size_t>(feat_id) <
                                s.workspace_.leafiter2.flatbox.size())
                            dom0 = s.workspace_.leafiter2.flatbox[feat_id];
                        Domain dom1 = dom.intersect(dom0);
                        FloatT x = example[feat_id];
                        FloatT d0 = std::max({dom0.lo - x, x - dom0.hi, FloatT(0.0)});
                        FloatT d1 = std::max({dom1.lo - x, x - dom1.hi, FloatT(0.0)});
                        lp = lp - std::pow(d0, p) + std::pow(d1, p);
                    }
                    min_lp = std::min(min_lp, lp); // pick the leaf with the lowest lp-distance
                }

                lp_h = std::max(min_lp, lp_h);
            }
            return lp_h;
        }

        FloatT compute_linf_heuristic_(
                const Search<MinDistToExampleHeuristic>& s,
                const MinDistToExampleState& state) const
        {
            FloatT lp_state = 0.0;
            for (auto &&[feat_id, dom] : state.box)
            {
                FloatT x = example[feat_id];
                FloatT d = std::max({dom.lo - x, x - dom.hi, FloatT(0.0)});
                lp_state = std::max(lp_state, d);
            }
            FloatT lp_h = lp_state;

            s.workspace_.leafiter2.setup_flatbox(state.box); // do once
            for (size_t tree_index = state.indep_set + 1;
                    tree_index < s.at_.size(); ++tree_index)
            {
                FloatT min_lp = FLOATT_INF;
                const Tree& t = s.at_[tree_index];
                s.workspace_.leafiter2.setup_tree(t);
                NodeId leaf_id = -1;
                while ((leaf_id = s.workspace_.leafiter2.next()) != -1)
                {
                    BoxRef box = s.node_box_[tree_index][leaf_id];
                    if (box.is_invalid_box())
                        continue;

                    FloatT lp = lp_state;
                    for (auto &&[feat_id, dom] : box)
                    {
                        Domain dom0;
                        if (static_cast<size_t>(feat_id) <
                                s.workspace_.leafiter2.flatbox.size())
                            dom0 = s.workspace_.leafiter2.flatbox[feat_id];
                        Domain dom1 = dom.intersect(dom0);
                        FloatT x = example[feat_id];
                        FloatT d1 = std::max({dom1.lo - x, x - dom1.hi, FloatT(0.0)});
                        lp = std::max(lp, d1);
                    }
                    min_lp = std::min(min_lp, lp); // pick the leaf with the lowest lp-distance
                }

                lp_h = std::max(min_lp, lp_h);
            }
            return lp_h;
        }

        void print_state(std::ostream& strm, const State& s)
        {
            strm << "State g=" << s.g << ", h=" << s.h
                << ", f=" << (s.g+s.h)
                << ", dist=" << s.dist
                << ", set=" << s.indep_set
                << std::endl;
        }
    };

} // namespace veritas
