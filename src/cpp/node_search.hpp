/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
 *
 * This is not used.
*/

#ifndef VERITAS_NODE_SEARCH_HPP
#define VERITAS_NODE_SEARCH_HPP

#include "domain.hpp"
#include "tree.hpp"
#include "block_store.hpp"
#include <iostream>
#include <chrono>

namespace veritas {

    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    struct State {
        size_t parent; // index into 'NodeSearch::states_` vector

        TreeId tree_id; // the split of the node (tree_id, node_id) was added ...
        NodeId node_id; // ... to this state's box

        FloatT g; // sum of 'certain' leaf values
        FloatT h; // heuristic value

        size_t cache; // index into `NodeSearch::caches_`, or NO_CACHE

        bool is_expanded;
        FloatT eps;

        inline FloatT fscore(FloatT the_eps = 1.0) const { return g + the_eps * h; }
    };

    struct Cache {
        BlockStore<DomainPair>::Ref box;
        BlockStore<NodeId>::Ref node_ids;
    };

    std::ostream&
    operator<<(std::ostream& strm, const State& s)
    {
        return strm
            << "State {" << std::endl
            << "   - parent: " << s.parent << std::endl
            << "   - tree_id, node_id: " << s.tree_id << ", " << s.node_id << std::endl
            << "   - g, h: " << s.g << ", " << s.h << std::endl
            << "   - expanded?: " << s.is_expanded << std::endl
            << "}";
    }

    struct Solution {
        size_t state_index;
        size_t solution_index;
        FloatT eps;
        FloatT output;
        std::vector<NodeId> nodes; // one leaf node id per tree in addtree
        Box box;
        double time;
    };

    struct SolutionRef {
        size_t state_index;
        double time;
    };

    std::ostream&
    operator<<(std::ostream& strm, const Solution& s)
    {
        strm
            << "Solution {" << std::endl
            << "   - state, solution index: " << s.state_index << ", " << s.solution_index << std::endl
            << "   - output, eps: " << s.output << ", " << s.eps << std::endl
            << "   - nodes: ";

        if (s.nodes.size() > 0)
        {
            strm << s.nodes.at(0);
            for (auto it = s.nodes.begin()+1; it != s.nodes.end(); ++it)
                strm << ", " << *it;
        }

        return strm
            << std::endl
            << "   - box: " << BoxRef(s.box) << std::endl
            << "   - time: " << s.time << std::endl
            << "}";
    }



    class NodeSearch;

    struct StateCmp {
        const NodeSearch& search;
        FloatT eps;

        bool operator()(size_t i, size_t j) const;

        inline bool operator()(const State& a, const State& b) const
        { return a.fscore(eps) < b.fscore(eps); }
    };

    /** Buffer for info about state currently being expanded. */
    struct NodeSearchWorkspace {
        Box box; // buffer for the Box of the current state
        Box box2; // second buffer for compute_fscore
        std::vector<FeatId> box_feat_ids; // feat_ids of box in contiguous memory
        std::vector<NodeId> node_ids; // which node id was last used for each tree?
    };

    struct Snapshot {
        double time;
        size_t num_steps;
        size_t num_impossible;
        size_t num_solutions;
        size_t num_states;
        FloatT bound;
    };

    struct Stats {
        size_t num_steps;
        size_t num_impossible;
        size_t num_solutions;
        size_t num_states;

        std::vector<Snapshot> snapshots;

        Stats() : num_steps(0), num_impossible(0), num_solutions(0), num_states(0), snapshots{} {}

        void push_snapshot(double time, FloatT bound)
        {
            snapshots.push_back({
                time,
                num_steps,
                num_impossible,
                num_solutions,
                num_states,
                bound
            });
        }
    };

    class NodeSearch {
        AddTree at_;
        StateCmp cmp_;
        StateCmp ara_cmp_; // two heaps, one for A*, one for ARA*
        NodeSearchWorkspace workspace_;

        std::vector<State> states_;
        std::vector<Cache> caches_;
        std::vector<size_t> heap_; // indices into states_
        std::vector<size_t> ara_heap_; // indices into states_
        std::vector<SolutionRef> solutions_; // indices into states_

        time_point start_time_;

        const FeatId FEAT_ID_SENTINEL = static_cast<FeatId>((1l<<31)-1);
        const size_t NO_CACHE = static_cast<size_t>(-1);

    public:
        size_t max_mem_size = static_cast<size_t>(4)*1024*1024*1024; // 4GB
        Stats stats;

        friend StateCmp;

        NodeSearch(const AddTree& at)
            : at_(at)
            , cmp_{*this, 1.0}
            , ara_cmp_{*this, 0.1}
            , start_time_{std::chrono::system_clock::now()}
        {
            if (at.size() == 0)
                throw std::runtime_error("NodeSearch: empty AddTree");

            // root state with self reference
            
            states_.push_back({0, 0, 0, at.base_score, 0.0f, NO_CACHE, false, 1.0});
            heap_.push_back(0);
            workspace_.node_ids.resize(at.size());
        }

        /** One step in the search, returns TRUE when search is done. */
        bool step()
        {
            FloatT eps;
            size_t state_index;
            while (true)
            {
                if (heap_.empty())
                    return true;

                std::tie(state_index, eps) = pq_pop();
                if (!states_[state_index].is_expanded)
                    break;
            }

            //std::cout << "state index " << state_index << std::endl;
            //std::cout << states_[state_index] << std::endl;

            visit_ancestors(state_index);
            //std::cout << "parent box " << get_workspace_box() << std::endl;
            //for (size_t c = 0; c < workspace_.node_ids.size(); ++c)
            //    std::cout << " * tree " << c << " at node " << workspace_.node_ids[c] << std::endl;
            //std::cout << "next tree " << next_tree << std::endl;
            
            if (!is_solution()) // current state is solution -> uses info gather in workspace by visit_ancestors
            {
                expand(state_index, eps);
            }
            else
            {
                push_solution(state_index);

                ara_cmp_.eps = std::min(1.0, ara_cmp_.eps + 0.1);
                //std::cout << "ARA increase " << ara_cmp_.eps << std::endl;
                //std::cout << "no ARA states " << ara_heap_.size() << ", " << heap_.size() << std::endl;
                if (ara_cmp_.eps < 1.0)
                    std::make_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
                else
                    ara_heap_.clear();
            }

            ++stats.num_steps;
            return false;
        }

        bool steps(size_t num_steps)
        {
            for (size_t i = 0; i < num_steps; ++i)
                if (step())
                    return true; // we're done
            stats.push_snapshot(time_since_start(), current_bound());
            return false;
        }

        bool step_for(double num_seconds)
        {
            double start = time_since_start();

            bool done = false;
            size_t num_steps = 10;
            const size_t max_num_steps = 1000;
            size_t current_num_solutions = num_solutions();

            while (!done && current_num_solutions == num_solutions())
            {
                double dur = time_since_start() - start;
                done = steps(num_steps);
                num_steps = std::min(max_num_steps, num_steps*2);
                if (dur >= num_seconds)
                    break;
            }

            return done;
        }

        size_t num_solutions() const
        {
            return solutions_.size();
        }

        size_t num_states() const
        {
            return states_.size();
        }

        Solution get_solution(size_t solution_index)
        {
            auto&& [state_index, time] = solutions_.at(solution_index);
            visit_ancestors(state_index);

            return {
                state_index,
                solution_index,
                states_[state_index].eps,
                states_[state_index].g,
                workspace_.node_ids, // copy
                workspace_.box,      // copy
                time,
            };
        }

        /** seconds since the construction of the search */
        double time_since_start() const
        {
            auto now = std::chrono::system_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                    now-start_time_).count() * 1e-6;
        }

        FloatT current_bound() const
        {
            const State& s = states_[heap_.front()];
            return s.fscore();
        }

        FloatT get_eps() const { return ara_cmp_.eps; }

        void set_eps(FloatT eps)
        { ara_cmp_.eps = std::max<FloatT>(0.0, std::min<FloatT>(1.0, eps)); }

    private:
        void pq_push(size_t index)
        {
            heap_.push_back(index);
            std::push_heap(heap_.begin(), heap_.end(), cmp_);
            if (states_[index].g != at_.base_score && ara_cmp_.eps < 1.0)
            {
                ara_heap_.push_back(index);
                std::push_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
            }
        }
        std::tuple<size_t, FloatT> pq_pop()
        {
            if (!ara_heap_.empty() && stats.num_steps%2 == 1)
                return {pq_pop_arastar(), ara_cmp_.eps};
            return {pq_pop_astar(), 1.0};
        }

        size_t pq_pop_astar()
        {
            std::pop_heap(heap_.begin(), heap_.end(), cmp_);
            size_t index = std::move(heap_.back());
            heap_.pop_back();
            return index;
        }

        size_t pq_pop_arastar()
        {
            std::pop_heap(ara_heap_.begin(), ara_heap_.end(), ara_cmp_);
            size_t index = std::move(ara_heap_.back());
            ara_heap_.pop_back();
            return index;
        }

        /** Visit each previous state to (1) reconstruct current state's box
         * and (2) find the node positions of this state in each tree */
        void visit_ancestors(size_t state_index)
        {
            workspace_.box.clear();
            workspace_.box_feat_ids.clear();
            std::fill(workspace_.node_ids.begin(), workspace_.node_ids.end(), 0);

            // length of this loop is sum{path length in each tree}
            // quite long and many random memory access, but probably not as
            // much work as updating the heuristic value
            while (state_index != 0)
            {
                const State& state = states_[state_index];

                NodeId& node_id = workspace_.node_ids[state.tree_id];
                if (node_id == 0) node_id = state.node_id;

                Tree::ConstRef n = at_[state.tree_id].node_const(state.node_id);
                Tree::ConstRef p = n.parent();

                try { refine_box(workspace_.box, p.get_split(), n.is_left_child()); }
                catch (const std::runtime_error& e) {
                    std::cerr << "ERROR: no overlap, invalid state on heap" << std::endl;
                    throw e;
                }
                
                state_index = state.parent;
            }
        }

        BoxRef get_workspace_box() const { return { workspace_.box }; }

        /**
         * Expand the given state.
         *
         * assumes `workspace_` is properly filled, i.e., we called
         * `visit_ancestors` for this state
         */
        void expand(size_t state_index, FloatT eps)
        {
            states_[state_index].is_expanded = true;

            // refine to child nodes in each tree
            TreeId num_trees = static_cast<TreeId>(at_.size());
            TreeId next_tree = (states_[states_[state_index].parent].tree_id + 1) % num_trees;
            NodeId node_id = workspace_.node_ids[next_tree];

            int num_leafs = 0;
            while (at_[next_tree].node_const(node_id).is_leaf())
            {
                if (++num_leafs == num_trees) return;
                next_tree = (next_tree + 1) % num_trees;
                node_id = workspace_.node_ids[next_tree];
            }

            // construct new states: one moves to the left child node, the other to the right
            State left_state, right_state;
            left_state = right_state = {
                state_index,
                next_tree,
                -1,  // node_id different for left & right, set below
                at_.base_score, // g, needs to be recomputed with h (= sum of leaf values in this state)
                0.0, // h, needs to be recomputed
                NO_CACHE,
                false,
                eps
            };

            left_state.node_id = at_[next_tree].node_const(node_id).left().id();
            right_state.node_id = at_[next_tree].node_const(node_id).right().id();

            //std::cout << "left_id " << left_state.node_id
            //    << " right_id " << right_state.node_id
            //    << " tree_id " << tree_id
            //    << std::endl;

            compute_fscore(left_state);
            //std::cout << "    L " << states_[state_index].fscore() << " => " << left_state.fscore();
            //std::cout << "\n";

            compute_fscore(right_state);
            //std::cout << "    R " << states_[state_index].fscore() << " => " << right_state.fscore();
            //std::cout << "\n";

            if (!std::isinf(left_state.fscore()))
                push_state(std::move(left_state));
            else ++stats.num_impossible;
            if (!std::isinf(right_state.fscore()))
                push_state(std::move(right_state));
            else ++stats.num_impossible;
        }

        /**
         * If the nodes of this state are all leafs, then this is a solution.
         *
         * assumes `workspace_` is properly filled, i.e., we called
         * `visit_ancestors` for this state
         */
        bool is_solution() const 
        {
            TreeId num_trees = static_cast<TreeId>(at_.size());
            for (TreeId tree_id = 0; tree_id < num_trees; ++tree_id)
            {
                NodeId node_id = workspace_.node_ids[tree_id];
                if (at_[tree_id].node_const(node_id).is_internal())
                    return false;
            }
            return true;
        }

        // assumes valid `workspace_` for parent state
        void compute_fscore(State& child_state)
        {
            workspace_.box2 = workspace_.box;
            workspace_.node_ids[child_state.tree_id] = child_state.node_id;
            auto n = at_[child_state.tree_id].node_const(child_state.node_id);
            auto p = n.parent();

            // copy feat_ids into separate array for fast search
            workspace_.box_feat_ids.clear();
            for (auto&& [k, v] : workspace_.box2)
                workspace_.box_feat_ids.push_back(k);
            workspace_.box_feat_ids.push_back(FEAT_ID_SENTINEL);

            // update box for child
            // if refinement fails, added node is incompatible (conflicting paths)
            if (!refine_box(workspace_.box2, p.get_split(), n.is_left_child()))
            {
                child_state.g = FLOATT_INF;
            }
            else // box valid
            {
                TreeId num_trees = static_cast<TreeId>(at_.size());
                for (TreeId tree_id = 0; tree_id < num_trees; ++tree_id)
                {
                    auto n = at_[tree_id].node_const(workspace_.node_ids[tree_id]);
                    if (n.is_leaf())
                        child_state.g += n.leaf_value();
                    else
                        child_state.h += compute_heuristic_at(n);
                }
            }

            // recover parent state node id
            workspace_.node_ids[child_state.tree_id] = at_[child_state.tree_id]
                .node_const(child_state.node_id)
                .parent()
                .id();
        }

        FloatT compute_heuristic_at(Tree::ConstRef n)
        {
            if (n.is_leaf())
                return n.leaf_value();

            const LtSplit& split = n.get_split();
            size_t r = linear_search(workspace_.box_feat_ids, split.feat_id);

            // a domain for split.feat_id found
            if (workspace_.box_feat_ids[r] == split.feat_id) // safe because FEAT_ID_SENTINEL
            {
                Domain ldom, rdom;
                std::tie(ldom, rdom) = split.get_domains();
                Domain box_dom = workspace_.box2[r].domain;

                FloatT h = -FLOATT_INF;
                if (ldom.overlaps(box_dom))
                {
                    h = std::max(h, compute_heuristic_at(n.left()));
                }
                if (rdom.overlaps(box_dom))
                {
                    h = std::max(h, compute_heuristic_at(n.right()));
                }
                return h;
            }
            else // no restrictions for split.feat_id, continue in both subtrees
            {
                return std::max(compute_heuristic_at(n.left()),
                        compute_heuristic_at(n.right()));
            }
        }

        void push_solution(size_t state_index)
        {
            states_[state_index].is_expanded = true;
            solutions_.push_back({ state_index, time_since_start() });

            for (size_t i = solutions_.size()-1; i > 0; --i)
            {
                State& s1 = states_[solutions_[i-1].state_index];
                State& s2 = states_[solutions_[i].state_index];
                if (s1.fscore() < s2.fscore())
                    std::swap(s1, s2);
            }

            stats.num_solutions = solutions_.size();
        }

        void push_state(State&& state)
        {
            size_t state_index = states_.size();
            states_.push_back(state);
            pq_push(state_index);
            stats.num_states = states_.size();
        }
    };
    
    bool
    StateCmp::operator()(size_t i, size_t j) const
    { return (*this)(search.states_[i], search.states_[j]); }


} // namespace veritas

#endif // VERITAS_NODE_SEARCH_HPP
