/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_SEARCH_HPP
#define VERITAS_SEARCH_HPP

#include "domain.hpp"
#include "new_tree.hpp"
#include "block_store.hpp"
#include <iostream>
#include <chrono>

namespace veritas {

    using time_point = std::chrono::time_point<std::chrono::system_clock>;

    struct State {
        size_t parent; // index into 'Search::states_` vector

        TreeId tree_id; // the split of the node (tree_id, node_id) was added ...
        NodeId node_id; // ... to this state's box

        FloatT g; // sum of 'certain' leaf values
        FloatT h; // heuristic value

        size_t cache; // index into `Search::caches_`, or NO_CACHE

        inline FloatT fscore(FloatT eps = 1.0) const { return g + eps * h; }
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
            << "}";
    }

    struct Solution {
        size_t state_index;
        size_t solution_index;
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
            << "   - output: " << s.output << std::endl
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



    class Search;

    struct StateCmp {
        const Search& search;
        FloatT eps;

        bool operator()(size_t i, size_t j) const;

        inline bool operator()(const State& a, const State& b) const
        { return a.fscore() < b.fscore(); }
    };

    /** Buffer for info about state currently being expanded. */
    struct SearchWorkspace {
        Box box; // buffer for the Box of the current state
        std::vector<FeatId> box_feat_ids; // feat_ids of box in contiguous memory
        std::vector<NodeId> node_ids; // which node id was last used for each tree?
    };

    struct Stats {
        size_t num_steps;
        size_t num_impossible;

        Stats() : num_steps(0), num_impossible(0) {}
    };

    class Search {
        AddTree at_; // copy of the tree so we can modify freely
        StateCmp cmp_;
        StateCmp ara_cmp_; // TODO have two heaps, one for A*, one for ARA*
        SearchWorkspace workspace_;
        SearchWorkspace workspace_backup_;

        std::vector<State> states_;
        std::vector<Cache> caches_;
        std::vector<size_t> heap_; // indices into states_
        std::vector<SolutionRef> solutions_; // indices into states_

        time_point start_time;

        const FeatId FEAT_ID_SENTINEL = static_cast<FeatId>((1l<<31)-1);
        const size_t NO_CACHE = static_cast<size_t>(-1);

    public:
        size_t max_mem_size = static_cast<size_t>(4)*1024*1024*1024; // 4GB
        Stats stats;

        friend StateCmp;

        Search(const AddTree& at)
            : at_(at)
            , cmp_{*this, 1.0}
            , ara_cmp_{*this, 0.1}
            , start_time{std::chrono::system_clock::now()}
        {
            if (at.size() == 0)
                throw std::runtime_error("Search: empty AddTree");

            // root state with self reference
            states_.push_back({0, -1, -1, at.base_score, 0.0f, NO_CACHE});
            heap_.push_back(0);
            workspace_.node_ids.resize(at.size());
        }

        /** One step in the search, returns TRUE when search is done. */
        bool step()
        {
            if (heap_.empty())
                return true;

            size_t state_index = pq_pop();

            //std::cout << "state index " << state_index << std::endl;
            //std::cout << states_[state_index] << std::endl;

            visit_ancestors(state_index);
            //std::cout << "parent box " << get_workspace_box() << std::endl;
            //for (size_t c = 0; c < workspace_.node_ids.size(); ++c)
            //    std::cout << " * tree " << c << " at node " << workspace_.node_ids[c] << std::endl;
            //std::cout << "next tree " << next_tree << std::endl;
            
            State new_left_state, new_right_state;
            bool success = expand(state_index, new_left_state, new_right_state);

            if (success)
            {
                compute_fscore(new_left_state);
                compute_fscore(new_right_state);

                // add left if valid
                if (!std::isinf(new_left_state.fscore()))
                {
                    size_t left_state_index = states_.size();
                    states_.push_back(new_left_state);
                    pq_push(left_state_index);
                }
                else ++stats.num_impossible;

                // add right if valid
                if (!std::isinf(new_right_state.fscore()))
                {
                    size_t right_state_index = states_.size();
                    states_.push_back(new_right_state);
                    pq_push(right_state_index);
                }
                else ++stats.num_impossible;
            }
            else
            {
                solutions_.push_back({ state_index, time_since_start() });
            }

            ++stats.num_steps;
            return false;
        }

        bool steps(size_t num_steps)
        {
            for (size_t i = 0; i < num_steps; ++i)
                if (step())
                    return true; // we're done
            return false;
        }

        size_t num_solutions() const
        {
            return solutions_.size();
        }

        Solution get_solution(size_t solution_index)
        {
            auto&& [state_index, time] = solutions_.at(solution_index);
            visit_ancestors(state_index);

            return {
                state_index,
                solution_index,
                states_[state_index].g,
                workspace_.node_ids, // copy
                workspace_.box,      // copy
                time,
            };
        }

    private:
        void pq_push(size_t index)
        {
            heap_.push_back(index);
            std::push_heap(heap_.begin(), heap_.end(), cmp_);
        }
        size_t pq_pop()
        {
            std::pop_heap(heap_.begin(), heap_.end(), cmp_);
            size_t index = std::move(heap_.back());
            heap_.pop_back();
            return index;
        }

        void rebuild_heap() { std::make_heap(heap_.begin(), heap_.end(), cmp_); }

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

        void copy_box_feat_ids()
        {
            // copy feat_ids into separate array for fast search
            workspace_.box_feat_ids.clear();
            for (auto&& [k, v] : workspace_.box)
                workspace_.box_feat_ids.push_back(k);
        }

        BoxRef get_workspace_box() const { return { workspace_.box }; }

        /** Expand the given state. Returns TRUE on success, FALSE on failure
         * (ie current state cannot be expanded because it is a solution).
         * `new_*_state` args are output arguments. */
        bool expand(size_t state_index, State& new_left_state, State& new_right_state)
        {
            // assumes `workspace_` is properly filled, i.e., we called `visit_ancestors` for this state
            const State& state = states_[state_index];

            // refine nodes in trees in order
            size_t num_trees = at_.size();
            NodeId next_tree = (states_[state.parent].tree_id + 1) % num_trees;
            NodeId node_id = workspace_.node_ids[next_tree];

            // the node we expand cannot be a leaf
            size_t skip_count = 0;
            while (at_[next_tree].node_const(node_id).is_leaf())
            {
                // this is a solution! all nodes in all trees are leafs
                if (++skip_count == num_trees) return false;
                next_tree = (next_tree + 1) % num_trees;
                node_id = workspace_.node_ids[next_tree];
            }

            // construct new states
            new_left_state = new_right_state = {
                state_index,
                next_tree,
                -1,  // node_id different for left & right
                at_.base_score, // g, needs to be recomputed with h (= sum of leaf values in this state)
                0.0, // h, needs to be recomputed
                NO_CACHE,
            };

            new_left_state.node_id = at_[next_tree].node_const(node_id).left().id();
            new_right_state.node_id = at_[next_tree].node_const(node_id).right().id();

            return true; // successfully expanded
        }

        // assumes valid `workspace_` for parent state
        void compute_fscore(State& child_state)
        {
            workspace_backup_ = workspace_; // backup workspace by copy
            workspace_.node_ids[child_state.tree_id] = child_state.node_id;
            auto n = at_[child_state.tree_id].node_const(child_state.node_id);
            auto p = n.parent();

            // if this refinement fails, added node is incompatible with
            // parent's box (conflicting paths in trees)
            try { refine_box(workspace_.box, p.get_split(), n.is_left_child()); }
            catch (const std::runtime_error& e) { child_state.g = FLOATT_INF; return; }

            copy_box_feat_ids(); // for search in `compute_heuristic_at`
            workspace_.box_feat_ids.push_back(FEAT_ID_SENTINEL);

            for (TreeId tree_id = 0; tree_id < static_cast<TreeId>(at_.size()); ++tree_id)
            {
                auto n = at_[tree_id].node_const(workspace_.node_ids[tree_id]);
                if (n.is_leaf())
                    child_state.g += n.leaf_value();
                else
                    child_state.h += compute_heuristic_at(n);
            }

            // restore backup of workspace
            std::swap(workspace_, workspace_backup_);
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
                Domain box_dom = workspace_.box[r].domain;

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

        double time_since_start() const
        {
            auto now = std::chrono::system_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                    now-start_time).count() * 1e-6;
        }
    };
    
    bool
    StateCmp::operator()(size_t i, size_t j) const
    { return (*this)(search.states_[i], search.states_[j]); }


} // namespace veritas

#endif // VERITAS_SEARCH_HPP
