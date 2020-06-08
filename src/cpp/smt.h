/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef TREECK_SMT_H
#define TREECK_SMT_H

#include <unordered_map>
#include <z3++.h>

#include "domain.h"
#include "tree.h"
#include "graph.h"

namespace treeck {

    class Solver  {
        FeatInfo finfo_;
        z3::context ctx_;
        z3::solver solver_;

        std::unordered_map<int, std::tuple<std::string, z3::expr>> var_map_;
        std::unordered_map<int, z3::expr> const_cache_;
        std::stringstream ss_;

    private:
        void fill_const_cache(const AddTree& at);
        void fill_const_cache(AddTree::TreeT::CRef node);

        void fill_var_map(int instance, const AddTree& at);
        void fill_var_map(int instance, AddTree::TreeT::CRef node);

        std::string var_name(int instance, FeatId feat_id) const;
        void mk_real_xvar(int instance, FeatId feat_id);
        void mk_bool_xvar(int instance, FeatId feat_id);
        std::tuple<std::string, z3::expr>& xvar_tuple(int instance, FeatId feat_id);

    public:
        //Solver(const AddTree& at);
        Solver(const AddTree& at0, const AddTree& at1,
                std::unordered_set<FeatId> matches,
                bool match_is_reuse);

        z3::solver& get_z3();
        z3::context& get_z3_ctx();
        void parse_smt(const char *smt);

        const FeatInfo& finfo() const;

        z3::expr& float_to_z3(FloatT value);

        int xvar_id(int instance, FeatId feat_id) const;
        const std::string& xvar_name(int instance, FeatId feat_id);
        z3::expr& xvar_by_id(int id);
        z3::expr& xvar(int instance, FeatId feat_id);

        z3::expr domain_to_z3(int id, const Domain& dom);

        template <typename I>
        z3::expr domains_to_z3(I begin, I end);

        bool check(z3::expr& e);
    };

} /* namespace treeck */

#endif // TREECK_SMT_H
