#include <sstream>
#include "util.h"
#include "smt.h"

namespace treeck {

    ReuseFeatIdMapper::ReuseFeatIdMapper(
            const AddTree& at0,
            const AddTree& at1,
            const std::unordered_set<FeatId>& matches,
            bool match_is_reuse)
        : id_map0_{}
        , id_map1_{}
    {
        // quick and dirty way to get all the used feat_ids in at0
        // not at all the fastest, but that does not matter, we only do it once
        auto at0_splits = at0.get_splits();
        auto at1_splits = at1.get_splits();

        int max_feat_id0 = 0;
        for (auto&& [feat_id, _] : at0_splits)
            max_feat_id0 = std::max(max_feat_id0, feat_id);
        id_map0_.resize(max_feat_id0+1, -1);
        for (auto&& [feat_id, _] : at0_splits)
            id_map0_[feat_id] = feat_id;

        int max_feat_id1 = 0;
        for (auto&& [feat_id, _] : at1_splits)
            max_feat_id1 = std::max(max_feat_id1, feat_id);
        id_map1_.resize(max_feat_id1+1, -1);
        for (auto&& [feat_id, _] : at1_splits)
        {
            bool in_matches = matches.find(feat_id) != matches.end();
            if (in_matches == match_is_reuse)
                id_map1_[feat_id] = feat_id; // same id as at0!
            else
                id_map1_[feat_id] = id_map1_.size() + feat_id;
        }

        std::cout << "MAPPING:" << std::endl;
        for (int i = 0; i < id_map0_.size(); ++i)
            std::cout << "  0: " << i << ", " << id_map0_[i] << " is_reused: " << is_reused(i) << std::endl;
        for (int i = 0; i < id_map1_.size(); ++i)
            std::cout << "  1: " << i << ", " << id_map1_[i] << " is_reused: " << is_reused(i) << std::endl;
    }

    bool
    ReuseFeatIdMapper::is_feat_id_used(int instance, FeatId feat_id) const
    {
        const auto& id_map = instance==0 ? id_map0_ : id_map1_;
        if (feat_id < id_map.size())
            return id_map[feat_id] != -1;
        return false;
    }

    bool
    ReuseFeatIdMapper::is_reused(FeatId feat_id) const
    {
        if (feat_id >= id_map0_.size() || feat_id >= id_map1_.size())
            return false;

        int id0 = id_map0_[feat_id];
        return id0 != -1 && id0 == id_map1_[feat_id];
    }

    int
    ReuseFeatIdMapper::operator()(FeatId feat_id) const
    {
        if (feat_id < id_map1_.size())
            return id_map1_[feat_id];
        else return -1;
    }

    std::vector<FeatId>
    ReuseFeatIdMapper::get_used_feat_ids(int instance) const
    {
        const auto& id_map = instance==0 ? id_map0_ : id_map1_;
        std::vector<FeatId> output;
        for (int i = 0; i < id_map.size(); ++i)
        {
            int id = id_map[i];
            if (id != -1)
                output.push_back(i);
        }
        return output;
    }


    //Solver::Solver(const AddTree& at)
    //    : fmap_{} // we won't be using this
    //    , ctx_{}
    //    , solver_{ctx_}
    //{
    //    // preprocess split values so we don't have to do the stupid float ->
    //    // str -> z3::expr repeatedly
    //    fill_const_cache(at);
    //    fill_var_map(0, at);

    //    std::cout << "single tree solver" << std::endl;
    //    std::cout << "size of const_cache_ " << const_cache_.size() << std::endl;
    //    for (auto&& [i, v] : const_cache_)
    //        std::cout << "const_cache_[" << i << "] " << v << std::endl;
    //}

    Solver::Solver(
            const AddTree& at0,
            const AddTree& at1,
            std::unordered_set<FeatId> matches,
            bool match_is_reuse)
        : fmap_{at0, at1, matches, match_is_reuse}
        , ctx_{}
        , solver_{ctx_}
    {
        fill_const_cache(at0);
        fill_const_cache(at1);
        fill_var_map(0, at0);
        fill_var_map(1, at1);

        std::cout << "two tree solver" << std::endl;
        std::cout << "size of const_cache_ " << const_cache_.size() << std::endl;
        for (auto&& [i, v] : const_cache_)
            std::cout << "const_cache_[" << i << "] " << v << std::endl;
    }

    void
    Solver::fill_const_cache(const AddTree& at)
    {
        for (const auto& tree : at.trees())
            fill_const_cache(tree.root());
    }

    void
    Solver::fill_const_cache(AddTree::TreeT::CRef node)
    {
        if (!node.is_internal())
            return;

        visit_split(
            [this](const LtSplit& s) {
                float_to_z3(s.split_value);
            },
            [](const BoolSplit& s) {},
            node.get_split());

        fill_const_cache(node.left());
        fill_const_cache(node.right());
    }

    void
    Solver::fill_var_map(int instance, const AddTree& at)
    {
        for (const auto& tree : at.trees())
            fill_var_map(instance, tree.root());
    }

    void
    Solver::fill_var_map(int instance, AddTree::TreeT::CRef node)
    {
        if (!node.is_internal())
            return;

        visit_split(
            [this, instance](const LtSplit& s) {
                mk_real_xvar(instance, s.feat_id);
            },
            [this, instance](const BoolSplit& s) {
                mk_bool_xvar(instance, s.feat_id);
            },
            node.get_split());

        fill_var_map(instance, node.left());
        fill_var_map(instance, node.right());
    }

    z3::expr&
    Solver::float_to_z3(FloatT value)
    {
        static_assert(sizeof(FloatT) == sizeof(int));
        int i = *reinterpret_cast<int *>(&value);
        auto fd = const_cache_.find(i);

        if (fd != const_cache_.end())
            return fd->second;

        ss_.seekp(0);
        ss_.seekg(0);
        ss_.str("");
        ss_.clear();
        ss_ << value;

        return const_cache_.emplace(i, ctx_.real_val(ss_.str().c_str())).first->second;
    }

    z3::solver&
    Solver::get_z3()
    {
        return solver_;
    }

    z3::context&
    Solver::get_z3_ctx()
    {
        return ctx_;
    }

    void
    Solver::parse_smt(const char *smt)
    {
        solver_.reset();
        std::stringstream ss;
        for (auto &&[id, pair] : var_map_)
        {
            std::cout << "var_map_[" << id << "] = " << std::get<0>(pair) << ", " << std::get<1>(pair) << std::endl;
            const std::string& xvar_name = std::get<0>(pair);
            const z3::expr& xvar = std::get<1>(pair);

            ss << "(declare-fun " << xvar_name << "() ";
            if (xvar.is_real()) ss << "Real";
            if (xvar.is_bool()) ss << "Bool";
            ss << ')' << std::endl;
        }
        ss << smt;
        std::cout << "SMT: " << std::endl << ss.str() << std::endl;
        std::cout << "=====" << std::endl;
        solver_.from_string(ss.str().c_str());
    }

    const ReuseFeatIdMapper&
    Solver::fmap() const
    {
        return fmap_;
    }

    std::string
    Solver::var_name(int instance, FeatId feat_id) const
    {
        if (instance != 0 && fmap_.is_reused(feat_id))
        {
            std::cout << "WARNING! reusing " << instance << ", " << feat_id << std::endl;
            instance = 0; // use same var as instance0
        }

        std::stringstream s;
        s << "x" << instance << "_" << feat_id;
        std::cout << "var_name " << instance << ", " << feat_id << " => " << s.str() << std::endl;
        return s.str();
    }

    int
    Solver::xvar_id(int instance, FeatId feat_id) const
    {
        return instance == 0
            ? static_cast<int>(feat_id)
            : fmap_(feat_id);
    }

    void
    Solver::mk_real_xvar(int instance, FeatId feat_id)
    {
        int id = xvar_id(instance, feat_id);

        if (var_map_.find(id) != var_map_.end())
            return;

        std::string s = var_name(instance, feat_id);
        z3::expr x = ctx_.real_const(s.c_str());
        var_map_.insert({id, {std::move(s), std::move(x)}});
    }
    
    void
    Solver::mk_bool_xvar(int instance, FeatId feat_id)
    {
        int id = xvar_id(instance, feat_id);

        if (var_map_.find(id) != var_map_.end())
            return;

        std::string s = var_name(instance, feat_id);
        z3::expr x = ctx_.bool_const(s.c_str());
        var_map_.insert({id, {std::move(s), std::move(x)}});
    }

    std::tuple<std::string, z3::expr>&
    Solver::xvar_tuple(int instance, FeatId feat_id)
    {
        int id = xvar_id(instance, feat_id);
        auto fd = var_map_.find(id);
        if (var_map_.find(id) != var_map_.end())
            return fd->second;
        throw std::runtime_error("xvar not found");
    }

    const std::string&
    Solver::xvar_name(int instance, FeatId feat_id)
    {
        return std::get<0>(xvar_tuple(instance, feat_id));
    }

    z3::expr&
    Solver::xvar(int instance, FeatId feat_id)
    {
        return std::get<1>(xvar_tuple(instance, feat_id));
    }

    z3::expr&
    Solver::xvar_by_id(int id)
    {
        auto fd = var_map_.find(id);
        if (fd == var_map_.end())
            throw std::runtime_error("unknown variable id");
        return std::get<1>(fd->second);
    }

    z3::expr
    Solver::domain_to_z3(int id, const Domain& dom)
    {
        return visit_domain(
            [this, id](const RealDomain& d) -> z3::expr {
                const z3::expr& xvar = xvar_by_id(id);
                if (!xvar.is_real())
                    throw std::runtime_error("RealDomain for non-real variable");
                if (!d.lo_is_inf() && !d.hi_is_inf())
                    return (float_to_z3(d.lo) <= xvar) && (xvar < float_to_z3(d.hi));
                if (!d.lo_is_inf())
                    return float_to_z3(d.lo) <= xvar;
                if (!d.hi_is_inf())
                    return xvar < float_to_z3(d.hi);
                else
                    throw std::runtime_error("unconstrained real domain");
            },
            [this, id](const BoolDomain& d) -> z3::expr {
                const z3::expr& xvar = xvar_by_id(id);
                if (!xvar.is_bool())
                    throw std::runtime_error("BoolDomain for non-bool variable");
                if (d.is_true())
                    return xvar;
                else if (d.is_false())
                    return !xvar;
                else
                    throw std::runtime_error("unconstrained bool domain");
            },
            dom);
    }

    template <typename I>
    z3::expr
    Solver::domains_to_z3(I begin, I end)
    {
        if (begin == end)
            return ctx_.bool_val(true);

        int id;
        Domain dom;

        std::tie(id, dom) = *(begin++);
        z3::expr e = domain_to_z3(id, dom);

        for (; begin != end; ++begin)
        {
            std::tie(id, dom) = *begin;
            e = e && domain_to_z3(id, dom);
        }

        return e;
    }

    template
    z3::expr
    Solver::domains_to_z3(
            std::vector<std::pair<int, Domain>>::const_iterator begin,
            std::vector<std::pair<int, Domain>>::const_iterator end);

    bool
    Solver::check(z3::expr& e)
    {
        return solver_.check(1, &e) == z3::sat;
    }

} /* namespace teeck */
