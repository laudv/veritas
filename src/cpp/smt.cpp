#include <sstream>
#include "util.h"
#include "smt.h"

namespace treeck {

    Solver::Solver(
            const FeatInfo* finfo,
            const AddTree& at0,
            const AddTree& at1)
        : finfo_{finfo}
        , ctx_{}
        , solver_{ctx_}
    {
        fill_const_cache(at0);
        fill_const_cache(at1);
        fill_var_map(0, at0);
        fill_var_map(1, at1);

        //std::cout << "size of const_cache_ " << const_cache_.size() << std::endl;
        //for (auto&& [i, v] : const_cache_)
        //    std::cout << "const_cache_[" << i << "] " << v << std::endl;
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
            [](const BoolSplit&) {},
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
        static_assert(sizeof(FloatT) == sizeof(uint32_t));
        uint32_t i = *reinterpret_cast<uint32_t *>(&value);
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
            //std::cout << "var_map_[" << id << "] = " << std::get<0>(pair) << ", " << std::get<1>(pair) << std::endl;
            const std::string& xvar_name = std::get<0>(pair);
            const z3::expr& xvar = std::get<1>(pair);

            ss << "(declare-fun " << xvar_name << "() ";
            if (xvar.is_real()) ss << "Real";
            if (xvar.is_bool()) ss << "Bool";
            ss << ')' << std::endl;
        }
        ss << smt;
        //std::cout << "SMT: " << std::endl << ss.str() << std::endl;
        //std::cout << "=====" << std::endl;
        solver_.from_string(ss.str().c_str());
    }

    std::string
    Solver::var_name(int instance, FeatId feat_id) const
    {
        int id = finfo_->get_id(instance, feat_id);
        if (instance != 0 && finfo_->is_instance0_id(id))
        {
            std::cout << "WARNING! reusing " << instance << ", " << feat_id
                << '(' << id << ')' << std::endl;
            instance = 0; // use same var as instance0
        }

        std::stringstream s;
        s << "x" << instance << "_" << feat_id;
        //std::cout << "var_name " << instance << ", " << feat_id << " => " << s.str() << std::endl;
        return s.str();
    }

    int
    Solver::xvar_id(int instance, FeatId feat_id) const
    {
        return finfo_->get_id(instance, feat_id);
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
    Solver::domain_to_z3(int id, const RealDomain& d)
    {
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
    }

    z3::expr
    Solver::domain_to_z3(int id, const BoolDomain& d)
    {
        const z3::expr& xvar = xvar_by_id(id);
        if (!xvar.is_bool())
            throw std::runtime_error("BoolDomain for non-bool variable");
        if (d.is_true())
            return xvar;
        else if (d.is_false())
            return !xvar;
        else
            throw std::runtime_error("unconstrained bool domain");
    }

    template <typename I>
    z3::expr
    Solver::domains_to_z3(I begin, I end)
    {
        if (begin == end)
            return ctx_.bool_val(true);

        //int id;
        //Domain dom;
        //std::tie(id, dom) = *(begin++);
        //z3::expr e = domain_to_z3(id, dom);

        //for (; begin != end; ++begin)
        //{
        //    std::tie(id, dom) = *begin;
        //    e = e && domain_to_z3(id, dom);
        //}
        //
        z3::expr e = ctx_.bool_val(true);
        for (int id = 0; begin != end; ++begin, ++id)
        {
            DomainT dom = *begin;
            if (dom.is_everything()) continue;
            if (finfo_->is_real(id))
            {
                e = e && domain_to_z3(id, dom);
            }
            else
            {
                BoolDomain bdom{dom == TRUE_DOMAIN};
                e = e && domain_to_z3(id, bdom);
            }

            //visit_domain(
            //    [this, &e, id](const RealDomain& d) { if (!d.is_everything()) e = e && domain_to_z3(id, d); },
            //    [this, &e, id](const BoolDomain& d) { if (!d.is_everything()) e = e && domain_to_z3(id, d); },
            //    *begin);
        }
        return e;
    }

    template
    z3::expr
    Solver::domains_to_z3(
            DomainBox::const_iterator begin,
            DomainBox::const_iterator end);

    bool
    Solver::check(z3::expr& e)
    {
        return solver_.check(1, &e) == z3::sat;
    }

} /* namespace teeck */
