#include <iostream>
#include "new_tree.hpp"


using namespace veritas;

/*
void test_very_simple()
{
    AddTree at;
    AddTree::TreeT t;
    t.root().split(LtSplit(0, 2.0));
    t.root().left().set_leaf_value(-1.0);
    t.root().right().set_leaf_value(1.0);
    at.add_tree(std::move(t));

    FeatInfo finfo(at, at, {}, true);

    std::cout << at << std::endl;
    KPartiteGraph g0(at, finfo, 0);
    KPartiteGraph g1(at, finfo, 1);

    KPartiteGraphOptimize opt0(g0, g1, KPartiteGraphOptimize::Heuristic::DYN_PROG);
    KPartiteGraphOptimize opt1(g0, g1);
    std::cout << g0 << std::endl;
    std::cout << g1 << std::endl;

    auto f = [](DomainStore&) {
        return true;
    };

    while (opt0.step(f, 0.0001));
    std::cout << "DYN_PROG:" << std::endl;
    for (auto& s : opt0.solutions)
        std::cout << s.output0 << ", " << s.output1 << " (" << s.output1 - s.output0 << ')' << "  " << s.box << std::endl;
    std::cout << "num_rejected " << opt0.num_rejected << std::endl;
    std::cout << "num_box_checks " << opt0.num_box_checks << std::endl;
    std::cout << "num_steps " << std::get<0>(opt0.num_steps) << ", " << std::get<1>(opt0.num_steps) << std::endl << std::endl;

    while (opt1.step(f, 0.0001));
    std::cout << "RECOMPUTE:" << std::endl;
    for (auto& s : opt1.solutions)
        std::cout << s.output0 << ", " << s.output1 << " (" << s.output1 - s.output0 << ')' << "  " << s.box << std::endl;
    std::cout << "num_rejected " << opt1.num_rejected << std::endl;
    std::cout << "num_box_checks " << opt1.num_box_checks << std::endl;
    std::cout << "num_steps " << std::get<0>(opt1.num_steps) << ", " << std::get<1>(opt1.num_steps) << std::endl;
}

void test_simple()
{
    AddTree at;
    AddTree::TreeT t;
    t.root().split(LtSplit(0, 1.24));
    t.root().left().set_leaf_value(-1.0);
    t.root().right().set_leaf_value(1.0);
    at.add_tree(std::move(t));
    t = AddTree::TreeT();
    t.root().split(LtSplit(0, 1.52));
    t.root().left().split(BoolSplit(1));
    t.root().left().left().set_leaf_value(4.0);
    t.root().left().right().set_leaf_value(-4.0);
    t.root().right().set_leaf_value(2.0);
    at.add_tree(std::move(t));

    FeatInfo finfo(at, at, {1}, true);

    std::cout << at << std::endl;
    KPartiteGraph g0(at, finfo, 0);
    KPartiteGraph g1(at, finfo, 1);
    KPartiteGraphOptimize opt(g0, g1);
    std::cout << g0 << std::endl;
    std::cout << g1 << std::endl;

    auto f = [](DomainStore&) {
        return true;
    };

    while (opt.step(f, 0.0));

    for (auto& s : opt.solutions)
    {
        std::cout << s.output0 << ", " << s.output1 << " (" << s.output1 - s.output0 << ')'
            << "  " << s.box << std::endl;
    }
    std::cout << "num_rejected " << opt.num_rejected << std::endl;
    std::cout << "num_steps " << std::get<0>(opt.num_steps) << ", " << std::get<1>(opt.num_steps) << std::endl;
}

void test_img()
{
    auto file = "tests/models/xgb-img-very-easy.json";
    AddTree at = AddTree::from_json_file(file);
    AddTree dummy;

    //std::cout << at << std::endl;

    FeatInfo finfo(at, dummy, {}, true); // minimize
    KPartiteGraph g0(at, finfo, 0);
    KPartiteGraph dummyg;
    KPartiteGraphOptimize opt(g0, dummyg);

    std::cout << g0 << std::endl;

    auto f = [](DomainStore&) {
        return true;
    };

    while (opt.step(f, 50.4, 0.0))
    {
        std::cout << "================================ " << std::endl;
    }

    for (auto& s : opt.solutions)
        std::cout << s.output0 << " " << s.box << std::endl;
}

void test_unconstrained_bounds(const char *model)
{
    AddTree at = AddTree::from_json_file(model);
    AddTree dummy;

    FeatInfo finfo(at, dummy, {}, true);
    KPartiteGraph g0;
    KPartiteGraph g1(at, finfo, 0);
    KPartiteGraphOptimize opt(g0, g1); // maximize, g0 is dummy
    opt.store().set_max_mem_size(1024*1024*50);
    //opt.set_eps(0.02, 0.02);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> timings;
    std::vector<double> num_steps;
    std::cout << "bound, mem, num_steps" << std::endl;
    while (opt.num_candidate_cliques() < 1000000)
    {
        try {
            if (!opt.steps(1000))
                break;
        } catch(...) {
            std::cout << "out of memory" << std::endl;
            break;
        }

        std::cout << std::get<1>(opt.current_bounds())
            << ", " << (opt.store().get_mem_size() / (1024*1024))
            << ", " << std::get<1>(opt.num_steps)
            << std::endl;

        while (timings.size() < opt.solutions.size())
        {
            double d = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start).count();
            timings.push_back(d / 1000000.0);
            num_steps.push_back(std::get<0>(opt.num_steps) + std::get<1>(opt.num_steps));
        }
    }

    for (size_t i = 0; i < opt.solutions.size(); ++i)
    {
        auto& sol = opt.solutions[i];
        std::cout << sol.output1 << ", " << sol.eps
            << ", " << num_steps[i]
            << ", " << sol.time
            << std::endl;
    }
}

void test_parallel(const char *model)
{
    AddTree at = AddTree::from_json_file(model);
    AddTree dummy;

    FeatInfo finfo(at, dummy, {}, true);
    KPartiteGraph g0;
    KPartiteGraph g1(at, finfo, 0);
    KPartiteGraphOptimize opt(g0, g1); // maximize, g0 is dummy
    opt.store().set_max_mem_size(1024*1024*50);
    opt.set_eps(0.1);

    //std::cout << "steps(100)" << std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    opt.steps(10);
    //auto stop = std::chrono::high_resolution_clock::now();
    //double dur = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
    //std::cout << "steps(100) done: " << opt.num_candidate_cliques()
    //    << " cliques in " << (dur * 1e-6)
    //    << " bound=" << std::get<1>(opt.current_bounds())
    //    << " num_steps=" << std::get<1>(opt.num_steps)
    //    << std::endl;

    KPartiteGraphParOpt paropt(6, opt);
    //paropt.set_box_filter([]() { return [](const DomainBox&) { return true; }; });
    while (paropt.get_eps() != 1.0 || paropt.num_new_valid_solutions() == 0)
    {
        if (paropt.num_new_valid_solutions() > 0) {
            FloatT new_eps = std::min(1.0, paropt.get_eps() + 0.05);
            std::cout << "increasing eps " << paropt.get_eps() << "->" << new_eps << std::endl;
            paropt.set_eps(new_eps);
        }
        paropt.steps_for(50);
    }
    std::cout << "joining..." << std::endl;
    //std::this_thread::sleep_for(std::chrono::seconds(1));
    paropt.join_all();
    std::cout << "joined!" << std::endl;

    for (size_t i = 0; i < paropt.num_threads(); ++i)
    {
        for (auto& sol : paropt.worker_opt(i).solutions)
        {
            std::cout << "worker " << i << " " << sol.output1
                << ", " << sol.eps
                << ", " << sol.time
                << ", " << sol.is_valid
                << std::endl;
        }
    }
}

void test_box_checker1()
{
    BoxChecker checker{3};
    std::vector<DomainPair> box{ {0, {2, INFINITY}}, {1, {-5.0, 5.0}}, {2, {-100, 50}} };

    int sumid = checker.add_sum(0, 1);
    std::cout << "sumid " << sumid << std::endl;
    int pow2id = checker.add_pow2(sumid);
    int sqrtid = checker.add_sqrt(pow2id);
    int constid = checker.add_const(10.0);
    int subid = checker.add_sub(sqrtid, constid);
    checker.add_eq(subid, 2);

    checker.copy_from_workspace(box);
    auto st = checker.update();
    std::cout << "update " << st << std::endl;
    st = checker.update();
    std::cout << "update " << st << std::endl;
    st = checker.update();
    std::cout << "update " << st << std::endl;
    st = checker.update();
    std::cout << "update " << st << std::endl;
    checker.copy_to_workspace(box);

    std::cout << std::endl << "----" << std::endl;
    for (auto p : box)
    {
        std::cout << "box: " << p.first << ", " << p.second << std::endl;
    }

    std::cout << "expr dom sum: " << checker.get_expr_dom(sumid) << std::endl;
    std::cout << "expr dom pow2: " << checker.get_expr_dom(pow2id) << std::endl;
    std::cout << "expr dom sqrt: " << checker.get_expr_dom(sqrtid) << std::endl;
    std::cout << "expr dom const: " << checker.get_expr_dom(constid) << std::endl;
    std::cout << "expr dom sub: " << checker.get_expr_dom(subid) << std::endl;
}

void test_box_checker2()
{
    BoxChecker checker{3};
    std::vector<DomainPair> box{{1, TRUE_DOMAIN}, {2, TRUE_DOMAIN}};

    checker.add_k_out_of_n({0, 1, 2}, 2, true);

    checker.copy_from_workspace(box);
    auto st = checker.update();
    std::cout << "update " << st << std::endl;
    st = checker.update();
    std::cout << "update " << st << std::endl;
    checker.copy_to_workspace(box);

    std::cout << std::endl << "----" << std::endl;
    for (auto p : box)
    {
        std::cout << "box: " << p.first << ", " << p.second << std::endl;
    }
}
*/

int main()
{
    //test_very_simple();
    //test_simple();
    //test_img();
    //test_unconstrained_bounds("tests/models/xgb-calhouse-hard.json");
    //test_unconstrained_bounds("tests/models/xgb-mnist-yis0-hard.json");
    //test_parallel("tests/models/xgb-calhouse-hard.json");
    //test_box_checker1();
    //test_box_checker2();

    std::cout << "yo" << std::endl;

    Tree tree;
    auto n = tree.root();
    n.split({1, 12.3});
    //n.left().set_leaf_value(12.4);
    //n.right().set_leaf_value(9.4);
    std::cout << "is root? " << n.is_root() << std::endl;
    std::cout << "number of leafs " << n.num_leafs() << std::endl;
}
