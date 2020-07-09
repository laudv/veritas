#include <functional>
#include <iostream>
#include <chrono>
#include "tree.h"
#include "graph.h"
#include "smt.h"

using namespace treeck;

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

    KPartiteGraphOptimize opt0(g0, g1);
    opt0.use_dyn_prog_heuristic();
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
    //opt.use_dyn_prog_heuristic();
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
    //opt.use_dyn_prog_heuristic();

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

int main()
{
    //test_very_simple();
    //test_simple();
    //test_img();
    //test_unconstrained_bounds("tests/models/xgb-calhouse-hard.json");
    //test_unconstrained_bounds("tests/models/xgb-mnist-yis0-hard.json");
    test_parallel("tests/models/xgb-calhouse-hard.json");
}
