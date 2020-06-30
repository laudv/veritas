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
    DomainStore store;
    KPartiteGraph g0(&store, at, finfo, 0);
    KPartiteGraph g1(&store, at, finfo, 1);

    KPartiteGraphOptimize opt0(g0, g1);
    opt0.use_dyn_prog_heuristic();
    KPartiteGraphOptimize opt1(g0, g1);
    std::cout << g0 << std::endl;
    std::cout << g1 << std::endl;

    auto box_filter = [](const DomainBox&) {
        return true;
    };

    while (opt0.step(box_filter, 0.0001));
    std::cout << "DYN_PROG:" << std::endl;
    for (auto& s : opt0.solutions)
        std::cout << s.output0 << ", " << s.output1 << " (" << s.output1 - s.output0 << ')' << "  " << s.box << std::endl;
    std::cout << "nrejected " << opt0.nrejected << std::endl;
    std::cout << "nbox_filter_calls " << opt0.nbox_filter_calls << std::endl;
    std::cout << "nsteps " << std::get<0>(opt0.nsteps) << ", " << std::get<1>(opt0.nsteps) << std::endl << std::endl;

    while (opt1.step(box_filter, 0.0001));
    std::cout << "RECOMPUTE:" << std::endl;
    for (auto& s : opt1.solutions)
        std::cout << s.output0 << ", " << s.output1 << " (" << s.output1 - s.output0 << ')' << "  " << s.box << std::endl;
    std::cout << "nrejected " << opt1.nrejected << std::endl;
    std::cout << "nbox_filter_calls " << opt1.nbox_filter_calls << std::endl;
    std::cout << "nsteps " << std::get<0>(opt1.nsteps) << ", " << std::get<1>(opt1.nsteps) << std::endl;
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
    DomainStore store;
    KPartiteGraph g0(&store, at, finfo, 0);
    KPartiteGraph g1(&store, at, finfo, 1);
    KPartiteGraphOptimize opt(g0, g1);
    //opt.use_dyn_prog_heuristic();
    std::cout << g0 << std::endl;
    std::cout << g1 << std::endl;

    auto box_filter = [](const DomainBox&) {
        return true;
    };

    while (opt.step(box_filter, 0.0));

    for (auto& s : opt.solutions)
    {
        std::cout << s.output0 << ", " << s.output1 << " (" << s.output1 - s.output0 << ')'
            << "  " << s.box << std::endl;
    }
    std::cout << "nrejected " << opt.nrejected << std::endl;
    std::cout << "nsteps " << std::get<0>(opt.nsteps) << ", " << std::get<1>(opt.nsteps) << std::endl;
}

void test_img()
{
    auto file = "tests/models/xgb-img-very-easy.json";
    AddTree at = AddTree::from_json_file(file);
    AddTree dummy;

    //std::cout << at << std::endl;

    FeatInfo finfo(at, dummy, {}, true); // minimize
    DomainStore store;
    KPartiteGraph g0(&store, at, finfo, 0);
    KPartiteGraph dummyg(&store);
    KPartiteGraphOptimize opt(g0, dummyg);
    //opt.use_dyn_prog_heuristic();

    std::cout << g0 << std::endl;

    auto box_filter = [](const DomainBox&) {
        return true;
    };

    while (opt.step(box_filter, 50.4, 0.0))
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
    DomainStore store;
    store.set_max_mem_size(1024*1024*50);
    KPartiteGraph g0(&store);
    KPartiteGraph g1(&store, at, finfo, 0);
    KPartiteGraphOptimize opt(g0, g1); // maximize, g0 is dummy
    //opt.set_eps(0.02, 0.02);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> timings;
    std::vector<double> num_steps;
    std::cout << "bound, mem, nsteps" << std::endl;
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
            << ", " << (store.get_mem_size() / (1024*1024))
            << ", " << std::get<1>(opt.nsteps)
            << std::endl;

        while (timings.size() < opt.solutions.size())
        {
            double d = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start).count();
            timings.push_back(d / 1000000.0);
            num_steps.push_back(std::get<0>(opt.nsteps) + std::get<1>(opt.nsteps));
        }
    }

    for (size_t i = 0; i < opt.solutions.size(); ++i)
    {
        auto sol = opt.solutions[i].output1;
        auto eps = opt.epses[i];
        std::cout << sol << ", " << eps
            << ", " << num_steps[i]
            << ", " << timings[i]
            << std::endl;
    }
}

int main()
{
    //test_very_simple();
    //test_simple();
    //test_img();
    //test_unconstrained_bounds("tests/models/xgb-calhouse-hard.json");
    test_unconstrained_bounds("tests/models/xgb-mnist-yis0-hard.json");
}
