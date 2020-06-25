#include <functional>
#include <iostream>
#include <chrono>
#include "tree.h"
#include "graph.h"
#include "smt.h"

using namespace treeck;

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
    DomainStore store(finfo);
    KPartiteGraph g0(&store, at, finfo, 0);
    KPartiteGraph g1(&store, at, finfo, 1);
    KPartiteGraphOptimize opt(g0, g1);
    std::cout << g0 << std::endl;
    std::cout << g1 << std::endl;

    auto box_filter = [](const DomainBox&) {
        return true;
    };

    while (opt.step(box_filter, 0.0))
    //while (opt.step())
    {
        std::cout << "================================ " << std::endl;
    }
}

void test_img()
{
    auto file = "tests/models/xgb-img-very-easy.json";
    AddTree at = AddTree::from_json_file(file);
    AddTree dummy;

    //std::cout << at << std::endl;

    FeatInfo finfo(at, dummy, {}, true); // minimize
    DomainStore store(finfo);
    KPartiteGraph g0(&store, at, finfo, 0);
    KPartiteGraph dummyg(&store);
    KPartiteGraphOptimize opt(g0, dummyg);

    std::cout << g0 << std::endl;

    auto box_filter = [](const DomainBox&) {
        return true;
    };

    while (opt.step(box_filter, 50.4, 0.0))
    {
        std::cout << "================================ " << std::endl;
    }
}

void test_calhouse_bounds()
{
    auto file = "tests/models/xgb-calhouse-intermediate.json";
    AddTree at = AddTree::from_json_file(file);
    AddTree dummy;

    FeatInfo finfo(at, dummy, {}, true);
    DomainStore store(finfo);
    KPartiteGraph g0(&store);
    KPartiteGraph g1(&store, at, finfo, 0);
    KPartiteGraphOptimize opt(g0, g1);
    opt.set_eps(0.02, 0.02);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> timings;
    std::vector<double> num_steps;
    while (opt.num_candidate_cliques() < 1000000 && opt.get_eps() < 1.0)
    {
        if (!opt.steps(10))
            break;

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
    //test_simple();
    //test_img();
    test_calhouse_bounds();
}
