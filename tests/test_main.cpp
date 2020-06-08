#include <functional>
#include <iostream>
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
    t.root().left().left().set_leaf_value(1.0);
    t.root().left().right().set_leaf_value(-1.0);
    t.root().right().set_leaf_value(1.0);
    at.add_tree(std::move(t));

    FeatInfo finfo(at, at, {}, true);

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

    //while (opt.step(box_filter, 0.0))
    while (opt.step())
    {
        std::cout << "================================ " << std::endl;
    }
}

void test_calhouse()
{
    //auto file = "tests/models/xgb-calhouse-very-easy.json";
    //std::shared_ptr<AddTree> at = std::make_shared<AddTree>(AddTree::from_json_file(file));
}

int main()
{
    test_simple();
}
