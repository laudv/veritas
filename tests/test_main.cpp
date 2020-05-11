#include <iostream>
#include "tree.h"
#include "graph.h"
#include "smt.h"

using namespace treeck;

int main()
{
    //auto file = "tests/models/xgb-calhouse-very-easy.json";
    //std::shared_ptr<AddTree> at = std::make_shared<AddTree>(AddTree::from_json_file(file));
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

    ReuseFeatIdMapper fmap(at, at, {0}, true);

    std::cout << at << std::endl;
    KPartiteGraph g0(at);
    KPartiteGraph g1(at, fmap);
    std::cout << g0 << std::endl;
    std::cout << g1 << std::endl;
    KPartiteGraphOptimize opt(g0, g1);

    auto box_filter = [](const DomainBox&) {
        return true;
    };

    while (opt.step(box_filter, 0.0))
    {
        std::cout << "================================ " << std::endl;
    }
}
