#include <iostream>
#include <memory>

#include "tree.h"
#include "searchspace.h"
#include "prune.h"

using namespace treeck;

int main()
{
    Tree<double> tree;
    tree.root().split(LtSplit(1, 0.5));
    tree.root().left().set_leaf_value(1.55);

    tree.root().right().skip_branch();

    std::cout << tree << std::endl;

    //auto file = "tests/models/xgb-calhouse-very-easy.json";
    //std::shared_ptr<AddTree> at = std::make_shared<AddTree>(AddTree::from_json_file(file));
    //SearchSpace sp(at);

    //std::cout << *at << std::endl;

    //size_t nleafs = 8;
    //sp.split(UnreachableNodesMeasure{}, NumDomTreeLeafsStopCond{nleafs});
    //Domains doms;

    //for (auto node_id : sp.leafs())
    //{
    //    sp.get_domains(node_id, doms);
    //    AddTree new_at = prune(*at, doms);

    //    std::cout << new_at << std::endl;

    //    //std::cout << "Domains for node_id " << node_id << ": " << std::endl;
    //    //for (size_t i = 0; i < doms.size(); ++i)
    //    //{
    //    //    if (doms[i].is_everything()) continue;
    //    //    std::cout << " - [" << i << "]: " << doms[i] << std::endl;
    //    //}
    //}
}
