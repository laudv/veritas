#include <iostream>
#include <memory>

#include "tree.h"
#include "searchspace.h"

using namespace treeck;

int main()
{
    auto file = "tests/models/xgb-calhouse-easy.json";
    std::shared_ptr<AddTree> at = std::make_shared<AddTree>(AddTree::from_json_file(file));
    SearchSpace sp(at);

    size_t nleafs = 8;
    sp.split(UnreachableNodesMeasure{}, NumDomTreeLeafsStopCond{nleafs});
    Domains doms;

    for (auto node_id : sp.leafs())
    {
        sp.get_domains(node_id, doms);

        //std::cout << "Domains for node_id " << node_id << ": " << std::endl;
        //for (size_t i = 0; i < doms.size(); ++i)
        //{
        //    if (doms[i].is_everything()) continue;
        //    std::cout << " - [" << i << "]: " << doms[i] << std::endl;
        //}
    }
}
