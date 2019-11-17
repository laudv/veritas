#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

#include "tree.h"
#include "addtree.h"
#include "searchspace.h"

using namespace treeck;

int main()
{
    std::ifstream t("tests/models/xgb-calhouse-easy.json");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string s(buffer.str());

    std::shared_ptr<AddTree> at = std::make_shared<AddTree>(AddTree::from_json(s));
    SearchSpace sp(at);

    size_t nleafs = 3;
    sp.split(UnreachableNodesMeasure{}, NumDomTreeLeafsStopCond{nleafs});
}
