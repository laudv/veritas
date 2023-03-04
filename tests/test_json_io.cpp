#include "tree.hpp"
#include "json_io.hpp"
#include <ostream>
#include <fstream>

using namespace veritas;

int test_json1() {
    std::stringstream s;

    Tree t;
    t.split(t.root(), {1, 12.3});
    t.leaf_value(t["l"]) = 4.0;
    t.split(t["r"], {2, 1351.0});
    t.leaf_value(t["rl"]) = -2.0;
    t.leaf_value(t["rr"]) = 8.0;

    tree_to_json(s, t);
    auto t2 = tree_from_json<Tree>(s);

    //std::cout << t << std::endl;
    //std::cout << t2 << std::endl;
  
    bool result = t == t2;

    std::cout << "test_json1 " << result << std::endl;
    return result;
}

int test_json2() {
    std::stringstream s;

    TreeFp t;
    t.split(t.root(), {1, 12});
    t.leaf_value(t["l"]) = 4.0;
    t.split(t["r"], {2, 1351});
    t.leaf_value(t["rl"]) = -2.0;
    t.leaf_value(t["rr"]) = 8.0;

    tree_to_json(s, t);
    auto t2 = tree_from_json<TreeFp>(s);
  
    bool result = t == t2;

    std::cout << "test_json2 " << result << std::endl;
    return result;
}

int test_json3() {
    std::stringstream s;

    AddTreeFp at;
    at.base_score = 9.99;
    {
        TreeFp& t = at.add_tree();
        t.split(t.root(), {1, 12});
        t.leaf_value(t["l"]) = 4.0;
        t.split(t["r"], {2, 1351});
        t.leaf_value(t["rl"]) = -2.0;
        t.leaf_value(t["rr"]) = 8.0;
    }
    {
        TreeFp& t = at.add_tree();
        t.leaf_value(t.root()) = 10.0;
    }

    addtree_to_json(s, at);
    std::cout << s.str();
    std::flush(std::cout);

    AddTreeFp at2 = addtree_from_json<AddTreeFp>(s);
  
    bool result = at == at2;

    std::cout << "test_json3 " << result << std::endl;
    return result;
}

int test_json4() {
    std::stringstream s;
    GTree<LtSplit, std::string> t;
    t.leaf_value(t.root()) = "hello world!";

    std::cout << t << std::endl;

    tree_to_json(s, t);
    auto t2 = tree_from_json<decltype(t)>(s);

    std::cout << t2 << std::endl;
  
    bool result = t == t2;

    std::cout << "test_json4 " << result << std::endl;
    return result;
}

int test_oldjson() {
    std::ifstream f("./tests/models/xgb-img-hard.json");
    if (!f) // from build/temp.linux... folder
        f = std::ifstream("../../tests/models/xgb-img-hard.json");
    if (!f) {
        std::cout << "cannot read xgb-img-hard.json\n";
        return false;
    }
    AddTree at = addtree_from_oldjson(f);

    //std::cout << at[0] << std::endl;
    std::cout << at << std::endl;

    int result = 1
        && at.size() == 50
        && at[0].get_split(at[0].root()) == LtSplit(0, 63)
        && at[0].leaf_value(at[0]["lllll"]) == 37.7239
        && at[0].leaf_value(at[0]["llllr"]) == 44.5408
        && at[49].leaf_value(at[49]["llllll"]) == -0.0201895
        && at[49].leaf_value(at[49]["lllllr"]) == 0.0045661
        ;
    std::cout << "test_oldjson " << result << std::endl;
    return result;
}

int main_json_io() {
    int result = 1
        && test_json1()
        && test_json2()
        && test_json3()
        && test_json4()
        && test_oldjson()
        ;
    return !result;
}
