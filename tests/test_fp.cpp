#include "basics.hpp"
#include "box.hpp"
#include "tree.hpp"
#include "fp.hpp"

using namespace veritas;

int test_fp() {
    FpMap map;

    map.add(0, 1.0);
    map.add(0, 8.0);
    map.add(0, 4.0);

    map.finalize();

    bool result = true
        && map(0, 0.0) == 0
        && map(0, 1.0) == 1
        && map(0, 1.5) == 1
        && map(0, 4.0) == 2
        && map(0, 6.0) == 2
        && map(0, 8.0) == 3
        && map(0, 9.0) == 3
        ;

    std::cout << "test_fp " << result << std::endl;
    return result;
}

int test_fp_with_tree1() {
    Tree t;
    t.split(t.root(), {1, 8.0});
    t.split(t["l"], {1, 2.0});
    t.split(t["lr"], {1, 4.0});
    t.leaf_value(t["ll"]) = 15.124;
    t.leaf_value(t["r"]) = 12.22;

    TreeFp tcheck;
    tcheck.split(tcheck.root(), {1, 2});
    tcheck.split(tcheck["l"], {1, 0});
    tcheck.split(tcheck["lr"], {1, 1});
    tcheck.leaf_value(tcheck["ll"]) = 15.124;
    tcheck.leaf_value(tcheck["r"]) = 12.22;

    FpMap map;
    map.add(t);
    map.finalize();

    //std::cout << t << std::endl;
    //std::cout << map.transform(t) << std::endl;

    bool result = true
        && tcheck == map.transform(t)
        ;

    std::cout << "test_fp_with_tree1 " << result << std::endl;
    return result;
}

int main_fp() {
    int result = 1
        && test_fp()
        && test_fp_with_tree1()
        ;
    return !result;
}
