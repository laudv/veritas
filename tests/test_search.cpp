#include "basics.hpp"
#include "box.hpp"
#include "addtree.hpp"
#include "fp_search.hpp"

using namespace veritas;

int test_search1() {
    AddTree at;

    Search::max_output(at);

    bool result = true;
    return result;
}

int main_search() {
    int result = 1
        && test_search1()
        ;
    return !result;
}
