#include "basics.hpp"
#include "box.hpp"

using namespace veritas;

int test_something() {

    bool result = true
        && 5 == 5
        ;

    std::cout << "test_something " << result << std::endl;
    return result;
}

int main_template() {
    int result = 1
        && test_something()
        ;
    return !result;
}
