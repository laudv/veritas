#include "basics.hpp"
#include "box.hpp"
#include "addtree.hpp"
#include "json_io.hpp"
#include "fp_search.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>

using namespace veritas;

/* test_search.cpp */
AddTree read_addtree(const char *x);

int test_memory1() {
    AddTree at = read_addtree("rf-img-multiclass.json");

    Config config(HeuristicType::MULTI_MAX_MAX_OUTPUT_DIFF);
    config.max_memory = 128 * 1024;
    config.memory_min_block_size = 16 * 1024;
    config.stop_when_optimal = false;
    auto search = config.get_search(at, {});

    bool result = false;

    StopReason done = StopReason::NONE;
    while (done != StopReason::NO_MORE_OPEN) {
        try {
            // This should return OUT_OF_MEMORY once, and throw an exception on
            // the next run
            done = search->steps(2000);
        } catch (const std::runtime_error& err) {
            result = strcmp(err.what(), "Cannot continue, previous StopReason "
                                        "was OUT_OF_MEMORY.") == 0;
            std::cout << "Caught OOM\n";
            break;
        }
        std::cout << "StopReason " << done << " memory "
            << search->get_used_memory() / 1024
            << "/" 
            << config.max_memory / 1024
            << std::endl;
    }

    // 1. last valid StopReason should be OUT_OF_MEMORY
    // 2. Veritas should have used all the available memory
    result = result
        && done == StopReason::OUT_OF_MEMORY
        && (search->get_used_memory() / 1024) == (config.max_memory / 1024)
        ;

    std::cout << "test_memory1 " << result << std::endl;
    return result;
}

int main_memory() {
    int result = 1
        && test_memory1()
        ;
    return !result;
}
