#include "basics.hpp"
#include "box.hpp"
#include "addtree.hpp"
#include "fp_search.hpp"

using namespace veritas;

AddTree get_simple_addtree1() {
    AddTree at;
    {
        Tree& t = at.add_tree();
        t.split(t[""], {1, 5.0});
        t.leaf_value(t["l"]) = 2.0;
        t.leaf_value(t["r"]) = 4.0;
    }
    {
        Tree& t = at.add_tree();
        t.split(t[""], {1, 3.0});
        t.leaf_value(t["l"]) = 10.0;
        t.leaf_value(t["r"]) = -10.0;
    }
    return at;
}

int test_simple1_1() {
    AddTree at = get_simple_addtree1();

    std::cout << at[0] << std::endl;
    std::cout << at[1] << std::endl;

    auto s = Search::max_output(at);
    StopReason r = StopReason::NONE;
    for (; r != StopReason::NO_MORE_OPEN;
            r = s->step()) {
        std::cout << s->current_bounds() << std::endl;
        std::cout << ">> StopReason " << r << ", nsteps " << s->stats.num_steps << "\n";
        std::cout << std::endl;
    }
    std::cout << "final StopReason " << r
        << ", time " << s->time_since_start()
        << ", #ignored " << s->stats.num_states_ignored
        << std::endl;

    for (size_t i = 0; i < s->num_solutions(); ++i)
        std::cout << " - " << s->get_solution(i) << '\n';

    bool result = s->num_solutions() == 3
        && s->get_solution(0).output == 12.0
        && s->get_solution(1).output == -6
        && s->get_solution(2).output == -8
        && s->get_solution(0).box.at(0) == IntervalPair(1, Interval::from_hi(3.0))
        && s->get_solution(1).box.at(0) == IntervalPair(1, Interval::from_lo(5.0))
        && s->get_solution(2).box.at(0) == IntervalPair(1, {3, 5})
        && s->get_solution(0).box.size() == 1
        && s->get_solution(1).box.size() == 1
        && s->get_solution(2).box.size() == 1
        ;


    std::cout << "test_simple1_1 " << result << std::endl;
    return result;
}

int test_simple1_2() {
    AddTree at = get_simple_addtree1();

    // with prune_box
    auto s = Search::max_output(at, {{}, {-1.0, 5.0}});
    StopReason r = StopReason::NONE;
    for (; r != StopReason::NO_MORE_OPEN;
            r = s->step()) {
        std::cout << s->current_bounds() << std::endl;
        std::cout << ">> StopReason " << r << ", nsteps " << s->stats.num_steps << "\n";
        std::cout << std::endl;
    }
    std::cout << "final StopReason " << r
        << ", time " << s->time_since_start()
        << ", #ignored " << s->stats.num_states_ignored
        << std::endl;

    for (size_t i = 0; i < s->num_solutions(); ++i)
        std::cout << " - " << s->get_solution(i) << '\n';

    bool result = s->num_solutions() == 2
        && s->get_solution(0).output == 12.0
        && s->get_solution(1).output == -8
        && s->get_solution(0).box.at(0) == IntervalPair(1, {-1, 3})
        && s->get_solution(1).box.at(0) == IntervalPair(1, {3, 5})
        && s->get_solution(0).box.size() == 1
        && s->get_solution(1).box.size() == 1
        ;


    std::cout << "test_simple1_2 " << result << std::endl;
    return result;
}

int test_simple1_3() {
    AddTree at = get_simple_addtree1();

    // no prune box, but we set a setting
    auto s = Search::max_output(at);
    s->settings.ignore_state_when_worse_than = 0.0;

    StopReason r = StopReason::NONE;
    for (; r != StopReason::NO_MORE_OPEN;
            r = s->step()) {
        std::cout << s->current_bounds() << std::endl;
        std::cout << ">> StopReason " << r << ", nsteps " << s->stats.num_steps << "\n";
        std::cout << std::endl;
    }
    std::cout << "final StopReason " << r
        << ", time " << s->time_since_start()
        << ", #ignored " << s->stats.num_states_ignored
        << std::endl;

    for (size_t i = 0; i < s->num_solutions(); ++i)
        std::cout << " - " << s->get_solution(i) << '\n';

    bool result = s->num_solutions() == 1
        && s->get_solution(0).output == 12.0
        && s->get_solution(0).box.at(0) == IntervalPair(1, Interval::from_hi(3.0))
        && s->get_solution(0).box.size() == 1
        ;


    std::cout << "test_simple1_3 " << result << std::endl;
    return result;
}
int main_search() {
    int result = 1
        && test_simple1_1()
        && test_simple1_2()
        && test_simple1_3()
        ;
    return !result;
}
