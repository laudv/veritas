#include "addtree.hpp"
#include "basics.hpp"
#include "box.hpp"
#include "fp_search.hpp"
#include "json_io.hpp"
#include <fstream>
#include <stdexcept>

using namespace veritas;

AddTree get_simple_addtree1() {
    AddTree at(1);
    at.base_score(0) = 2;
    {
        Tree& t = at.add_tree();
        t.split(t[""], {1, 5.0});
        t.leaf_value(t["l"], 1) = 0.0;
        t.leaf_value(t["r"], 1) = 2.0;
    }
    {
        Tree& t = at.add_tree();
        t.split(t[""], {1, 3.0});
        t.leaf_value(t["l"], 1) = 10.0;
        t.leaf_value(t["r"], 1) = -10.0;
    }
    return at;
}

AddTree get_old_addtree(const char *x) {
    std::string fname("tests/models/xgb-img-");
    fname.append(x);
    fname.append(".json");
    std::ifstream f(fname);
    if (!f) { // from build/temp.linux... folder
        std::string fname2("../");
        fname2.append(fname);
        std::cout << "second try from " << fname2 << std::endl;
        f = std::ifstream(fname2);
    }
    if (!f) {
        throw std::runtime_error("cannot read xgb-img-hard.json");
    }
    AddTree at = addtree_from_oldjson(f);
    return at;
}

int test_simple1_1() {
    AddTree at = get_simple_addtree1();

    std::cout << at[0] << std::endl;
    std::cout << at[1] << std::endl;

    auto s = Search::max_output(at);
    StopReason r = StopReason::NONE;
    for (; r != StopReason::NO_MORE_OPEN; r = s->step()) {
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
    for (; r != StopReason::NO_MORE_OPEN; r = s->step()) {
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

    std::cout << at << std::endl;

    // no prune box, but we set a setting
    auto s = Search::max_output(at);
    s->settings.ignore_state_when_worse_than = 0.0;

    StopReason r = StopReason::NONE;
    for (; r != StopReason::NO_MORE_OPEN; r = s->step()) {
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

int test_old_at_easy() {
    std::cout << "\n\n===========================\n\n";
    AddTree at = get_old_addtree("easy");

    // neutralize leaf values here so that we can do equality tests on the
    // solution outputs
    at = at.neutralize_negative_leaf_values();

    // no prune box, but we set a setting
    auto s = Search::max_output(at);
    //s->settings.ignore_state_when_worse_than = 100.0;
    s->settings.stop_when_optimal = false;

    StopReason r = StopReason::NONE;
    for (; r != StopReason::NO_MORE_OPEN; r = s->steps(100)) {}

    std::cout << "final StopReason " << r
        << ", time " << s->time_since_start()
        << ", #ignored " << s->stats.num_states_ignored
        << ", #steps " << s->stats.num_steps
        << ", #sols " << s->num_solutions()
        << std::endl;

    int result = 1;

    std::vector<FloatT> example = {0.0, 0.0};
    for (size_t i = 0; i < s->num_solutions(); ++i) {
        auto sol = s->get_solution(i);
        for (auto&& [fid, ival] : sol.box)
            example[fid] = ival.lo;

        std::vector<FloatT> out { 0.0 };
        data<FloatT> outdata(out);
        at.eval(data<FloatT>(example), outdata);
        result = result && (outdata[0] == sol.output);
    }

    std::cout << "test_old_at_easy " << result << std::endl;
    return result;
}

int main_search() {
    int result = 1
        //&& test_simple1_1()
        //&& test_simple1_2()
        //&& test_simple1_3()
        && test_old_at_easy()
        ;
    return !result;
}
