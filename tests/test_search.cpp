#include "addtree.hpp"
#include "basics.hpp"
#include "box.hpp"
#include "fp_search.hpp"
#include "interval.hpp"
#include "json_io.hpp"
#include <fstream>
#include <stdexcept>
#include <unordered_set>

using namespace veritas;

static AddTree get_simple_addtree1() {
    AddTree at(1);
    at.base_score(0) = 2;
    {
        Tree& t = at.add_tree();
        t.split(t[""], {1, 5.0});
        t.leaf_value(t["l"], 0) = 0.0;
        t.leaf_value(t["r"], 0) = 2.0;
    }
    {
        Tree& t = at.add_tree();
        t.split(t[""], {1, 3.0});
        t.leaf_value(t["l"], 0) = 10.0;
        t.leaf_value(t["r"], 0) = -10.0;
    }
    return at;
}

static AddTree read_old_addtree(const char *x) {
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
        throw std::runtime_error(fname);
    }
    AddTree at = addtree_from_oldjson(f);
    return at;
}

AddTree read_addtree(const char *x) {
    std::string fname("tests/models/");
    fname.append(x);
    std::ifstream f(fname);
    if (!f) { // from build/temp.linux... folder
        std::string fname2("../");
        fname2.append(fname);
        std::cout << "second try from " << fname2 << std::endl;
        f = std::ifstream(fname2);
    }
    if (!f) {
        throw std::runtime_error(fname);
    }
    AddTree at = addtree_from_json<AddTree>(f);
    return at;
}

int test_simple1_1() {
    AddTree at = get_simple_addtree1();

    std::cout << at[0] << std::endl;
    std::cout << at[1] << std::endl;

    Config config(HeuristicType::MAX_OUTPUT);
    FlatBox prune_box;
    auto s = config.get_search(at, prune_box);
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
    Config config(HeuristicType::MAX_OUTPUT);
    FlatBox prune_box {{}, {-1.0, 5.0}};
    auto s = config.get_search(at, prune_box);
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

    // no prune box, but we set a config value
    Config config(HeuristicType::MAX_OUTPUT);
    config.ignore_state_when_worse_than = 0.0;
    FlatBox prune_box;
    auto s = config.get_search(at, prune_box);

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
    AddTree at = read_old_addtree("easy");

    // neutralize leaf values here so that we can do equality tests on the
    // solution outputs
    at = at.neutralize_negative_leaf_values();

    // no prune box, but we set a config value
    Config config(HeuristicType::MAX_OUTPUT);
    config.stop_when_optimal = false;
    auto s = config.get_search(at, {});

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

int test_simple_counting() {
    AddTree at = get_simple_addtree1();

    std::cout << at[0] << std::endl;
    std::cout << at[1] << std::endl;

    //auto s = Search::max_output(at);
    Config config = Config(HeuristicType::MAX_COUNTING_OUTPUT);
    config.stop_when_optimal = false;
    auto s = config.get_search(at, {});

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


    // TODO test this somehow
    auto s2 = config.reuse_heuristic(*s, {});
    StopReason r2 = StopReason::NONE;
    for (; r2 != StopReason::NO_MORE_OPEN; r2 = s2->step()) {
        std::cout << s2->current_bounds() << std::endl;
        std::cout << ">> StopReason " << r2 << ", nsteps " << s2->stats.num_steps << "\n";
        std::cout << std::endl;
    }

    int result = 1;


    std::cout << "test_simple_counting " << result << std::endl;
    return result;
}

int do_test_coverage(const AddTree& at, HeuristicType h) {
    Config config(h);
    config.stop_when_optimal = false;
    config.focal_eps = 0.5;
    auto s = config.get_search(at, {});

    StopReason r = StopReason::NONE;
    for (; r != StopReason::NO_MORE_OPEN; r = s->steps(100)) {}

    std::cout << "  final StopReason " << r
        << ", time " << s->time_since_start()
        << ", #ignored " << s->stats.num_states_ignored
        << ", #steps " << s->stats.num_steps
        << ", #sols " << s->num_solutions()
        << std::endl;

    int result = 1;
    size_t sol_index = 0;

    std::vector<int> raw_coverage(100*100);
    data<int> coverage(raw_coverage.data(), 100, 100, 100, 1);

    for (; sol_index < s->num_solutions(); ++sol_index) {
        Solution sol = s->get_solution(sol_index);
        FlatBox box { Interval(0, 100), Interval(0, 100) };
        for (const auto& ip : sol.box)
            box[ip.feat_id] = box[ip.feat_id].intersect(ip.interval);

        //std::cout << sol_index << " " << sol.output << ':';
        //for (const auto& i : box)
        //    std::cout << " " << i;
        //std::cout << "|";
        //for (const auto& k : s->get_solution_nodes(sol_index))
        //    std::cout << " " << k;
        //std::cout << "\n";

        int ilo = static_cast<int>(box[0].lo);
        int ihi = static_cast<int>(box[0].hi);
        int jlo = static_cast<int>(box[1].lo);
        int jhi = static_cast<int>(box[1].hi);

        for (int i = ilo; i < ihi; ++i) {
            for (int j = jlo; j < jhi; ++j) {
                if (coverage.get_elem(i, j) != 0) {
                    std::cout << "  VIOLATION\n";
                    goto done;
                }
                coverage.get_elem(i, j) += 1;
            }
        }
    }

done:

    //std::cout << "---------------------------------------------------"
    //          << "-------------------------------------------------\n";
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            //if (coverage.get_elem(i, j) == 0)
            //    std::cout << '.';
            //else
            //    std::cout << coverage.get_elem(i, j);

            // Everything should be 1
            result &= coverage.get_elem(i, j) == 1;
        }
        //std::cout << "\n";
    }
    //std::cout << "---------------------------------------------------"
    //          << "-------------------------------------------------\n\n";

    std::cout << "  result: " << result << std::endl;
    return result;
}

int test_coverage() {
    std::cout << "\n\n===========================\n\n";
    int result = 1;

    for (const char *datapath : {"xgb-img-multiclass.json", "rf-img-multiclass.json"}) {
        AddTree at_mult = read_addtree(datapath);

        for (int c = 0; c < 4; ++c) {
            std::cout << "\n\n=== MAX_OUTPUT ============ " << datapath
                      << " class " << c << "\n";
            AddTree at = at_mult.make_singleclass(c);
            result &= do_test_coverage(at, HeuristicType::MAX_OUTPUT);
        }

        std::cout << "\n=== MULTI_MAX_MAX ========= " << datapath << "\n";
        result &= do_test_coverage(at_mult, HeuristicType::MULTI_MAX_MAX_OUTPUT_DIFF);
    }

    std::cout << "test_coverage " << result << std::endl;
    return result;
}


int test_multiclass() {
    std::cout << "\n\n===========================\n\n";
    AddTree at = read_addtree("rf-img-multiclass.json");
    at.swap_class(3);

    Config c(HeuristicType::MULTI_MAX_MAX_OUTPUT_DIFF);
    c.stop_when_optimal = false;
    c.ignore_state_when_worse_than = 0.0;

    FlatBox prune_box { Interval(10, 30), Interval(10, 30) };
    auto s = c.get_search(at, prune_box);

    int result = 1;

    StopReason r = StopReason::NONE;
    for (; r != StopReason::NO_MORE_OPEN; r = s->steps(100)) {}

    std::cout << "final StopReason " << r
        << ", time " << s->time_since_start()
        << ", #ignored " << s->stats.num_states_ignored
        << ", #steps " << s->stats.num_steps
        << ", #sols " << s->num_solutions()
        << std::endl;


    std::vector<FloatT> rout(4);
    data<FloatT> out(rout);

    for (size_t i = 0; i < s->num_solutions(); ++i) {
        Solution sol = s->get_solution(i);

        std::vector<FloatT> rdata { sol.box[0].interval.lo, sol.box[1].interval.lo };
        data<FloatT> d{rdata};
        at.eval(d, out);

        FloatT expected = out[0] - std::max({out[1], out[2], out[3]});

        std::cout << "sol out=" << sol.output
                  << ", exp=" << expected
                  << ", diff=" << (sol.output-expected) << "\n";

        result &= std::abs(sol.output-expected) < 1e-10;
    }

    std::cout << "test_multiclass " << result << std::endl;
    return result;
}

int test_heuristic_consistency() {
    std::cout << "\n\n===========================\n\n";
    AddTree at = read_addtree("xgb-img-multiclass.json");
    at.swap_class(2);

    Config c(HeuristicType::MULTI_MAX_MAX_OUTPUT_DIFF);
    c.stop_when_optimal = false;
    //c.ignore_state_when_worse_than = 0.0;
    c.stop_when_num_solutions_exceeds = 1;
    c.focal_eps = 1.0;

    auto s = c.get_search(at, {});

    int result = 1;

    FloatT prev_top_of_open = s->current_bounds().top_of_open;
    StopReason r = StopReason::NONE;
    for (; r != StopReason::NO_MORE_OPEN; r = s->step()) {
        Bounds bounds = s->current_bounds();
        bool violation = (prev_top_of_open - bounds.top_of_open) < -1e-14;
        if (violation) {
            std::cout << "VIOLATION " << prev_top_of_open
                << " < " << bounds.top_of_open
                << " -> " << (prev_top_of_open-bounds.top_of_open)
                << "\n";
        }

        result &= !violation;
        prev_top_of_open = bounds.top_of_open;
    }

    std::cout << "test_heuristic_consistency " << result << std::endl;
    return result;
}

int main_search() {
    int result = 1
        && test_simple1_1()
        && test_simple1_2()
        && test_simple1_3()
        && test_old_at_easy()
        && test_simple_counting()
        && test_coverage()
        && test_multiclass()
        && test_heuristic_consistency()
        ;
    return !result;
}
