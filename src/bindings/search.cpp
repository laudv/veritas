#include "bindings.h"
#include "fp_search.hpp"

namespace py = pybind11;
using namespace veritas;

void init_search(py::module &m) {
    py::enum_<StopReason>(m, "StopReason")
        .value("NONE", StopReason::NONE)
        .value("NO_MORE_OPEN", StopReason::NO_MORE_OPEN)
        .value("NUM_SOLUTIONS_EXCEEDED", StopReason::NUM_SOLUTIONS_EXCEEDED)
        .value("NUM_NEW_SOLUTIONS_EXCEEDED", StopReason::NUM_NEW_SOLUTIONS_EXCEEDED)
        .value("OPTIMAL", StopReason::OPTIMAL)
        .value("ATLEAST_BOUND_BETTER_THAN", StopReason::ATLEAST_BOUND_BETTER_THAN)
        .value("OUT_OF_TIME", StopReason::OUT_OF_TIME)
        .value("OUT_OF_MEMORY", StopReason::OUT_OF_MEMORY)
        ; // StopReason

    py::enum_<HeuristicType>(m, "HeuristicType")
        .value("MAX_OUTPUT",          HeuristicType::MAX_OUTPUT)
        .value("MIN_OUTPUT",          HeuristicType::MIN_OUTPUT)
        .value("MAX_COUNTING_OUTPUT", HeuristicType::MAX_COUNTING_OUTPUT)
        .value("MIN_COUNTING_OUTPUT", HeuristicType::MIN_COUNTING_OUTPUT)

        .value("MULTI_MAX_MAX_OUTPUT_DIFF", HeuristicType::MULTI_MAX_MAX_OUTPUT_DIFF)
        .value("MULTI_MAX_MIN_OUTPUT_DIFF", HeuristicType::MULTI_MAX_MIN_OUTPUT_DIFF)
        .value("MULTI_MIN_MAX_OUTPUT_DIFF", HeuristicType::MULTI_MIN_MAX_OUTPUT_DIFF)
        ; // HeuristicType
    
    py::class_<Bounds>(m, "Bounds", R"pbdoc(
            BoundsClass
        )pbdoc")
        .def_readonly("atleast", &Bounds::atleast)
        .def_readonly("best", &Bounds::best)
        .def_readonly("top_of_open", &Bounds::top_of_open)
        ; // Bounds
    
    py::class_<Statistics>(m, "Statistics")
        .def_readonly("num_steps", &Statistics::num_steps)
        .def_readonly("num_states_ignored", &Statistics::num_states_ignored)
        .def_readonly("num_update_scores_fails", &Statistics::num_update_scores_fails)
        ; // Statistics

    py::class_<Config>(m, "Config")
        .def(py::init<HeuristicType>())
        .def_readwrite("focal_eps", &Config::focal_eps)
        .def_readwrite("max_focal_size", &Config::max_focal_size)
        .def_readwrite("stop_when_num_solutions_exceeds",
                       &Config::stop_when_num_solutions_exceeds)
        .def_readwrite("stop_when_num_new_solutions_exceeds",
                       &Config::stop_when_num_new_solutions_exceeds)
        .def_readwrite("stop_when_optimal", &Config::stop_when_optimal)
        .def_readwrite("ignore_state_when_worse_than",
                       &Config::ignore_state_when_worse_than)
        .def_readwrite("stop_when_atleast_bound_better_than",
                       &Config::stop_when_atleast_bound_better_than)
        .def_readwrite("multi_ignore_state_when_class0_worse_than",
                       &Config::multi_ignore_state_when_class0_worse_than)
        .def("get_search", [](const Config &conf, const AddTree &at,
                              const py::object &pybox) {
            auto buf = tobox(pybox);
            auto fbox = BoxRef(buf).to_flatbox();
            return conf.get_search(at, fbox);
        }, py::arg("at"), py::arg("prune_box") = py::list())
        ; // Config

    py::class_<Search, std::shared_ptr<Search>>(m, "Search")
        .def("step", &Search::step)
        .def("steps", &Search::steps)
        .def("step_for", &Search::step_for)
        .def("num_solutions", &Search::num_solutions)
        .def("num_open", &Search::num_open)
        .def("get_max_memory", &Search::get_max_memory)
        .def("set_max_memory", &Search::set_max_memory)
        .def("get_used_memory", &Search::get_used_memory)
        .def("time_since_start", &Search::time_since_start)
        .def("current_bounds", &Search::current_bounds)
        .def("get_solution", &Search::get_solution)
        .def("get_solution_nodes", &Search::get_solution_nodes)
        .def("is_optimal", &Search::is_optimal)
//        .def("get_at_output_for_box", [](const VSearch& s, const py::list& pybox) {
//            Box box = tobox(pybox);
//            BoxRef b(box);
//            return s.get_at_output_for_box(b);
//        })
        .def_readonly("stats", &Search::stats)
        .def_readonly("config", &Search::config)
        ; // Search

    py::class_<Solution>(m, "Solution")
        //.def_readonly("eps", &Solution::eps)
        .def_readonly("time", &Solution::time)
        .def_readonly("output", &Solution::output)
        .def("box", [](const Solution& s) {
            py::dict d;
            for (auto&& [feat_id, ival] : s.box)
                d[py::int_(feat_id)] = ival;
            return d;
        })
        .def("__str__", [](const Solution& s) { return tostr(s); })
        ; // Solution
}
