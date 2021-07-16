#include "features.hpp"
//#include "node_search.hpp"
#include "new_graph.hpp"
#include "graph_search.hpp"

#include <iostream>
#include <fstream>
#include <assert.h>

using namespace veritas;

/*
void test_very_simple()
{
    AddTree at;
    AddTree::TreeT t;
    t.root().split(LtSplit(0, 2.0));
    t.root().left().set_leaf_value(-1.0);
    t.root().right().set_leaf_value(1.0);
    at.add_tree(std::move(t));

    FeatInfo finfo(at, at, {}, true);

    std::cout << at << std::endl;
    KPartiteGraph g0(at, finfo, 0);
    KPartiteGraph g1(at, finfo, 1);

    KPartiteGraphOptimize opt0(g0, g1, KPartiteGraphOptimize::Heuristic::DYN_PROG);
    KPartiteGraphOptimize opt1(g0, g1);
    std::cout << g0 << std::endl;
    std::cout << g1 << std::endl;

    auto f = [](DomainStore&) {
        return true;
    };

    while (opt0.step(f, 0.0001));
    std::cout << "DYN_PROG:" << std::endl;
    for (auto& s : opt0.solutions)
        std::cout << s.output0 << ", " << s.output1 << " (" << s.output1 - s.output0 << ')' << "  " << s.box << std::endl;
    std::cout << "num_rejected " << opt0.num_rejected << std::endl;
    std::cout << "num_box_checks " << opt0.num_box_checks << std::endl;
    std::cout << "num_steps " << std::get<0>(opt0.num_steps) << ", " << std::get<1>(opt0.num_steps) << std::endl << std::endl;

    while (opt1.step(f, 0.0001));
    std::cout << "RECOMPUTE:" << std::endl;
    for (auto& s : opt1.solutions)
        std::cout << s.output0 << ", " << s.output1 << " (" << s.output1 - s.output0 << ')' << "  " << s.box << std::endl;
    std::cout << "num_rejected " << opt1.num_rejected << std::endl;
    std::cout << "num_box_checks " << opt1.num_box_checks << std::endl;
    std::cout << "num_steps " << std::get<0>(opt1.num_steps) << ", " << std::get<1>(opt1.num_steps) << std::endl;
}

void test_simple()
{
    AddTree at;
    AddTree::TreeT t;
    t.root().split(LtSplit(0, 1.24));
    t.root().left().set_leaf_value(-1.0);
    t.root().right().set_leaf_value(1.0);
    at.add_tree(std::move(t));
    t = AddTree::TreeT();
    t.root().split(LtSplit(0, 1.52));
    t.root().left().split(BoolSplit(1));
    t.root().left().left().set_leaf_value(4.0);
    t.root().left().right().set_leaf_value(-4.0);
    t.root().right().set_leaf_value(2.0);
    at.add_tree(std::move(t));

    FeatInfo finfo(at, at, {1}, true);

    std::cout << at << std::endl;
    KPartiteGraph g0(at, finfo, 0);
    KPartiteGraph g1(at, finfo, 1);
    KPartiteGraphOptimize opt(g0, g1);
    std::cout << g0 << std::endl;
    std::cout << g1 << std::endl;

    auto f = [](DomainStore&) {
        return true;
    };

    while (opt.step(f, 0.0));

    for (auto& s : opt.solutions)
    {
        std::cout << s.output0 << ", " << s.output1 << " (" << s.output1 - s.output0 << ')'
            << "  " << s.box << std::endl;
    }
    std::cout << "num_rejected " << opt.num_rejected << std::endl;
    std::cout << "num_steps " << std::get<0>(opt.num_steps) << ", " << std::get<1>(opt.num_steps) << std::endl;
}

void test_img()
{
    auto file = "tests/models/xgb-img-very-easy.json";
    AddTree at = AddTree::from_json_file(file);
    AddTree dummy;

    //std::cout << at << std::endl;

    FeatInfo finfo(at, dummy, {}, true); // minimize
    KPartiteGraph g0(at, finfo, 0);
    KPartiteGraph dummyg;
    KPartiteGraphOptimize opt(g0, dummyg);

    std::cout << g0 << std::endl;

    auto f = [](DomainStore&) {
        return true;
    };

    while (opt.step(f, 50.4, 0.0))
    {
        std::cout << "================================ " << std::endl;
    }

    for (auto& s : opt.solutions)
        std::cout << s.output0 << " " << s.box << std::endl;
}

void test_unconstrained_bounds(const char *model)
{
    AddTree at = AddTree::from_json_file(model);
    AddTree dummy;

    FeatInfo finfo(at, dummy, {}, true);
    KPartiteGraph g0;
    KPartiteGraph g1(at, finfo, 0);
    KPartiteGraphOptimize opt(g0, g1); // maximize, g0 is dummy
    opt.store().set_max_mem_size(1024*1024*50);
    //opt.set_eps(0.02, 0.02);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> timings;
    std::vector<double> num_steps;
    std::cout << "bound, mem, num_steps" << std::endl;
    while (opt.num_candidate_cliques() < 1000000)
    {
        try {
            if (!opt.steps(1000))
                break;
        } catch(...) {
            std::cout << "out of memory" << std::endl;
            break;
        }

        std::cout << std::get<1>(opt.current_bounds())
            << ", " << (opt.store().get_mem_size() / (1024*1024))
            << ", " << std::get<1>(opt.num_steps)
            << std::endl;

        while (timings.size() < opt.solutions.size())
        {
            double d = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start).count();
            timings.push_back(d / 1000000.0);
            num_steps.push_back(std::get<0>(opt.num_steps) + std::get<1>(opt.num_steps));
        }
    }

    for (size_t i = 0; i < opt.solutions.size(); ++i)
    {
        auto& sol = opt.solutions[i];
        std::cout << sol.output1 << ", " << sol.eps
            << ", " << num_steps[i]
            << ", " << sol.time
            << std::endl;
    }
}

void test_parallel(const char *model)
{
    AddTree at = AddTree::from_json_file(model);
    AddTree dummy;

    FeatInfo finfo(at, dummy, {}, true);
    KPartiteGraph g0;
    KPartiteGraph g1(at, finfo, 0);
    KPartiteGraphOptimize opt(g0, g1); // maximize, g0 is dummy
    opt.store().set_max_mem_size(1024*1024*50);
    opt.set_eps(0.1);

    //std::cout << "steps(100)" << std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    opt.steps(10);
    //auto stop = std::chrono::high_resolution_clock::now();
    //double dur = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
    //std::cout << "steps(100) done: " << opt.num_candidate_cliques()
    //    << " cliques in " << (dur * 1e-6)
    //    << " bound=" << std::get<1>(opt.current_bounds())
    //    << " num_steps=" << std::get<1>(opt.num_steps)
    //    << std::endl;

    KPartiteGraphParOpt paropt(6, opt);
    //paropt.set_box_filter([]() { return [](const DomainBox&) { return true; }; });
    while (paropt.get_eps() != 1.0 || paropt.num_new_valid_solutions() == 0)
    {
        if (paropt.num_new_valid_solutions() > 0) {
            FloatT new_eps = std::min(1.0, paropt.get_eps() + 0.05);
            std::cout << "increasing eps " << paropt.get_eps() << "->" << new_eps << std::endl;
            paropt.set_eps(new_eps);
        }
        paropt.steps_for(50);
    }
    std::cout << "joining..." << std::endl;
    //std::this_thread::sleep_for(std::chrono::seconds(1));
    paropt.join_all();
    std::cout << "joined!" << std::endl;

    for (size_t i = 0; i < paropt.num_threads(); ++i)
    {
        for (auto& sol : paropt.worker_opt(i).solutions)
        {
            std::cout << "worker " << i << " " << sol.output1
                << ", " << sol.eps
                << ", " << sol.time
                << ", " << sol.is_valid
                << std::endl;
        }
    }
}

void test_box_checker1()
{
    BoxChecker checker{3};
    std::vector<DomainPair> box{ {0, {2, INFINITY}}, {1, {-5.0, 5.0}}, {2, {-100, 50}} };

    int sumid = checker.add_sum(0, 1);
    std::cout << "sumid " << sumid << std::endl;
    int pow2id = checker.add_pow2(sumid);
    int sqrtid = checker.add_sqrt(pow2id);
    int constid = checker.add_const(10.0);
    int subid = checker.add_sub(sqrtid, constid);
    checker.add_eq(subid, 2);

    checker.copy_from_workspace(box);
    auto st = checker.update();
    std::cout << "update " << st << std::endl;
    st = checker.update();
    std::cout << "update " << st << std::endl;
    st = checker.update();
    std::cout << "update " << st << std::endl;
    st = checker.update();
    std::cout << "update " << st << std::endl;
    checker.copy_to_workspace(box);

    std::cout << std::endl << "----" << std::endl;
    for (auto p : box)
    {
        std::cout << "box: " << p.first << ", " << p.second << std::endl;
    }

    std::cout << "expr dom sum: " << checker.get_expr_dom(sumid) << std::endl;
    std::cout << "expr dom pow2: " << checker.get_expr_dom(pow2id) << std::endl;
    std::cout << "expr dom sqrt: " << checker.get_expr_dom(sqrtid) << std::endl;
    std::cout << "expr dom const: " << checker.get_expr_dom(constid) << std::endl;
    std::cout << "expr dom sub: " << checker.get_expr_dom(subid) << std::endl;
}

void test_box_checker2()
{
    BoxChecker checker{3};
    std::vector<DomainPair> box{{1, TRUE_DOMAIN}, {2, TRUE_DOMAIN}};

    checker.add_k_out_of_n({0, 1, 2}, 2, true);

    checker.copy_from_workspace(box);
    auto st = checker.update();
    std::cout << "update " << st << std::endl;
    st = checker.update();
    std::cout << "update " << st << std::endl;
    checker.copy_to_workspace(box);

    std::cout << std::endl << "----" << std::endl;
    for (auto p : box)
    {
        std::cout << "box: " << p.first << ", " << p.second << std::endl;
    }
}
*/

void test_tree1()
{
    Tree tree;
    auto n = tree.root();
    n.split({1, 12.3});
    n.left().set_leaf_value(4);
    n.right().set_leaf_value(9.4);

    assert(n.is_root());
    assert(!n.left().is_root());
    assert(n.left().is_leaf());
    assert(n.left().leaf_value() == 4);
    assert(n.num_leafs() == 2);
    assert(n.tree_size() == 3);
}

void test_tree2()
{
    AddTree at;
    Tree& t = at.add_tree();
    t.root().split({1, 2.0});
    t.root().left().split({2, 4.0});
    t.root().left().right().split({2, 8.0});
    auto splits = at.get_splits();

    assert(splits[1][0] == 2.0);
    assert(splits[2][0] == 4.0);
    assert(splits[2][1] == 8.0);
    assert(splits[1].size() == 1);
    assert(splits[2].size() == 2);
}

void test_tree3()
{
    AddTree at;
    Tree& t = at.add_tree();
    t.root().split({1, 8.0});
    t.root().left().split({1, 2.0});
    t.root().left().right().split({1, 4.0});

    {
        auto n = at[0].root().left().right().left();
        auto doms = n.compute_box();
        assert(doms[0].feat_id == 1);
        std::cout << at[0] << std::endl << doms[0].domain << std::endl;
        assert(doms[0].domain == Domain::exclusive(2.0, 4.0));
    }
    {
        auto n = at[0].root().left().right().right();
        auto doms = n.compute_box();
        assert(doms[0].feat_id == 1);
        assert(doms[0].domain == Domain::exclusive(4.0, 8.0));
    }
    {
        auto n = at[0].root().left();
        auto doms = n.compute_box();
        assert(doms[0].feat_id == 1);
        assert(doms[0].domain == Domain::from_hi_exclusive(8.0));
    }
}

void test_json1() {
  std::stringstream s;

  Tree tree;
  auto n = tree.root();
  n.split({1, 12.3});
  n.left().set_leaf_value(4);
  n.right().split({2, 1351});
  n.right().left().set_leaf_value(9.4);
  n.right().right().set_leaf_value(9.5);

  tree.to_json(s);

  Tree tree2;
  tree2.from_json(s);

  assert(tree == tree2);
}

void test_json2() {
  std::stringstream s;

  AddTree at;
  at.base_score = 124.2;
  {
      Tree& tree = at.add_tree();
      auto n = tree.root();
      n.split({1, 12.3});
      n.left().set_leaf_value(4);
      n.right().split({2, 1351});
      n.right().left().set_leaf_value(9.4);
      n.right().right().set_leaf_value(9.5);
  }
  {
      Tree& tree = at.add_tree();
      tree.root().set_leaf_value(14.129);
  }
  {
      Tree& tree = at.add_tree();
      tree.root().split({1, 23.4});
      tree.root().left().set_leaf_value(-124.2);
      tree.root().right().set_leaf_value(-8.2);
  }

  at.to_json(s);
  AddTree at2;
  //std::cout << s.str() << std::endl;
  at2.from_json(s);

  assert(at == at2);
}

void test_eval1()
{
    AddTree at;
    Tree& t = at.add_tree();;
    t.root().split({0, 1.5});
    t.root().left().split({1, 1.5});
    t.root().left().left().split({2, 1.5});
    t.root().left().left().left().set_leaf_value(1.0);
    t.root().left().left().right().set_leaf_value(2.0);
    t.root().left().right().set_leaf_value(3.0);
    t.root().right().set_leaf_value(4.0);

    std::vector<FloatT> data = {1, 1, 1,  // 1
                                2, 1, 1,  // 4
                                2, 2, 1,  // 4
                                2, 2, 2,  // 4
                                1, 2, 2,  // 3
                                1, 1, 2,  // 2
                                1, 2, 1, // 3
                                2, 1, 2}; // 4
    row_major_data d1 {{&data[0], 8, 3}};

    std::vector<FloatT> expected {1, 4, 4, 4, 3, 2, 3, 4};

    for (size_t i = 0; i < 8; ++i)
    {
        FloatT v = at.eval<row_major_data>({d1, i});
        //std::cout << "value=" << v << ", expected = " << expected.at(i) << std::endl;
        assert(v == expected.at(i));
    }
}

void test_eval2()
{
    AddTree at;
    {
        Tree& t = at.add_tree();;
        t.root().split({0, 1.5});
        t.root().left().split({1, 1.5});
        t.root().left().left().split({2, 1.5});
        t.root().left().left().left().set_leaf_value(1.0);
        t.root().left().left().right().set_leaf_value(2.0);
        t.root().left().right().set_leaf_value(3.0);
        t.root().right().set_leaf_value(4.0);
    }
    {
        Tree& t = at.add_tree();
        t.root().set_leaf_value(10.0);
    }

    std::vector<FloatT> data = {1, 2, 2, 2, 1, 1, 1, 2,
                                1, 1, 2, 2, 2, 1, 2, 1,
                                1, 1, 1, 2, 2, 2, 1, 2};
    col_major_data d {{&data[0], 8, 3}};

    std::vector<FloatT> expected0 {1, 4, 4, 4, 3, 2, 3, 4};
    std::vector<FloatT> expected {11, 14, 14, 14, 13, 12, 13, 14};

    for (size_t i = 0; i < 8; ++i)
    {
        FloatT v = at.eval<col_major_data>({d, i});
        //std::cout << "value=" << v << ", expected = " << expected.at(i) << std::endl;
        assert(v == expected.at(i));
        v = at[0].eval<col_major_data>({d, i});
        assert(v == expected0.at(i));
    }
}

void test_prune1()
{
    AddTree at;
    Tree& t = at.add_tree();
    t.root().split({1, 8.0});
    t.root().left().split({1, 2.0});
    t.root().left().right().split({1, 4.0});

    Box box = {{1, {2.5, 5.0}}};

    //std::cout << at[0] << std::endl;
    AddTree new_at = at.prune(box);
    //std::cout << new_at[0] << std::endl;

    assert(new_at.num_leafs() == 2);
    assert(new_at.num_nodes() == 3);
}

void test_block_store1()
{
    size_t max_mem = 1024*1024; // 1mb
    BlockStore<int> store;

    std::vector<int> v {1, 2, 3, 4, 5};

    auto r = store.store(v, max_mem-store.get_used_mem_size());

    assert(r.begin[0] == 1);
    assert(r.begin[1] == 2);
    assert(r.begin[2] == 3);
    assert(r.begin[3] == 4);
    assert(r.begin[4] == 5);
    assert(r.end == r.begin+5);

    assert(store.get_used_mem_size() == 5*4);
}

void test_graph1()
{
    AddTree at;
    {
        Tree& t = at.add_tree();
        t.root().split({1, 8.0});
        t.root().left().split({2, 2.0});
        t.root().left().left().set_leaf_value(1.0);
        t.root().left().right().set_leaf_value(2.0);
        t.root().right().set_leaf_value(3.0);
    }
    {
        Tree& t = at.add_tree();
        t.root().split({1, 16.0});
        t.root().left().split({2, 4.0});
        t.root().left().right().split({1, 6.0});

        t.root().left().left().set_leaf_value(1.0);
        t.root().left().right().left().set_leaf_value(2.0);
        t.root().left().right().right().set_leaf_value(3.0);
        t.root().right().set_leaf_value(4.0);
    }

    Domain c, d;
    std::swap(c, d);

    std::cout << at[0] << std::endl;
    std::cout << at[1] << std::endl;

    Graph g(at);

    //g.store.refine_workspace({1, 9.0}, true);
    //const Box b = g.store.get_workspace_box();
    Box b {{1, Domain::from_hi_exclusive(9.0)}};

    std::cout << g << std::endl;
    std::cout << b << std::endl;
    //g.prune([b](const Box& box) {
    //            bool res = b.overlaps(box); 
    //            std::cout << b << " overlaps " << box << " -> " << res << std::endl;
    //            return !res;
    //        });
    g.prune_by_box(b, true);
    std::cout << g << std::endl;
}

//void test_node_search1()
//{
//    AddTree at;
//    {
//        Tree& t = at.add_tree();
//        t.root().split({1, 8.0});
//        t.root().left().split({2, 2.0});
//        t.root().left().left().set_leaf_value(1.0);
//        t.root().left().right().set_leaf_value(2.0);
//        t.root().right().set_leaf_value(3.0);
//    }
//    {
//        Tree& t = at.add_tree();
//        t.root().split({1, 16.0});
//        t.root().left().split({2, 4.0});
//        t.root().left().right().split({1, 6.0});
//
//        t.root().left().left().set_leaf_value(1.0);
//        t.root().left().right().left().set_leaf_value(2.0);
//        t.root().left().right().right().set_leaf_value(3.0);
//        t.root().right().set_leaf_value(4.0);
//    }
//
//    //std::cout << at[0] << std::endl;
//    //std::cout << at[1] << std::endl;
//
//    Search s(at);
//    while (!s.step()) { } 
//
//    std::cout << "number of steps " << s.stats.num_steps << std::endl;
//    std::cout << "number of impossible " << s.stats.num_impossible << std::endl;
//    std::cout << "number of solutions " << s.num_solutions() << std::endl;
//
//    std::vector<FloatT> expected {7, 6, 5, 4, 4, 3, 2};
//    for (size_t i = 0; i < s.num_solutions(); ++i)
//    {
//        Solution sol = s.get_solution(i);
//        std::cout << sol << std::endl;
//        for (size_t i = 0; i < at.size(); i++)
//            assert(at[i].node_const(sol.nodes[i]).is_leaf());
//        assert(sol.output == expected.at(i));
//    }
//}

void test_feat_map1()
{
    std::vector<std::string> features = {"feat16", "feat2", "feat3=4", "feat4"};
    FeatMap map(features);
    map.share_all_features_between_instances();
    map.use_same_id_for(map.get_feat_id("feat3=4"), map.get_feat_id("feat4"));

    assert(map.get_feat_id("feat2") == 1);
    std::vector<FeatId> expected {0, 1, 2, 2, 0, 1, 2, 2};
    for (auto index : map)
        assert(map.get_feat_id(index) == expected[index]);

    for (auto index : map.iter_instance(0))
        assert(map.get_feat_id(index) == expected[index]);
    for (auto index : map.iter_instance(1))
    {
        assert(index >= 4);
        assert(map.get_feat_id(index) == expected[index]);
    }

    //std::cout << map << std::endl;

    for (auto&&[k,v]:map.get_indices_map(0)) { assert(k < 4); assert(v < 4); }
    for (auto&&[k,v]:map.get_indices_map(1)) { assert(k < 4); assert(v >= 4); }
}

void test_feat_map2()
{
    FeatMap map(5);
    map.use_same_id_for(0, 2);
    map.use_same_id_for(0, 8);

    assert(map.get_feat_id("feature0", 0) == 0);
    assert(map.get_feat_id("feature2", 0) == 0);
    assert(map.get_feat_id("feature3", 1) == 0);
    assert(map.get_feat_id("feature0", 1) == 5);
}

void test_feat_map3()
{
    FeatMap map(3);
    map.use_same_id_for(0, 0);
    map.use_same_id_for(1, 0);
    map.use_same_id_for(map.get_index(0, 1), 0);

    AddTree at;
    {
        Tree& t = at.add_tree();
        t.root().split({0, 0.3});
        t.root().left().split({1, 0.3});
        t.root().left().right().split({2, 0.3});
    }

    AddTree renamed0 = map.transform(at, 0);
    AddTree renamed1 = map.transform(at, 1);

    //std::cout << at[0] << std::endl;
    //std::cout << renamed0[0] << std::endl;
    //std::cout << renamed1[0] << std::endl;

    //std::cout << map << std::endl;

    assert(renamed0[0].root().get_split().feat_id == 0);
    assert(renamed0[0].root().left().get_split().feat_id == 0);
    assert(renamed0[0].root().left().right().get_split().feat_id == 2);

    assert(renamed1[0].root().get_split().feat_id == 0);
    assert(renamed1[0].root().left().get_split().feat_id == 4);
    assert(renamed1[0].root().left().right().get_split().feat_id == 5);
}

void test_hash1()
{
    AddTree at;
    GraphSearch s(at);

    //size_t h1, h2;
    //h1 = h2 = 0;
    //h1 = s.hash(h1, 0, 1);
    //h1 = s.hash(h1, 1, 5);
    //h1 = s.hash(h1, 10, 24);
    //h2 = s.hash(h2, 1, 5); // reorder
    //h2 = s.hash(h2, 10, 24);
    //h2 = s.hash(h2, 0, 1);

    //std::cout << h1 << std::endl;
    //std::cout << h2 << std::endl;
    

    // build/testveritas | uniq | wc -l == build/testveritas | wc -l
    
    int N = 50;

    for (int v0 = 0; v0 < N; ++v0)
    for (int v1 = 0; v1 < N; ++v1)
    for (int v2 = 0; v2 < N; ++v2)
    for (int v3 = 0; v3 < N; ++v3)
    //for (int v4 = 0; v4 < N; ++v4)
    {
        size_t h = 0;
        h = s.hash(h, 1, v0);
        h = s.hash(h, 2, v1);
        h = s.hash(h, 3, v2);
        h = s.hash(h, 4, v3);
        //h = s.hash(h, 5, v4);
        std::cout << h << std::endl;
    }
}

void test_graph_search1()
{
    AddTree at;
    at.base_score = 10;
    {
        Tree& t = at.add_tree();
        t.root().split({1, 8.0});
        t.root().left().split({2, 4.0});
        t.root().left().right().split({1, 3.0});

        t.root().left().left().set_leaf_value(1.0);
        t.root().left().right().left().set_leaf_value(2.0);
        t.root().left().right().right().set_leaf_value(3.0);
        t.root().right().set_leaf_value(4.0);
    }
    {
        Tree& t = at.add_tree();
        t.root().split({1, 16.0});
        t.root().left().split({2, 4.0});
        t.root().left().right().split({1, 6.0});

        t.root().left().left().set_leaf_value(1.0);
        t.root().left().right().left().set_leaf_value(2.0);
        t.root().left().right().right().set_leaf_value(3.0);
        t.root().right().set_leaf_value(4.0);
    }
    std::cout << at[0] << std::endl;
    std::cout << at[1] << std::endl;
    GraphSearch s(at);
    s.step();
    s.step();
    s.step();
    while (!s.step());
}

void test_graph_search2()
{
    AddTree at;
    {
        std::ifstream f;
        f.open("tests/models/xgb-img-hard.json");
        at.from_json(f);
    }
    for (const Tree& t : at)
        std::cout << t << std::endl;

    GraphSearch s(at);

    for (size_t i = 0; i < 100; i++)
    {
        if (s.step())
        {
            std::cout << "DONE " << i << std::endl;
            break;
        }
    }
    std::cout << "numsols: " << s.num_solutions() << std::endl;
}

int main()
{
    //test_tree1();
    //test_tree2();
    //test_tree3();
    //test_json1();
    //test_json2();

    //test_eval1();
    //test_eval2();

    //test_prune1();
    //test_block_store1();

    //test_graph1();

    //test_node_search1();
    
    //test_feat_map1();
    //test_feat_map2();
    //test_feat_map3();

    //test_hash1();

    test_graph_search1();
    //test_graph_search2();

}
