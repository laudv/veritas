#include "basics.hpp"
#include "box.hpp"
#include "interval.hpp"
#include "tree.hpp"

using namespace veritas;

int test_tree1() {
    TreeFp t(1);
    t.split(t[""], {1, 5});
    t.leaf_value(t["l"], 0) = 4;
    t.leaf_value(t["r"], 0) = 2;

    bool result = true
        && t.get_split(0) == LtSplitFp(1, 5)
        && !t.is_root(t["l"])
        && t.is_leaf(t["l"])
        && t.leaf_value(t["l"], 0) == 4
        && t.leaf_value(t["r"], 0) == 2
        && t.num_leaves() == 2
        && t.tree_size(t.root()) == 3
        && t.tree_size(t["l"]) == 1
        && t.tree_size(t["r"]) == 1
        && t.num_nodes() == 3
        ;

    std::cout << "test_tree1 " << result << std::endl;
    return result;
}

int test_tree2() {
    Tree t(1);
    t.split(t.root(), bool_ltsplit(1));
    t.leaf_value(t["l"], 0) = 4;
    t.leaf_value(t["r"], 0) = 2;

    bool result = true
        && t.get_split(0) == LtSplit(1, BOOL_SPLIT_VALUE)
        && !t.is_root(t["l"])
        && t.is_leaf(t["l"])
        && t.leaf_value(t["l"], 0) == 4
        && t.leaf_value(t["r"], 0) == 2
        && t.num_leaves() == 2
        && t.tree_size(t.root()) == 3
        && t.tree_size(t["l"]) == 1
        && t.tree_size(t["r"]) == 1
        && t.num_nodes() == 3
        ;

    std::cout << "test_tree2 " << result << std::endl;
    return result;
}

int test_tree_multi() {
    Tree t(2);
    t.split(t.root(), bool_ltsplit(1));
    t.leaf_value(t["l"], 0) = 4;
    t.leaf_value(t["r"], 0) = 2;
    t.leaf_value(t["l"], 1) = 8;
    t.leaf_value(t["r"], 1) = 6;

    bool result = true
        && t.get_split(0) == LtSplit(1, BOOL_SPLIT_VALUE)
        && !t.is_root(t["l"])
        && t.is_leaf(t["l"])
        && t.leaf_value(t["l"], 0) == 4
        && t.leaf_value(t["r"], 0) == 2
        && t.leaf_value(t["l"], 1) == 8
        && t.leaf_value(t["r"], 1) == 6
        && t.num_leaves() == 2
        && t.tree_size(t.root()) == 3
        && t.tree_size(t["l"]) == 1
        && t.tree_size(t["r"]) == 1
        && t.num_nodes() == 3
        ;

    std::cout << "test_tree_multi " << result << std::endl;
    return result;
}

int test_generic_tree1() {
    GTree<GLtSplit<bool>, char> t(1);
    t.split(t.root(), {1, true});
    t.leaf_value(t["l"], 0) = 'a';
    t.leaf_value(t["r"], 0) = 'b';

    int result = true
        && t.get_split(t[""]) == GLtSplit(1, true)
        && t.leaf_value(t["l"], 0) == 'a'
        && t.leaf_value(t["r"], 0) == 'b'
        ;

    std::cout << "test_generic_tree1 " << result << std::endl;
    return result;
}

int test_get_splits1() {
    Tree t(1);
    t.split(t.root(), {1, 2.0});
    t.split(t["l"], {2, 4.0});
    t.split(t["r"], {2, 8.0});
    auto splits = t.get_splits();

    bool result = true
        && splits[1][0] == 2.0
        && splits[2][0] == 4.0
        && splits[2][1] == 8.0
        && splits[1].size() == 1
        && splits[2].size() == 2
        ;

    std::cout << "test_get_splits1 " << result << std::endl;
    return result;
}

int test_get_splits2() {
    TreeFp t(1);
    t.split(t.root(), {1, 2});
    t.split(t["l"], {2, 4});
    t.split(t["r"], {2, 8});
    auto splits = t.get_splits();

    bool result = true
        && splits[1][0] == 2
        && splits[2][0] == 4
        && splits[2][1] == 8
        && splits[1].size() == 1
        && splits[2].size() == 2
        ;

    std::cout << "test_get_splits2 " << result << std::endl;
    return result;
}

int test_compute_box1() {
    Tree t(1);
    t.split(t.root(), {1, 8.0});
    t.split(t["l"], {1, 2.0});
    t.split(t["lr"], {1, 4.0});

    bool result = true;

    Tree::BoxT::BufT buf;
    Tree::BoxT box(buf);
    t.compute_box(t["lrl"], box);
    result = result && buf[0] == IntervalPair(1, Interval(2.0, 4.0));

    box.clear();
    t.compute_box(t["lrr"], box);
    result = result && buf[0] == IntervalPair(1, Interval(4.0, 8.0));

    box.clear();
    t.compute_box(t["l"], box);
    result = result && buf[0] == IntervalPair(1, Interval::from_hi(8.0));

    std::cout << "test_compute_box1 " << result << std::endl;
    return result;
}

int test_eval1() {
    GTree<LtSplit, int> t(2);
    t.split(t.root(), {0, 1.5});
    t.split(t["l"], {1, 1.5});
    t.split(t["ll"], {2, 1.5});
    t.leaf_value(t["lll"], 0) = 1;
    t.leaf_value(t["llr"], 0) = 2;
    t.leaf_value(t["lr"], 0) = 3;
    t.leaf_value(t["r"], 0) = 4;
    t.leaf_value(t["lll"], 1) = -1;
    t.leaf_value(t["llr"], 1) = -2;
    t.leaf_value(t["lr"], 1) = -3;
    t.leaf_value(t["r"], 1) = -4;

    std::vector<FloatT> buf = {1, 1, 1,  // 1
                               2, 1, 1,  // 4
                               2, 2, 1,  // 4
                               2, 2, 2,  // 4
                               1, 2, 2,  // 3
                               1, 1, 2,  // 2
                               1, 2, 1, // 3
                               2, 1, 2}; // 4
    data d {&buf[0], 8, 3, 3, 1};

    bool result = true
        && d.row(6)[0] == 1
        && d.row(6)[1] == 2
        && d.row(6)[2] == 1
        ;

    std::vector<FloatT> expected {1, 4, 4, 4, 3, 2, 3, 4};

    for (size_t i = 0; i < 8; ++i) {
        NodeId leaf = t.eval_node(d.row(i));
        result = result && (t.leaf_value(leaf, 0) == expected.at(i));
        result = result && (t.leaf_value(leaf, 1) == -expected.at(i));
    }

    std::cout << "test_eval1 " << result << std::endl;
    return result;
}

int test_find_minmax() {
    GTree<LtSplit, int> t(1);
    t.split(t.root(), {0, 1.5});
    t.split(t["l"], {1, 1.5});
    t.split(t["ll"], {2, 1.5});
    t.leaf_value(t["lll"], 0) = 1;
    t.leaf_value(t["llr"], 0) = 2;
    t.leaf_value(t["lr"], 0) = 3;
    t.leaf_value(t["r"], 0) = 4;

    auto minmax1 = t.find_minmax_leaf_value();

    bool result = true
        && minmax1[0].first == 1
        && minmax1[0].second == 4;

    decltype(minmax1) minmax2 {{0, 0}};
    t.find_minmax_leaf_value(t["l"], minmax2);

    result = result
        && minmax2[0].first == 1
        && minmax2[0].second == 2;

    std::cout << "test_find_minmax " << result << std::endl;
    return result;
}

int test_prune1() {
    TreeFp t(1);
    t.split(t[""], {1, 5});
    t.leaf_value(t["l"], 0) = 4;
    t.leaf_value(t["r"], 0) = 2;

    BoxFp::BufT buf {{1, {2,4}}}; // left only
    TreeFp tleft = t.prune(BoxRefFp(buf));
    buf = {{1, {5,8}}};
    TreeFp tright = t.prune(BoxRefFp(buf));
    buf = {{1, {4,8}}};
    TreeFp tboth = t.prune(BoxRefFp(buf));

    bool result = true
        && tleft.is_root(tleft.root())
        && tright.is_root(tright.root())
        && tleft.leaf_value(tleft.root(), 0) == 4
        && tright.leaf_value(tright.root(), 0) == 2
        && tboth.num_nodes() == 3
        && tboth.get_split(tboth.root()) == LtSplitFp{1, 5}
        && tboth.leaf_value(tboth["l"], 0) == 4
        && tboth.leaf_value(tboth["r"], 0) == 2
        ;

    std::cout << "test_prune1 " << result << std::endl;
    return result;
}

int test_prune2() {
    TreeFp t(1);
    t.split(t[""], {1, 5});
    t.split(t["l"], {1, 4});
    t.split(t["ll"], {1, 3});
    t.split(t["lll"], {1, 2});
    t.leaf_value(t["r"], 0) = 5;
    t.leaf_value(t["lr"], 0) = 4;
    t.leaf_value(t["llr"], 0) = 3;
    t.leaf_value(t["lllr"], 0) = 2;
    t.leaf_value(t["llll"], 0) = 1;


    std::cout << t << std::endl;

    bool result = true;

    std::vector<FpT> d {0, 1};
    d[1] = 1; result &= t.eval_node(data<FpT>(d)) == t["llll"];
    d[1] = 2; result &= t.eval_node(data<FpT>(d)) == t["lllr"];
    d[1] = 3; result &= t.eval_node(data<FpT>(d)) == t["llr"];
    d[1] = 4; result &= t.eval_node(data<FpT>(d)) == t["lr"];
    d[1] = 5; result &= t.eval_node(data<FpT>(d)) == t["r"];
    d[1] = 6; result &= t.eval_node(data<FpT>(d)) == t["r"];

    std::vector<FloatT> r { 0.0 };
    data<FloatT> dr(r);

    d[1] = 1; r[0] = 0; t.eval(data<FpT>(d), dr); result &= r[0] == 1.0;
    d[1] = 2; r[0] = 0; t.eval(data<FpT>(d), dr); result &= r[0] == 2.0;
    d[1] = 3; r[0] = 0; t.eval(data<FpT>(d), dr); result &= r[0] == 3.0;
    d[1] = 4; r[0] = 0; t.eval(data<FpT>(d), dr); result &= r[0] == 4.0;
    d[1] = 5; r[0] = 0; t.eval(data<FpT>(d), dr); result &= r[0] == 5.0;
    d[1] = 6; r[0] = 0; t.eval(data<FpT>(d), dr); result &= r[0] == 5.0;

    BoxFp::BufT buf {{1, {2,4}}}; // left only
    TreeFp tp = t.prune(BoxRefFp(buf));

    result = result
        && tp.num_nodes() == 3
        && tp.get_split(tp.root()) == LtSplitFp(1, 3)
        && tp.leaf_value(tp["l"], 0) == 2.0
        && tp.leaf_value(tp["r"], 0) == 3.0
        ;

    std::cout << "test_prune2 " << result << std::endl;
    return result;
}

int test_negate_leaf_values() {
    TreeFp t(1);
    t.split(t[""], {1, 5});
    t.leaf_value(t["l"], 0) = 4;
    t.leaf_value(t["r"], 0) = 2;

    TreeFp tneg = t.negate_leaf_values();

    int result = true
        && tneg.leaf_value(tneg["l"], 0) == -4
        && tneg.leaf_value(tneg["r"], 0) == -2
        ;

    std::cout << "test_negate_leaf_values " << result << std::endl;
    return result;
}

int test_make_multiclass() {
    TreeFp t(1);
    t.split(t[""], {1, 5});
    t.leaf_value(t["l"], 0) = 4;
    t.leaf_value(t["r"], 0) = 2;

    int c = 3;
    TreeFp tm = t.make_multiclass(c, 10);

    int result = true
        && tm.leaf_value(tm["l"], c) == 4
        && tm.leaf_value(tm["r"], c) == 2;
        ;

    for (int cc = 0; cc < 10; ++cc) {
        if (cc == c) continue;
        result &= tm.leaf_value(tm["l"], cc) == 0;
        result &= tm.leaf_value(tm["r"], cc) == 0;
    }

    // swap_class
    tm.swap_class(c);

    result &= true
        && tm.leaf_value(tm["l"], 0) == 4
        && tm.leaf_value(tm["r"], 0) == 2;
        ;

    for (int cc = 1; cc < 10; ++cc) {
        result &= tm.leaf_value(tm["l"], cc) == 0;
        result &= tm.leaf_value(tm["r"], cc) == 0;
    }

    std::cout << "test_make_multiclass " << result << std::endl;
    return result;
}

int main_tree() {
    int result = 1
        && test_tree1()
        && test_tree2()
        && test_tree_multi()
        && test_generic_tree1()
        && test_get_splits1()
        && test_get_splits2()
        && test_compute_box1()
        && test_eval1()
        && test_prune1()
        && test_prune2()
        && test_negate_leaf_values()
        && test_make_multiclass()
        ;
    return !result;
}
