#include "basics.hpp"
#include "box.hpp"
#include "interval.hpp"
#include "tree.hpp"

using namespace veritas;

int test_tree1() {
    TreeFp t;
    t.split(t[""], {1, 5});
    t.leaf_value(t["l"]) = 4;
    t.leaf_value(t["r"]) = 2;

    bool result = true
        && t.get_split(0) == LtSplitFp(1, 5)
        && !t.is_root(t["l"])
        && t.is_leaf(t["l"])
        && t.leaf_value(t["l"]) == 4
        && t.leaf_value(t["r"]) == 2
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
    Tree t;
    t.split(t.root(), bool_ltsplit(1));
    t.leaf_value(t["l"]) = 4;
    t.leaf_value(t["r"]) = 2;

    bool result = true
        && t.get_split(0) == LtSplit(1, BOOL_SPLIT_VALUE)
        && !t.is_root(t["l"])
        && t.is_leaf(t["l"])
        && t.leaf_value(t["l"]) == 4
        && t.leaf_value(t["r"]) == 2
        && t.num_leaves() == 2
        && t.tree_size(t.root()) == 3
        && t.tree_size(t["l"]) == 1
        && t.tree_size(t["r"]) == 1
        && t.num_nodes() == 3
        ;

    std::cout << "test_tree2 " << result << std::endl;
    return result;
}

int test_generic_tree1() {
    GTree<GLtSplit<bool>, char> t;
    t.split(t.root(), {1, true});
    t.leaf_value(t["l"]) = 'a';
    t.leaf_value(t["r"]) = 'b';

    int result = true
        && t.get_split(t[""]) == GLtSplit(1, true)
        && t.leaf_value(t["l"]) == 'a'
        && t.leaf_value(t["r"]) == 'b'
        ;

    std::cout << "test_generic_tree1 " << result << std::endl;
    return result;
}

int test_get_splits1() {
    Tree t;
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
    TreeFp t;
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
    Tree t;
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
    GTree<LtSplit, int> t;
    t.split(t.root(), {0, 1.5});
    t.split(t["l"], {1, 1.5});
    t.split(t["ll"], {2, 1.5});
    t.leaf_value(t["lll"]) = 1;
    t.leaf_value(t["llr"]) = 2;
    t.leaf_value(t["lr"]) = 3;
    t.leaf_value(t["r"]) = 4;

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

    for (size_t i = 0; i < 8; ++i)
    {
        FloatT v = t.eval(d.row(i));
        //std::cout << "value=" << v << ", expected = " << expected.at(i) << std::endl;
        result = result && (v == expected.at(i));
    }

    std::cout << "test_eval1 " << result << std::endl;
    return result;
}

int test_find_minmax() {
    GTree<LtSplit, int> t;
    t.split(t.root(), {0, 1.5});
    t.split(t["l"], {1, 1.5});
    t.split(t["ll"], {2, 1.5});
    t.leaf_value(t["lll"]) = 1;
    t.leaf_value(t["llr"]) = 2;
    t.leaf_value(t["lr"]) = 3;
    t.leaf_value(t["r"]) = 4;

    auto&& [min, max] = t.find_minmax_leaf_value();

    bool result = true
        && min == 1
        && max == 4;

    auto&& [min2, max2] = t.find_minmax_leaf_value(t["l"]);

    result = result
        && min2 == 1
        && max2 == 2;

    std::cout << "test_find_minmax " << result << std::endl;
    return result;
}

int main_tree() {
    int result = 1
        && test_tree1()
        && test_tree2()
        && test_generic_tree1()
        && test_get_splits1()
        && test_get_splits2()
        && test_compute_box1()
        && test_eval1()
        ;
    return !result;
}
