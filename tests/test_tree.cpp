#include "basics.hpp"
#include "box.hpp"
#include "tree.hpp"

using namespace veritas;

int test_tree1() {
    TreeFp tree;
    auto n = tree.root();
    n.split({1, 5});
    n.left().set_leaf_value(4);
    n.right().set_leaf_value(2);

    bool result = true
        && n.is_root()
        && n.get_split() == LtSplitFp(1, 5)
        && !n.left().is_root()
        && n.left().is_leaf()
        && n.left().leaf_value() == 4
        && n.right().leaf_value() == 2
        && n.num_leafs() == 2
        && n.tree_size() == 3
        ;

    std::cout << "test_tree1 " << result << std::endl;
    return result;
}

int test_tree2() {
    Tree tree;
    auto n = tree.root();
    n.split(1); // boolean split
    n.left().set_leaf_value(4.0);
    n.right().set_leaf_value(2.0);

    bool result = true
        && n.is_root()
        && n.get_split() == LtSplit(1, BOOL_SPLIT_VALUE)
        && !n.left().is_root()
        && n.left().is_leaf()
        && n.left().leaf_value() == 4.0
        && n.right().leaf_value() == 2.0
        && n.num_leafs() == 2
        && n.tree_size() == 3
        ;

    std::cout << "test_tree2 " << result << std::endl;
    return result;
}

int test_generic_tree1() {
    GTree<GLtSplit<bool>, char> tree;
    auto n = tree.root();
    n.split({1, true});
    n.left().set_leaf_value('a');
    n.right().set_leaf_value('b');

    int result = true
        && n.get_split() == GLtSplit(1, true)
        && n.left().leaf_value() == 'a'
        && n.right().leaf_value() == 'b'
        ;

    std::cout << "test_generic_tree1 " << result << std::endl;
    return result;
}

int test_get_splits1() {
    Tree tree;
    auto n = tree.root();
    n.split({1, 2.0});
    n.left().split({2, 4.0});
    n.left().right().split({2, 8.0});
    auto splits = tree.get_splits();

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
    TreeFp tree;
    auto n = tree.root();
    n.split({1, 2});
    n.left().split({2, 4});
    n.left().right().split({2, 8});
    auto splits = tree.get_splits();

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

int main_tree() {
    int result = 1
        && test_tree1()
        && test_tree2()
        && test_generic_tree1()
        && test_get_splits1()
        && test_get_splits2()
        ;
    return !result;
}
