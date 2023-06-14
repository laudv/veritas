#include "basics.hpp"
#include "box.hpp"
#include "interval.hpp"
#include "tree.hpp"
#include "leafiter.hpp"

using namespace veritas;

int test_leafiter1() {
    TreeFp t(1);
    t.split(t[""], {0, 2});
    t.split(t["l"], {0, 1});
    t.split(t["r"], {0, 5});

    std::cout << t << std::endl;

    FlatBoxFp prune_box;
    LeafIter<TreeFp> iter;
    bool result = true;

    BoxFp::BufT buf1{{0, {1, 3}}, {2, {0, 2}}};
    iter.setup(t, BoxRefFp(buf1), prune_box);
    for (NodeId expected_id : {4, 5, -1})
        result = result && (expected_id == iter.next());
    std::cout << "buf1 << " << result << std::endl;

    BoxFp::BufT buf2{{0, {0, 3}}};
    iter.setup(t, BoxRefFp(buf2), prune_box);
    for (NodeId expected_id : {3, 4, 5, -1})
        result = result && (expected_id == iter.next());
    std::cout << "buf2 << " << result << std::endl;

    BoxFp::BufT buf3{{0, {1, 2}}};
    iter.setup(t, BoxRefFp(buf3), prune_box);
    for (NodeId expected_id : {4, -1, -1})
        result = result && (expected_id == iter.next());
    std::cout << "buf3 << " << result << std::endl;

    BoxFp::BufT buf4{{0, IntervalFp::from_lo(10)}};
    iter.setup(t, BoxRefFp(buf4), prune_box);
    for (NodeId expected_id : {6, -1})
        result = result && (expected_id == iter.next());
    std::cout << "buf4 << " << result << std::endl;

    BoxFp::BufT buf5{{0, IntervalFp()}};
    iter.setup(t, BoxRefFp(buf5), prune_box);
    for (NodeId expected_id : {3, 4, 5, 6, -1})
        result = result && (expected_id == iter.next());
    std::cout << "buf5 << " << result << std::endl;

    prune_box = {{1, 2}};
    BoxFp::BufT buf6{{0, IntervalFp()}};
    iter.setup(t, BoxRefFp(buf6), prune_box);
    for (NodeId expected_id : {4, -1})
        result = result && (expected_id == iter.next());
    std::cout << "buf6 << " << result << std::endl;

    prune_box = {{0, 3}};
    BoxFp::BufT buf7{{0, {1, 2}}};
    iter.setup(t, BoxRefFp(buf7), prune_box);
    for (NodeId expected_id : {4, -1})
        result = result && (expected_id == iter.next());
    std::cout << "buf7 << " << result << std::endl;

    std::cout << "test_leafiter1 " << result << std::endl;
    return result;
}

int main_leafiter() {
    int result = 1
        && test_leafiter1()
        ;
    return !result;
}
