#include "basics.hpp"
#include "box.hpp"
#include "tree.hpp"
#include "fp.hpp"

using namespace veritas;

int test_fp() {
    FpMap map;

    map.add(0, 1.0);
    map.add(0, 8.0);
    map.add(0, 4.0);

    map.finalize();

    // transform FloatT -> FpT
    bool result = true
        && map.transform(0, 0.0) == 0
        && map.transform(0, 1.0) == 1
        && map.transform(0, 1.5) == 1
        && map.transform(0, 4.0) == 2
        && map.transform(0, 6.0) == 2
        && map.transform(0, 8.0) == 3
        && map.transform(0, 9.0) == 3
        ;

    // itransform FpT -> FloatT
    result = result
        && map.itransform(0, 1) == 1.0
        && map.itransform(0, 2) == 4.0
        && map.itransform(0, 3) == 8.0
        ;

    //map.print();

    std::cout << "test_fp " << result << std::endl;
    return result;
}

int test_fp_interval() {
    FpMap map;
                        // 0
    map.add(0, 1.0);    // 1
    map.add(0, 4.0);    // 2
    map.add(0, 8.0);    // 3
                        // 4
    map.finalize();

    bool result = true
        && map.transform(0, {0.0, 0.9})  == IntervalFp::from_hi(1)
        && map.transform(0, {0.0, 1.0})  == IntervalFp::from_hi(1)
        && map.transform(0, {1.0, 1.1})  == IntervalFp(1, 2)
        && map.transform(0, {-1.0, 3.1}) == IntervalFp::from_hi(2)
        && map.transform(0, {-1.0, 4.0}) == IntervalFp::from_hi(2)
        && map.transform(0, {-1.0, 4.1}) == IntervalFp::from_hi(3)
        && map.transform(0, {0.99, 4.1}) == IntervalFp::from_hi(3)
        && map.transform(0, {1.0, 4.1})  == IntervalFp(1, 3)
        && map.transform(0, {1.1, 4.1})  == IntervalFp(1, 3)
        && map.transform(0, {4.0, 4.1})  == IntervalFp(2, 3)
        && map.transform(0, {4.0, 8.0})  == IntervalFp(2, 3)
        && map.transform(0, {4.0, 8.1})  == IntervalFp::from_lo(2)
        && map.transform(0, {4.1, 8.1})  == IntervalFp::from_lo(2)
        && map.transform(0, {-4.0, 8.1}) == IntervalFp()
        ;

    std::cout << "test_fp_interval " << result << std::endl;
    return result;
}

int test_fp_with_tree1() {
    Tree t(2);
    t.split(t.root(), {1, 4.1});
    t.split(t["l"], {1, 2.1});
    t.split(t["lr"], {1, 8.1});
    t.leaf_value(t["ll"], 0) = 15.124;
    t.leaf_value(t["r"], 0) = 12.22;
    t.leaf_value(t["ll"], 1) = 10.124;
    t.leaf_value(t["r"], 1) = 10.222;

    FpMap map;
    map.add(t);
    map.add(1, 3.245); // add some random additional points
    map.add(1, 10.1241);
    map.finalize();

    TreeFp tcheck(2);
    tcheck.split(tcheck.root(), {1, 3});
    tcheck.split(tcheck["l"], {1, 1});
    tcheck.split(tcheck["lr"], {1, 4});
    tcheck.leaf_value(tcheck["ll"], 0) = 15.124;
    tcheck.leaf_value(tcheck["r"], 0) = 12.22;
    tcheck.leaf_value(tcheck["ll"], 1) = 10.124;
    tcheck.leaf_value(tcheck["r"], 1) = 10.222;

    TreeFp tt = map.transform(t);

    map.print();

    std::cout << t << std::endl;
    std::cout << tt << std::endl;
    std::cout << "\ntcheck\n" << tcheck << std::endl;

    Box::BufT buf;
    Box box(buf);
    BoxFp::BufT buffp;
    BoxFp boxfp(buffp);

    bool result = true;
    result = result && tcheck == tt;

    for (NodeId id : {3, 5, 6, 2}) {
        std::cout << "\nnode " << id << std::endl;
        box.clear();
        t.compute_box(id, box);
        boxfp.clear();
        tt.compute_box(id, boxfp);

        IntervalFp a = buffp[0].interval; // interval for feat_id 1
        IntervalFp b = map.transform(1, buf[0].interval); // interval for feat_id 1

        std::cout << a << "->" << b << "\n";

        Interval c = buf[0].interval;
        Interval d = map.itransform(1, buffp[0].interval);

        std::cout << c << "->" << d << "\n";

        result = result && (a == b) && (c == d);
    }

    std::cout << "test_fp_with_tree1 " << result << std::endl;
    return result;
}

int main_fp() {
    int result = 1
        && test_fp()
        && test_fp_interval()
        && test_fp_with_tree1()
        ;
    return !result;
}
