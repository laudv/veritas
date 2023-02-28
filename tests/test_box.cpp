#include "basics.hpp"
#include "box.hpp"
#include <stdexcept>

using namespace veritas;

int test_box_overlaps() {
    Box::BufT box1 {{0, Interval(0.0, 1.0)}, {1, Interval(0.0, 1.0)}};
    Box::BufT box2 {{0, Interval(0.0, 5.0)}, {2, Interval(0.0, 1.0)}};
    Box::BufT box3 {{0, Interval(2.0, 5.0)}, {2, Interval(0.2, 0.5)}};

    BoxRef b1{box1};
    BoxRef b2{box2};
    BoxRef b3{box3};

    bool result = true
        && b1.overlaps(b2)
        && b2.overlaps(b1)
        && b2.overlaps(b3)
        && !b1.overlaps(b3)
        ;

    std::cout << "test_box_overlaps " << result << std::endl;
    return result;
}

int test_box_get_or_insert() {
    BoxFp::BufT buf1{{0, {0, 1}}, {2, {0, 2}}};
    BoxFp b1{buf1};

    bool result = true
        && b1.get_or_insert(0) == IntervalFp(0, 1)
        && b1.get_or_insert(1).is_everything()
        && b1.get_or_insert(2) == IntervalFp(0, 2)
        && buf1.at(1).feat_id == 1
        && buf1.at(1).interval == IntervalFp();

    std::cout << "test_box_get_or_insert " << result << std::endl;
    return result;
}

int test_box_refine() {
    BoxFp::BufT buf1{{0, {0, 1}}, {2, {0, 2}}};
    BoxFp b1{buf1};

    b1.refine_box(1, IntervalFp(0, 10));
    b1.refine_box(2, IntervalFp(1, 3));

    bool result = true
        && buf1.at(1).feat_id == 1
        && buf1.at(1).interval == IntervalFp(0, 10)
        && buf1.at(2).feat_id == 2
        && buf1.at(2).interval == IntervalFp(1, 2);

    b1.refine_box(LtSplitFp(1, 5), false);

    result = result
        && buf1.at(1).interval == IntervalFp(5, 10);

    b1.refine_box(LtSplitFp(1, 7), true);
    b1.refine_box(LtSplitFp(5, 2), true);

    result = result
        && buf1.at(1).interval == IntervalFp(5, 7)
        && buf1.at(3).feat_id == 5
        && buf1.at(3).interval == IntervalFp(Limits<FpT>::min, 2);

    std::cout << "test_box_refine " << result << std::endl;
    return result;
}

int test_to_flatbox() {
    FlatBoxFp fbox(3, IntervalFp());
    BoxFp::BufT buf1{{0, {0, 1}}, {2, {0, 2}}};
    BoxRefFp b1{buf1};

    b1.to_flatbox(fbox);

    bool result = true
        && fbox.at(0) == IntervalFp(0, 1)
        && fbox.at(1) == IntervalFp()
        && fbox.at(2) == IntervalFp(0, 2);

    std::cout << "test_to_flatbox " << result << std::endl;
    return result;
}

int test_combine_boxes() {
    BoxFp::BufT buf1{{0, {1, 5}}, {1, {0, 9}}};
    BoxFp::BufT buf2{{0, {0, 2}}, {5, {0, 2}}};
    BoxFp::BufT buf3;
    BoxFp b3{buf3};

    // with copy_b == true
    b3.combine_boxes(BoxRefFp(buf1), BoxRefFp(buf2), true);

    bool result = true
        && buf3.size() == 3
        && buf3.at(0) == IntervalPairFp(0, {1, 2})
        && buf3.at(1) == IntervalPairFp(1, {0, 9})
        && buf3.at(2) == IntervalPairFp(5, {0, 2});

    // with copy_b == false
    b3.clear();
    b3.combine_boxes(BoxRefFp(buf1), BoxRefFp(buf2), false);

    result = result
        && buf3.size() == 2
        && buf3.at(0) == IntervalPairFp(0, {1, 2})
        && buf3.at(1) == IntervalPairFp(1, {0, 9});

    if (check_sanity()) {
        try {
            BoxFp::BufT buf4{{0, {0, 1}}}; // does not overlap with buf1
            b3.clear();
            b3.combine_boxes(BoxRefFp(buf1), BoxRefFp(buf4), false);
            result = false;
        } catch (const std::invalid_argument& e) {}
    }

    std::cout << "test_combine_boxes " << result << std::endl;
    return result;
}

int test_clear() {
    BoxFp::BufT buf1{{0, {0, 5}}, {1, {0, 9}}, {0, {0, 2}}};
    BoxFp b1(buf1, 2);

    b1.clear();

    bool result = true
        && buf1.size() == 2
        && buf1.at(0) == IntervalPairFp(0, {0, 5})
        && buf1.at(1) == IntervalPairFp(1, {0, 9});

    std::cout << "test_clear " << result << std::endl;
    return result;
}

int main_box() {
    int result = 1
        && test_box_overlaps()
        && test_box_get_or_insert()
        && test_box_refine()
        && test_to_flatbox()
        && test_clear()
        && test_combine_boxes()
        ;
    return !result;
}
