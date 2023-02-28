#include "basics.hpp"
#include "interval.hpp"
#include <iostream>
#include <stdexcept>

using namespace veritas;

int test_is_everything() {
    IntervalFp ival0;
    IntervalFp ival1{1, 10};

    Interval ival2;
    Interval ival3{1.0, 10.0};

    bool result = true
        &&  ival0.is_everything()
        && !ival1.is_everything()
        &&  ival2.is_everything()
        && !ival3.is_everything();

    std::cout << "test_is_everything " << result << std::endl;
    return result;
}

int test_overlaps() {
    IntervalFp ival0{2, 10};
    IntervalFp ival1{2, 10}; // yes
    IntervalFp ival2{0, 10}; // yes
    IntervalFp ival3{0, 20}; // yes
    IntervalFp ival4{3, 20}; // yes
    IntervalFp ival5{3, 5}; // yes
    IntervalFp ival6{100, 400}; // no
    IntervalFp ival7{0, 2}; // no

    bool result = true
        &&  ival0.overlaps(ival1)
        &&  ival0.overlaps(ival2)
        &&  ival0.overlaps(ival3)
        &&  ival0.overlaps(ival4)
        &&  ival0.overlaps(ival5)
        && !ival0.overlaps(ival6)
        && !ival0.overlaps(ival7);

    std::cout << "test_overlaps " << result << std::endl;
    return result;
}

int test_intersect() {
    bool result = true
        && IntervalFp(0, 1).intersect({0, 1}) == IntervalFp(0, 1)
        && IntervalFp(0, 10).intersect({5, 10}) == IntervalFp(5, 10)
        && IntervalFp(2, 10).intersect({0, 5}) == IntervalFp(2, 5)
        && IntervalFp(1, 2).intersect({0, 5}) == IntervalFp(1, 2)
        && Interval(0.0, 1.0).intersect({0.0, 1.0}) == Interval(0.0, 1.0)
        && Interval(0.0, 10.0).intersect({5.0, 10.0}) == Interval(5.0, 10.0)
        && Interval(0.0, 10.0).intersect({-2.0, 5.0}) == Interval(0.0, 5.0)
        && Interval(0.0, 2.0).intersect({-2.0, 5.0}) == Interval(0.0, 2.0);

    if (check_sanity()) {
        bool catch_result;
        try {
            auto x = IntervalFp(0, 2).intersect({6, 9});
            catch_result = x == IntervalFp();
        } catch (const std::invalid_argument& e) {
            catch_result = true;
        }
        result = result && catch_result;
    }

    std::cout << "test_intersect " << result << std::endl;
    return result;
}

int test_contains() {
    bool result = true
        && Interval(0.0, 1.0).contains(0.0)
        && Interval(0.0, 1.0).contains(0.5)
        && !Interval(0.0, 1.0).contains(1.5)
        && IntervalFp(0, 10).contains(0)
        && IntervalFp(0, 10).contains(5)
        && !IntervalFp(0, 10).contains(-1)
        && !IntervalFp(0, 10).contains(10)
        && !IntervalFp(0, 10).contains(11)
        ;

    std::cout << "test_contains " << result << std::endl;
    return result;

}

int test_split() {

    auto [ival0, ival1] = Interval(0.0, 1.0).split(0.5);
    auto [ival2, ival3] = IntervalFp(0, 2).split(1);

    bool result = true
        && ival0 == Interval(0.0, 0.5)
        && ival1 == Interval(0.5, 1.0)
        && ival2 == IntervalFp(0, 1)
        && ival3 == IntervalFp(1, 2);

    if (check_sanity()) {
        bool catch_result;
        try {
            auto [ival0, ival1] = IntervalFp(0, 1).split(1);
            catch_result = ival0 == IntervalFp();
        } catch (const std::invalid_argument& e) {
            catch_result = true;
        }
        result = result && catch_result;
    }

    std::cout << "test_split " << result << std::endl;
    return result;
}

int main_interval() {
    bool result = 1
        && test_is_everything()
        && test_overlaps()
        && test_intersect()
        && test_contains()
        && test_split()
        ;
    return !result;
}
