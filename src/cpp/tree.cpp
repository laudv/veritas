#include <limits>
#include <type_traits>

#include "tree.h"

namespace treeck {

    LtSplit::LtSplit(LtSplit::ValueT split_value) : split_value(split_value) {}

    std::tuple<RealDomain, RealDomain>
    LtSplit::get_domains() const
    {
        auto dom = RealDomain();
        return dom.split(this->split_value);
    }

    bool
    LtSplit::test(LtSplit::ValueT value) const
    {
        return value < this->split_value;
    }


    EqSplit::EqSplit(EqSplit::ValueT category) : category(category) {}

    bool
    EqSplit::test(EqSplit::ValueT value) const
    {
        return value == this->category;
    }


    template<class T> struct always_false : std::false_type {};

    template <typename T>
    bool
    test_split(Split& split, T value)
    {
        return std::visit([value](auto&& split) -> bool {
            using S = std::decay_t<decltype(split)>;
            static_assert(std::is_same_v<T, S::ValueT>, "invalid test type T");
            return split.test(value);
            //if constexpr (std::is_same_v<S, LtSplit>)
            //    return value < split.split_value;
            //else if constexpr (std::is_same_v<S, EqSplit>)
            //    return value == split.category;
            //else 
            //    static_assert(always_false<T>::value, "non-exhaustive visitor!");
        }, split);
    }

    template <> bool test_split(Split& split, LtSplit::ValueT);
    template <> bool test_split(Split& split, EqSplit::ValueT);
}
