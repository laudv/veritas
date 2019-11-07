#ifndef TREECK_TREE_H
#define TREECK_TREE_H

#include <tuple>
#include <variant>

#include "domain.h"

namespace treeck {

    struct SplitBase {
        int feat_id;
    };

    struct LtSplit : public SplitBase {
        using ValueT = double;

        ValueT split_value;

        LtSplit(ValueT split_value);
        std::tuple<RealDomain, RealDomain> get_domains() const;

        bool test(ValueT value) const;
    };

    struct EqSplit : public SplitBase {
        using ValueT = int;

        ValueT category;

        EqSplit(ValueT category);
        bool test(ValueT value) const;
    };

    using Split = std::variant<LtSplit, EqSplit>;

    template <typename T>
    bool
    test_split(Split& split, T value);
}

#endif /* TREECK_TREE_H */
