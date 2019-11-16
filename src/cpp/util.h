#ifndef TREECK_UTIL_H
#define TREECK_UTIL_H

#include <type_traits>

namespace treeck {
    namespace util {
        template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
        template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
        template<class T> struct always_false : std::false_type {};
    } /* namespace util */
} /* namespace treeck */



#endif /* TREECK_UTIL_H */
