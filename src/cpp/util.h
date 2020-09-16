/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_UTIL_H
#define VERITAS_UTIL_H

#include <sstream>
#include <type_traits>
#include <variant>

namespace veritas {
    namespace util {
        template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
        template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
        template<class T> struct always_false : std::false_type {};

        template <typename T>
        struct Typename { static const char* get(); };

#define VERITAS_ENABLE_TYPENAME(T) \
        template <> struct util::Typename<T> { static const char* get() { return #T; } }

        [[maybe_unused]] static void
        _get_or_msg(std::ostringstream&) {}

        template <typename A, typename... Args>
        static void
        _get_or_msg(std::ostringstream& ss, A arg, Args... args)
        {
            ss << arg;
            _get_or_msg(ss, args...);
        }

        template <typename R, typename V, typename... Args>
        static const R&
        get_or(const V& v, const Args&... args)
        {
            if (std::holds_alternative<R>(v))
            {
                return std::get<R>(v);
            }
            else
            {
                std::ostringstream ss;
                ss << "Expected " << Typename<R>::get();
                _get_or_msg(ss, args...);
                throw std::runtime_error(ss.str());
            }
        }
    } /* namespace util */
} /* namespace veritas */



#endif /* VERITAS_UTIL_H */
