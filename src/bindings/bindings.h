/**
 * \file bindings.h
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef BINDINGS_H
#define BINDINGS_H

#include "addtree.hpp"
#include "basics.hpp"
#include "box.hpp"
#include "tree.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

// Utility
template <typename T>
std::string tostr(const T& o) {
    std::stringstream s;
    s << o;
    return s.str();
}

/** Convert between Python list and C++ Veritas box. */
veritas::Box::BufT tobox(pybind11::object pybox);

/** Get the veritas data object from a Python handle. */
veritas::data<veritas::FloatT> get_data(pybind11::handle h);

/* Avoid invalid pointers to Tree's by storing indexes rather than pointers */
struct TreeRef {
    std::shared_ptr<veritas::AddTree> at;
    size_t i;
    inline veritas::Tree& get() { return at->operator[](i); }
    inline const veritas::Tree& get() const { return at->operator[](i); }
};




///////////////////////////////////////////////////////////////////////////////
// Module elements
///////////////////////////////////////////////////////////////////////////////

void init_interval(pybind11::module& m);    /* bindings/interval.cpp */
void init_box(pybind11::module& m);         /* bindings/box.cpp */
void init_tree(pybind11::module& m);        /* bindings/tree.cpp */
void init_addtree(pybind11::module& m);     /* bindings/addtree.cpp */
void init_featmap(pybind11::module& m);     /* bindings/featmap.cpp */
void init_search(pybind11::module& m);      /* bindings/search.cpp */

#endif
