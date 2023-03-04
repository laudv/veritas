/**
 * \file jsonio.hpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_JSONIO_HPP
#define VERITAS_JSONIO_HPP

#include <tree.hpp>
#include <addtree.hpp>
#include <iostream>

namespace veritas {

template <typename SplitT, typename ValueT>
void tree_to_json(std::ostream& s, const GTree<SplitT, ValueT>& t);

template <typename TreeT>
TreeT tree_from_json(std::istream& s);

template <typename TreeT>
void addtree_to_json(std::ostream& s, const GAddTree<TreeT>& t);

template <typename AddTreeT>
AddTreeT addtree_from_json(std::istream& s);

AddTree addtree_from_oldjson(std::istream& s);

} // namespace veritas

#endif // VERITAS_JSONIO_HPP
