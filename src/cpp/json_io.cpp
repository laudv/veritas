/**
 * \file json_io.cpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "addtree.hpp"
#include "basics.hpp"
#include "json_io.hpp"
#include "tree.hpp"

#include <iostream>
#include <istream>
#include <sstream>

namespace veritas {

// OLD JSON FORMAT
void tree_from_oldjson(std::istream& s, Tree& tree, NodeId id) {
    std::string buf;
    FloatT leaf_value, split_value;
    FeatId feat_id;
    char c;

    while (s.get(c))
    {
        switch (c)
        {
            case ' ':
            case '\n':
            case '{':
            case ',':
                break;
            case '"':
            case '\'':
                buf.clear();
                while (s.get(c))
                    if (c != '"') buf.push_back(c); else break;
                break;
            case '}':
                goto the_end;
            case ':':
                if (buf == "feat_id")
                    s >> feat_id;
                else if (buf == "split_value")
                    s >> split_value;
                else if (buf == "leaf_value") {
                    s >> leaf_value;
                    tree.leaf_value(id, 0) = leaf_value;
                }
                else if (buf == "lt") { // left branch 
                    tree.split(id, {feat_id, split_value});
                    tree_from_oldjson(s, tree, tree.left(id));
                }
                else if (buf == "gteq") { // expected after lt (split already applied)
                    tree_from_oldjson(s, tree, tree.right(id));
                }
                else
                    throw std::runtime_error("tree parse error: unknown key");
                break;
            default:
                throw std::runtime_error("tree parse error: unexpected char");
        }
    }

    the_end: return;
}

AddTree addtree_from_oldjson(std::istream& s) {
    AddTree at(1, AddTreeType::REGR);

    std::string buf;
    char c;
    Tree tree(1);

    while (s.get(c)) {
        switch (c) {
            case ' ':
            case '\n':
            case ',':
            case '{':
                break;
            case '"':
            case '\'':
                buf.clear();
                while (s.get(c))
                    if (c != '"') buf.push_back(c); else break;
                break;
            case ':':
                if (buf == "base_score")
                    s >> at.base_score(0);
                else if (buf == "trees")
                    goto loop2;
                else
                    throw std::runtime_error("addtree parse error: unknown key");
                break;
            default:
                throw std::runtime_error("addtree parse error: unexpected char");
        }
    }

    loop2: while (s.get(c)) {
        switch (c) {
            case ' ':
            case ']':
            case '}':
            case '\n':
                break;
            case '[':
            case ',': {
                Tree& t = at.add_tree();
                tree_from_oldjson(s, t, t.root());
                break;
            }
            default:
                throw std::runtime_error("addtree parse error (2): unexpected char");
        }
    }
    return at;
} 

} // namespace veritas
