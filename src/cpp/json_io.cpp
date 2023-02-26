/**
 * \file json_io.cpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#include "basics.hpp"
#include "json_io.hpp"
#include "tree.hpp"

#include <iostream>
#include <istream>
#include <sstream>

namespace veritas {

namespace json_detail {

template <typename ValueT>
struct EncDec {};

struct JsonStream {
    std::istream& s;
    size_t line_count;
    std::string key;
    std::stringstream msg;

    JsonStream(std::istream& s) : s{s}, line_count{1}, key{}, msg{} {}

    bool next_char(char& c) {
        while (s.get(c)) {
            switch (c) {
            case ' ': break;
            case '\n': ++line_count; break;
            default:
                return c;
            }
        }
        throw std::runtime_error("unexpected eof");
    }

    template <typename T>
    bool operator>>(T& v) {
        s >> v;
        return !s.fail();
    }

    bool read_quoted_string(std::string& buf) {
        char c;
        while (next_char(c)) {
            switch (c) {
            case '{': break; // ignore before opining quote (e.g. after comma)
            case '"': // opening quote, read until closing quote
                while (s.get(c)) { // don't use next_char here, we want everything
                    if (c == '"') return true;
                    else buf.push_back(c);
                }
                return false;
            default:
                msg << "read_quoted_string: invalid char '" << c << "': ";
                return false;
            }
        }
        return false;
    }

    bool read_key() {
        key.clear();
        char c;
        while (next_char(c)) {
            switch (c) {
            case ':':
            case '}':
            case '[': break;
            case '{':
            case ',': // validate key
                return read_quoted_string(key);
            default:
                msg << "read_key: invalid char '" << c << "': ";
                return false;
            }
        }
        return false;
    }

    void expect_key(const char *expected_key) {
        if (!read_key()) {
            msg << "error reading key " << key;
            throw_msg();
        }
        if (key != expected_key) {
            msg << "expected key " << expected_key << ", got " << key;
            throw_msg();
        }
    }

    template <typename F>
    bool read_value(F f) {
        char c;
        while (next_char(c)) {
            switch (c) {
            case ' ':
            case '\n':
                break;
            case ':': // read value
                return f(*this);
            case '{':
            case ',': // should see comma or colon here
            default:
                return false;
            }
        }
        return false;
    }

    template <typename T>
    bool read_encdec_value(T& buf) {
        return read_value([&buf](JsonStream& s) {
            return EncDec<T>().decode(s, buf);
        });
    }

    template <typename T>
    void expect_encdec_value(T& buf, const char *msg = "") {
        if (!read_encdec_value(buf)) {
            this->msg << "exected " << EncDec<T>().name();
            if (msg) this->msg << ": " << msg;
            throw_msg();
        }
    }

    void throw_msg() {
        msg << " (line " << line_count << ")";
        throw std::runtime_error(msg.str());
    }
};

template <>
struct EncDec<FloatT> {
    const char *name() const { return "FloatT"; }
    void encode(std::ostream& s, const FloatT& f) const {
        s << f;
    }
    bool decode(JsonStream& s, FloatT& value) const {
        return s >> value;
    }
};

template <>
struct EncDec<size_t> {
    const char *name() const { return "size_t"; }
    void encode(std::ostream& s, const size_t& f) const {
        s << f;
    }
    bool decode(JsonStream& s, size_t& value) const {
        return s >> value;
    }
};

template <>
struct EncDec<FpT> {
    const char *name() const { return "FpT"; }
    void encode(std::ostream& s, const FpT& f) const {
        s << f;
    }
    bool decode(JsonStream& s, FpT& value) const {
        return s >> value;
    }
};

template <>
struct EncDec<std::string> {
    const char *name() const { return "std::string"; }
    void encode(std::ostream& s, const std::string& f) const {
        s << '"' << f << '"';
    }
    bool decode(JsonStream& s, std::string& buf) const {
        return s.read_quoted_string(buf);
    }
};

template <>
struct EncDec<LtSplit> {
    const char *name() const { return "LtSplit"; }
    void encode(std::ostream& s, const LtSplit& split) const {
        s << "{\"feat_id\": " << split.feat_id
          << ", \"split_value\": " << split.split_value
          << '}';
    }
    bool decode(JsonStream& s, LtSplit& split) const {
        if (!s.read_key()) return false;
        if (s.key != "feat_id") return false;
        if (!s.read_encdec_value(split.feat_id)) return false;
        if (!s.read_key()) return false;
        if (s.key != "split_value") return false;
        if (!s.read_encdec_value(split.split_value)) return false;
        return true;
    }
};

template <>
struct EncDec<LtSplitFp> {
    const char *name() const { return "LtSplitFp"; }
    void encode(std::ostream& s, const LtSplitFp& split) const {
        s << "{\"feat_id\": " << split.feat_id
          << ", \"split_value\": " << split.split_value
          << '}';
    }

    bool decode(JsonStream& s, LtSplitFp& split) const {
        if (!s.read_key()) return false;
        if (s.key != "feat_id") return false;
        if (!s.read_encdec_value(split.feat_id)) return false;
        if (!s.read_key()) return false;
        if (s.key != "split_value") return false;
        if (!s.read_encdec_value(split.split_value)) return false;
        return true;
    }
};

} // namespace json_detail

template <typename SplitT, typename ValueT>
static void tree_to_json(std::ostream &s, const GTree<SplitT, ValueT> &t,
                         NodeId id, int depth = 0) {
    std::string indent(2*depth, ' ');
    if (t.is_leaf(id)) {
        s << "{\"leaf_value\": ";
        json_detail::EncDec<ValueT>().encode(s, t.leaf_value(id));
        s << '}';
    } else {
        s << '{' << '\n' << indent << "\"split\": ";
        json_detail::EncDec<SplitT>().encode(s, t.get_split(id));
        s << ",\n" << indent << "\"left\": ";
        tree_to_json(s, t, t.left(id), depth+1);
        s << ",\n" << indent << "\"right\": ";
        tree_to_json(s, t, t.right(id), depth+1);
        s << '}';
    }
}

template <typename SplitT, typename ValueT>
void tree_to_json(std::ostream& s, const GTree<SplitT, ValueT>& t) {
    // meta data about SplitT and ValueT
    s << '{';
    s << "\"split_type\": "
      << '"' << json_detail::EncDec<SplitT>().name() << '"'
      << ", \"value_type\": "
      << '"' << json_detail::EncDec<ValueT>().name() << '"'
      << ",\n  \"structure\": ";

    tree_to_json(s, t, t.root(), 2);

    s << "\n}";
}

template void tree_to_json(std::ostream& s, const Tree& t);
template void tree_to_json(std::ostream& s, const TreeFp& t);
template void tree_to_json(std::ostream& s, const GTree<LtSplit, std::string>& t);

template <typename TreeT>
static void tree_from_json(json_detail::JsonStream& s, TreeT& tree, NodeId id) {
    std::string buf;
    if (!s.read_key()) {
        s.msg << "error reading key 'leaf_value' or 'split'";
        s.throw_msg();
    }
    if (s.key == "leaf_value") {
        s.expect_encdec_value(tree.leaf_value(id), "leaf_value");
    } else if (s.key == "split") {
        typename TreeT::SplitType split;
        s.expect_encdec_value(split, "split");
        tree.split(id, split);
        s.expect_key("left");
        tree_from_json(s, tree, tree.left(id));
        s.expect_key("right");
        tree_from_json(s, tree, tree.right(id));
    } else {
        s.msg << "expected key 'leaf_value' or 'split'";
        s.throw_msg();
    }
}

template <typename TreeT>
static TreeT tree_from_json(json_detail::JsonStream& s) {
    using SplitT = typename TreeT::SplitType;
    using ValueT = typename TreeT::ValueType;

    TreeT tree;

    std::string buf;
    s.expect_key("split_type");
    s.expect_encdec_value(buf, "split_type");
    if (buf != json_detail::EncDec<SplitT>().name())
        throw std::runtime_error("invalid split type");

    buf.clear();
    s.expect_key("value_type");
    s.expect_encdec_value(buf);
    if (buf != json_detail::EncDec<ValueT>().name())
        throw std::runtime_error("invalid split type");

    buf.clear();
    s.expect_key("structure");
    s.read_value([&tree](json_detail::JsonStream &s) {
        tree_from_json(s, tree, tree.root());
        return false; // ignored, we throw anyway
    });

    return tree;
}
template <typename TreeT>
TreeT tree_from_json(std::istream& std_s) {
    json_detail::JsonStream s(std_s);
    return tree_from_json<TreeT>(s);
}

template Tree tree_from_json(std::istream& s);
template TreeFp tree_from_json(std::istream& s);
template GTree<LtSplit, std::string> tree_from_json(std::istream& s);

template <typename TreeT>
void addtree_to_json(std::ostream& s, const GAddTree<TreeT>& at) {
    s << "{\"base_score\": ";
    json_detail::EncDec<typename TreeT::ValueType>().encode(s, at.base_score);
    s << "{\"num_trees\": ";
    json_detail::EncDec<size_t>().encode(s, at.size());
    s << ", \"trees\": [\n";
    for (size_t i = 0; i < at.size(); ++i)
    {
        if (i != 0)
            s << ',' << std::endl;
        tree_to_json(s, at[i]);
    }
    s << "\n]}";
}

template void addtree_to_json(std::ostream &s, const AddTree &t);
template void addtree_to_json(std::ostream &s, const AddTreeFp &t);

template <typename AddTreeT>
AddTreeT addtree_from_json(std::istream &std_s) {
    using TreeT = typename AddTreeT::TreeType;
    json_detail::JsonStream s(std_s);
    AddTreeT at;

    s.expect_key("base_score");
    s.expect_encdec_value(at.base_score, "base_score");
    s.expect_key("num_trees");
    size_t num_trees;
    s.expect_encdec_value(num_trees, "num_trees");
    s.expect_key("trees");

    for (size_t i = 0; i < num_trees; ++i) {
        at.add_tree(tree_from_json<TreeT>(s));
    }

    return at;
}

template AddTree addtree_from_json(std::istream &s);
template AddTreeFp addtree_from_json(std::istream &s);

} // namespace veritas
