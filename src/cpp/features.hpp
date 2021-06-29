/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_FEAT_MAP_HPP
#define VERITAS_FEAT_MAP_HPP

#include "basics.hpp"
#include "new_tree.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <iostream>

namespace veritas {

    class FeatMap {
        const std::vector<std::string> names_;
        std::unordered_map<const std::string&, FeatId> internal_ids_;

        // disjoint set data structure mapping internal id -> feat id
        std::vector<FeatId> feat_ids_;

        using IterValueType = std::tuple<const std::string&, FeatId>;

    public:

        template <typename ...Args>
        explicit
        FeatMap(Args... feature_names) : names_(feature_names...)
        {
            FeatId internal_count = 0;
            for (const std::string& name : names_)
            {
                internal_ids_[name] = internal_count;
                feat_ids_[internal_count] = internal_count;
            }
        }

        size_t num_features() const { return names_.size(); }

        AddTree transform(const AddTree& at) const
        {
            return {};
        }

        /*
        // https://en.cppreference.com/w/cpp/iterator/iterator
        // member typedefs provided through inheriting from std::iterator
        class iterator: public std::iterator<
                            std::input_iterator_tag,   // iterator_category
                            IterValueType, // value_type
                            size_t,                      // difference_type
                            const IterValueType*, // pointer
                            IterValueType                       // reference
                                          >
        {

            int iid_;

        public:
            explicit iterator(int iid) : iid_(iid) {}
            iterator& operator++() {iid_ += 1; return *this;}
            //iterator operator++(int) {iterator retval = *this; ++(*this); return retval;}
            bool operator==(iterator other) const {return iid_ == other.iid_;}
            bool operator!=(iterator other) const {return !(*this == other);}
            reference operator*() const {

            }
        };

        decltype(FeatMap::feat2id_)::const_iterator
        begin() const { return feat2id_.begin(); }

        decltype(FeatMap::feat2id_)::const_iterator
        end() const { return feat2id_.end(); }
        */

        //Feature operator[](const std::string& feat_name);
    private:

        // https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        FeatId uf_find(FeatId internal_id)
        {
            // function Find(x) is
            //     while x.parent ≠ x do
            //         x.parent := x.parent.parent
            //         x := x.parent
            //     end while
            //     return x
            // end function

            while (feat_ids_[internal_id] != internal_id)
            {
                feat_ids_[internal_id] = feat_ids_[feat_ids_[internal_id]];
                internal_id = feat_ids_[internal_id];
            }
            return internal_id;
        }

        void uf_union(FeatId internal_id1, FeatId internal_id2)
        {
            // function Union(x, y) is
            //     // Replace nodes by roots
            //     x := Find(x)
            //     y := Find(y)

            //     if x = y then
            //         return  // x and y are already in the same set
            //     end if

            //     // If necessary, rename variables to ensure that
            //     // x has at least as many descendants as y
            //     if x.size < y.size then
            //         (x, y) := (y, x)
            //     end if

            //     // Make x the new root
            //     y.parent := x
            //     // Update the size of x
            //     x.size := x.size + y.size
            // end function

            FeatId x = uf_find(internal_id1);
            FeatId y = uf_find(internal_id2);

            if (x == y) return;

            feat_ids_[x] = y;
        }

    };

    /*class Feature {
        FeatMap& map_;
        FeatId feat_id_;

    public:
        Feature(const FeatMap& map) : map_(map), feat_id_(0)
        {

        }
    };

    Feature
    FeatMap::operator[](const std::string& feat_name) const
    {
        return {*this};
    }*/

    //std::ostream& operator<<(std::ostream& s, const FeatMap& fm)
    //{
    //    s << "FeatMap {" << std::endl;
    //    for (auto&& [f, id] : fm)
    //        std::cout << "   - `" << f << "` -> " << id << std::endl;
    //    s << '}';
    //    return s;
    //}
}

#endif // VERITAS_FEAT_MAP_HPP
